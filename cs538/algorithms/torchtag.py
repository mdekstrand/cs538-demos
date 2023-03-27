"""
Implementation of a PyTorch-based recommender.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import NamedTuple
from tqdm.auto import tqdm
import math
from scipy.sparse import csr_matrix, triu
import numpy as np
import pandas as pd
from numba import njit

from csr import CSR
import seedbank

from sklearn.utils import shuffle
import torch
from torch import nn
from torch.optim import AdamW
from torch.linalg import vecdot

from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings, sampling
from lenskit import util
from lenskit.algorithms.bias import Bias

# I want a logger for information
_log = logging.getLogger(__name__)


class ItemTags(NamedTuple):
    """
    Item tags suitable for input to an EmbeddingBag.
    """

    tag_ids: torch.Tensor
    offsets: torch.Tensor

    @classmethod
    def from_items(cls, matrix: CSR, items: np.ndarray):
        # pick_rows gets a subset of the CSR with the specified rows.
        # its row pointers and column indexes are exactly what the embedding
        # matrix needs.
        tmat = matrix.pick_rows(items.ravel(), include_values=False)
        return cls(torch.from_numpy(tmat.colinds), torch.from_numpy(tmat.rowptrs))

    def to(self, dev):
        return ItemTags(self.tag_ids.to(dev), self.offsets.to(dev))


class MFBatch(NamedTuple):
    "Representation of a single batch."

    "The user IDs (B,1)"
    users: torch.Tensor
    "The item IDs (B,2); column 0 is positive, 1 negative"
    items: torch.Tensor
    "The item tags"
    item_tags: ItemTags

    "The batch size"
    size: int

    def to(self, dev):
        "move this batch to a device"
        return self._replace(
            users=self.users.to(dev),
            items=self.items.to(dev),
            item_tags=self.item_tags.to(dev),
        )


@dataclass
class SampleEpochData:
    """
    Permuted data for a single epoch of sampled training.
    """

    data: TagMFTrainData
    permutation: np.ndarray

    @property
    def n_samples(self):
        return self.data.n_samples

    @property
    def batch_size(self):
        return self.data.batch_size

    @property
    def batch_count(self):
        return math.ceil(self.n_samples / self.batch_size)

    def batch(self, batchno: int) -> MFBatch:
        start = batchno * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        size = end - start

        # find the rows for this sample
        rows = self.permutation[start:end]

        # get user IDs
        uv = self.data.uinds[rows].reshape((size, 1))

        # we will get a pair of items for each user - initialize array
        iv = np.empty((size, 2), dtype='int32')
        # get positive item IDs
        iv[:, 0] = self.data.matrix.colinds[rows]
        # get negative item IDs
        # it only works with vectors, not matrices, of user ids, so get column
        iv[:, 1], scts = sampling.neg_sample(self.data.matrix, uv[:, 0], sampling.sample_unweighted)
        # quick debug check
        if np.max(scts) > 7:
            _log.info('%d triples took more than 7 samples', np.sum(scts > 5))

        # get the tags
        item_tags = ItemTags.from_items(self.data.tag_matrix, iv)
        
        # we're done, send to torch and return
        return MFBatch(torch.from_numpy(uv), torch.from_numpy(iv), item_tags, size)


@dataclass
class TagMFTrainData:
    """
    Class capturing MF training data/context
    """

    # the user-item matrix
    matrix: CSR
    # the user IDs for each element of the CSR
    uinds: np.ndarray
    # the item-tag matrix
    tag_matrix: CSR

    batch_size: int

    @property
    def n_samples(self):
        return self.matrix.nnz

    @property
    def n_users(self):
        return len(self.users)
    
    @property
    def n_items(self):
        return len(self.items)

    def for_epoch(self, rng: np.random.Generator) -> SampleEpochData:
        perm = rng.permutation(self.n_samples)
        return SampleEpochData(self, perm)


class TagMFNet(nn.Module):
    """
    Torch module that defines the matrix factorization model.

    Args:
        n_users(int): the number of users
        n_items(int): the number of items
        n_feats(int): the embedding dimension
    """
    def __init__(self, n_users, n_items, n_tags, n_feats, reg):
        super().__init__()
        self.n_feats = n_feats
        self.n_users = n_users
        self.n_items = n_items
        self.n_tags = n_tags

        if isinstance(reg, float):
            self.ub_reg = self.ib_reg = self.p_reg = self.q_reg = reg
        elif len(reg) == 2:
            ureg, ireg = reg
            self.ub_reg = self.p_reg = ureg
            self.ib_reg = self.q_reg = ireg
        elif len(reg) == 4:
            self.ub_reg, self.ib_reg, self.p_reg, self.q_reg = reg
        else:
            raise ValueError('invalid regularization term')
        
        # global bias term
        self.g_bias = nn.Parameter(torch.as_tensor(0.0))
        
        # user and item bias terms
        self.u_bias = nn.Embedding(n_users, 1)
        self.i_bias = nn.Embedding(n_items, 1)

        # user and item embeddings
        self.u_embed = nn.Embedding(n_users, n_feats)
        self.i_embed = nn.Embedding(n_items, n_feats)

        # tag embeddings - multiple tags per item, so we need EmbeddingBag
        self.t_embed = nn.EmbeddingBag(n_tags, n_feats)

        # rescale all initial values for better starting point
        # they started out as standard normals, those are pretty big
        self.u_bias.weight.data.mul_(0.05)
        self.i_bias.weight.data.mul_(0.05)
        self.u_embed.weight.data.mul_(0.05)
        self.i_embed.weight.data.mul_(0.05)
        self.t_embed.weight.data.mul_(0.05)

    def forward(self, user, item, item_tags):
        # look up biases and embeddings
        ub = self.u_bias(user).reshape(user.shape)
        ib = self.i_bias(item).reshape(item.shape)

        uvec = self.u_embed(user)
        ivec = self.i_embed(item)

        # Get tag embeddings from the embedding bag
        it_in, it_off = item_tags
        tvec = self.t_embed(it_in, it_off)
        # embedding bags only support 1D inputs, so we received the
        # item tag data raveled (items stacked atop each other).
        # reshape this so that the items are the right shape
        tvec = tvec.reshape(ivec.shape)
        
        # Sum item and tag vectors to a combined item embedding
        itvec = ivec + tvec

        # compute the inner score
        score = ub + ib + vecdot(uvec, itvec)
        
        # we're done
        return score


class TorchTagMF(Predictor):
    """
    Implementation of a tag-aware hybrid MF in PyTorch.
    """

    _configured_device = None
    _current_device = None

    def __init__(self, n_features, *, batch_size=1024, epochs=5, reg=0.001, device=None, rng_spec=None):
        """
        Initialize the Torch MF predictor.

        Args:
            n_features(int):
                The number of latent features (embedding size).
            batch_size(int):
                The batch size for training.  Since this model is relatively simple,
                large batch sizes work well.
            reg(float):
                The regularization term to apply in AdamW weight decay.
            epochs(int):
                The number of training epochs to run.
            rng_spec:
                The random number specification.
        """
        self.n_features = n_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg = reg
        self.rng_spec = rng_spec

        self._configured_device = device

    def fit(self, ratings, *, tags, **kwargs):
        """
        Fit the model.  This needs tags - call it with::

            mldata = MovieLens('data/ml-25m')
            tags = mldata.tags
            algo = TorchMF(...)
            algo = Recommender.adapt(algo)
            algo.fit(ratings, tags=tags)
        """
        # run the iterations
        timer = util.Stopwatch()
        
        _log.info('[%s] preparing input data set', timer)
        self._prepare_data(ratings, tags)

        dev = self._device
        if dev is None:
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._prepare_model(dev)

        # now _data has the training data, and __model has the trainable model

        for epoch in range(self.epochs):
            _log.info('[%s] beginning epoch %d', timer, epoch + 1)
        
            self._fit_iter()

            unorm = torch.linalg.norm(self._model.u_embed.weight.data).item()
            inorm = torch.linalg.norm(self._model.i_embed.weight.data).item()
            _log.info('[%s] epoch %d finished (|P|=%.3f, |Q|=%.3f, b=%.3f)',
                        timer, epoch + 1, unorm, inorm, self._model.g_bias.data.item())

        _log.info('finished training')
        self._finalize()
        self._cleanup()
        return self

    def _prepare_data(self, ratings, tags):
        "Set up a training data structure for the MF model"
        # index users and items
        rmat, users, items = sparse_ratings(ratings)

        # normalize by removing the global mean
        rvs = np.require(ratings['rating'], 'f4')
        mean = np.mean(rvs)
        rvs -= mean

        # count tag applications - how many times each tag is applied to each item
        tag_freq = tags.groupby(['item', 'tag'], observed=True)['user'].count()
        # normalize to proportions
        tag_freq = tag_freq / tag_freq.groupby('item').sum()
        # convert and sort DF, remember for future use
        tag_df = tag_freq.to_frame('frequency').reset_index()
        tag_df = tag_df.astype({'tag': 'category'}).sort_values(['item', 'tag'])
    
        # create a sparse matrix of item-tag frequencies
        # get item numbers (row numbers)
        tag_inos = items.get_indexer(tag_df['item']).astype('i4')
        # filter in case we have tags for an unrated movie
        tag_df = tag_df[tag_inos >= 0]
        tag_inos = tag_inos[tag_inos >= 0]
        tag_ids = tag_df['tag'].cat.codes.values.astype('i4')
        tag_vals = tag_df['frequency'].values.astype('f4')
        
        # make the CSR
        # shape is necessary b/c some items might not have tags
        shape = (len(items), len(tag_df['tag'].cat.categories))
        tag_mat = CSR.from_coo(tag_inos, tag_ids, tag_vals, shape)

        # create the trainign data structure
        data = TagMFTrainData(rmat, tag_mat, self.batch_size)
        self._data = data
        self.user_index_ = users
        self.item_index_ = items
        self.item_tags_ = tag_mat
        self.global_bias_ = mean
        return data

    def _prepare_model(self, train_dev=None):
        n_users = len(self.user_index_)
        n_items = len(self.item_index_)
        n_tags = self.item_tags_.ncols
        self._rng = seedbank.numpy_rng(self.rng_spec)
        model = TagMFNet(n_users, n_items, n_tags, self.n_features)
        self._model = model
        if train_dev:
            _log.info('preparing to train on %s', train_dev)
            self._train_dev = train_dev
            # initialize model to global mean
            self._model.g_bias.data = torch.as_tensor(np.mean(self._data.r_vals))
            # move device to model
            self._model = model.to(train_dev)
            # put model in training mode
            self._model.train(True)
            # set up training features
            self._loss = nn.MSELoss()
            self._opt = AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.reg)

    def _finalize(self):
        "Finalize model training"
        self._model.eval()

    def _cleanup(self):
        "Clean up data not needed after training"
        del self._data
        del self._loss, self._opt
        del self._rng

    def to(self, device):
        "Move the model to a different device."
        self._model.to(device)
        self._current_device = device
        return self

    def _fit_iter(self):
        """
        Run one iteration of the recommender training.
        """
        n = self._data.n_samples
        # permute the training data
        epoch_data = self._data.for_epoch(self._rng)
        loop = tqdm(range(epoch_data.batch_count))
        for i in loop:
            batch = epoch_data.batch(i).to(self._current_device)

            # compute scores and loss
            pred = self._model(batch.users, batch.items, batch.tags)
            pred_loss = self._loss(pred, batch.ratings)
            # add regularization loss
            reg_loss = self._model.reg_loss()
            loss = pred_loss + reg_loss

            # update model
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

            loop.set_postfix_str('loss: {:.3f}'.format(pred_loss.item()))

            _log.debug('batch %d has loss %s', i, loss.item())
        
        loop.clear()
        
    def predict_for_user(self, user, items, ratings=None):
        """
        Generate item scores for a user.

        This needs to do two things:

        1. Look up the user's ratings (because ratings is usually none)
        2. Score the items using them

        Note that user and items are both user and item IDs, not positions.
        """

        # convert user and items into rows and columns
        u_row = self.user_index_.get_loc(user)
        u_tensor = torch.IntTensor([u_row])
        
        i_cols = self.item_index_.get_indexer(items)
        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable = items[i_cols >= 0]
        i_cols = i_cols[i_cols >= 0]
        i_tensor = torch.from_numpy(i_cols)

        t_info = ItemTags.from_items(self.item_tags_, i_cols)
        if self._train_dev:
            u_tensor = u_tensor.to(self._train_dev)
            i_tensor = i_tensor.to(self._train_dev)
            t_info = t_info.to(self._train_dev)

        # get scores
        with torch.inference_mode():
            scores = self._model(u_tensor, i_tensor, t_info).to('cpu')
            scores += self.global_bias_
        
        # and we can finally put in a series to return
        results = pd.Series(scores, index=scorable)
        return results.reindex(items)  # fill in missing values with nan

    
    def __str__(self):
        return 'TorchTagMF(features={}, reg={})'.format(self.n_features, self.reg)


    def __getstate__(self):
        state = dict(self.__dict__)
        if '_model' in state:
            del state['_model']
            state['_model_weights_'] = self._model.state_dict()
        if '_current_device' in state:
            # we always go back to CPU in pickling
            del state['_current_device']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_model_weights_' in state:
            self._prepare_model()
            self._model.load_state_dict(self._model_weights_)
            # put model in evaluation mode
            self._model.eval()
            del self._model_weights_