"""
Implementation of a PyTorch-based implicit matrix factorization recommender.
"""

import logging
from dataclasses import dataclass, replace
from typing import NamedTuple
from tqdm.auto import tqdm
import math
from scipy.sparse import csr_matrix, triu
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.linalg import vecdot

import seedbank
from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings, sampling
from lenskit import util
from lenskit.algorithms.bias import Bias

# I want a logger for information
_log = logging.getLogger(__name__)


class MFBatch(NamedTuple):
    "Representation of a single batch"
    users: torch.Tensor
    items: torch.Tensor
    rates: torch.Tensor

    def to(self, dev):
        "move this batch to a device"
        return self._replace(**{
            n: v.to(dev)
            for (n, v) in self._asdict().items()
        })


@dataclass
class MFEpochData:
    """
    Permuted data for a single epoch, already moved to Torch.
    """

    data: object
    permutation: np.ndarray
    t_users: torch.Tensor
    t_items: torch.Tensor
    t_rates: torch.Tensor

    @property
    def n_samples(self):
        return self.data.n_samples

    @property
    def batch_size(self):
        return self.data.batch_size

    @property
    def batch_count(self):
        return math.ceil(self.n_samples / self.batch_size)

    def batch(self, num) -> MFBatch:
        start = num * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        rows = self.permutation[start:end]
        ut = self.t_users[rows]
        it = self.t_items[rows]
        rt = self.t_rates[rows]
        return MFBatch(ut, it, rt)


@dataclass
class MFTrainData:
    """
    Class capturing MF training data/context
    """

    users: pd.Index
    items: pd.Index

    r_users: np.ndarray
    r_items: np.ndarray
    r_rates: np.ndarray

    batch_size: int

    @property
    def n_samples(self):
        return len(self.r_users)

    @property
    def n_users(self):
        return len(self.users)
    
    @property
    def n_items(self):
        return len(self.items)

    def for_epoch(self, rng: np.random.Generator) -> MFEpochData:
        perm = rng.permutation(self.n_samples)
        ut = torch.from_numpy(self.r_users)
        it = torch.from_numpy(self.r_items)
        rt = torch.from_numpy(self.r_rates)
        return MFEpochData(self, perm, ut, it, rt)


class MFNet(nn.Module):
    """
    Torch module that defines the matrix factorization model.

    Args:
        n_users(int): the number of users
        n_items(int): the number of items
        n_feats(int): the embedding dimension
    """
    def __init__(self, n_users, n_items, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.n_users = n_users
        self.n_items = n_items

        # user and item bias terms
        self.u_bias = nn.Embedding(n_users, 1)
        self.i_bias = nn.Embedding(n_items, 1)

        # user and item embeddings
        self.u_embed = nn.Embedding(n_users, n_feats)
        self.i_embed = nn.Embedding(n_items, n_feats)

        # rescale all initial values for better starting point
        # they started out as standard normals, those are pretty big
        self.u_bias.weight.data.mul_(0.05)
        self.u_bias.weight.data.square_()
        self.i_bias.weight.data.mul_(0.05)
        self.i_bias.weight.data.square_()
        self.u_embed.weight.data.mul_(0.05)
        self.i_embed.weight.data.mul_(0.05)

    def forward(self, user, item):
        # user & item: tensors of user/item row numbers
        # both have length B

        # look up biases and embeddings
        # biases have dimension (N, 1); remove the 1 by reshaping to match user/item inputs
        ub = self.u_bias(user).reshape(user.shape)
        ib = self.i_bias(item).reshape(item.shape)

        uvec = self.u_embed(user)
        ivec = self.i_embed(item)

        # compute the inner product
        ips = vecdot(uvec, ivec)

        # compute final score
        score = ub + ib + ips
        
        # we're done
        return score


class TorchImplicitMFUserMSE(Predictor):
    """
    Implementation of explicit-feedback MF in PyTorch.
    """

    _device = None
    _train_dev = None

    def __init__(self, n_features, *, confweight=40, lr=0.001, epochs=5, reg=0.01, device=None, rng_spec=None):
        """
        Initialize the Torch MF predictor.

        Args:
            n_features(int):
                The number of latent features (embedding size).
            confweight(float):
                The confidence weight for implicit feedback.
            lr(float):
                The learning rate.
            reg(float):
                The regularization term, used as an AdamW weight decay.
            epochs(int):
                The number of training epochs to run.
            rng_spec:
                The random number specification.
        """
        self.n_features = n_features
        self.confweight = confweight
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.rng_spec = rng_spec

        self._device = device

    def fit(self, ratings, **kwargs):
        # run the iterations
        timer = util.Stopwatch()
        
        _log.info('[%s] preparing input data set', timer)
        self._prepare_data(ratings)

        dev = self._device
        if dev is None:
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._prepare_model(dev)

        for epoch in range(self.epochs):
            _log.info('[%s] beginning epoch %d', timer, epoch + 1)
        
            self._fit_iter()

            unorm = torch.linalg.norm(self._model.u_embed.weight.data).item()
            inorm = torch.linalg.norm(self._model.i_embed.weight.data).item()
            _log.info('[%s] epoch %d finished (|P|=%.3f, |Q|=%.3f)',
                      timer, epoch + 1, unorm, inorm)

        _log.info('finished training')
        self._finalize()
        self._cleanup()
        return self

    def _prepare_data(self, ratings):
        "Set up a training data structure for the IMF model"
        # all we need is user-item matrix of 1/0
        # don't even need to store 1s
        rmat, users, items = sparse_ratings(ratings[['user', 'item']])

        # save the data for training
        self.user_index_ = users
        self.item_index_ = items
        self.matrix_ = rmat

    def _prepare_model(self, train_dev=None):
        n_users = len(self.user_index_)
        n_items = len(self.item_index_)
        self._rng = seedbank.numpy_rng(self.rng_spec)
        model = MFNet(n_users, n_items, self.n_features)
        self._model = model
        if train_dev:
            _log.info('preparing to train on %s', train_dev)
            self._train_dev = train_dev
            # move device to model
            self._model = model.to(train_dev)
            # set up training features
            self._loss = nn.MSELoss()
            self._opt = AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.reg)

    def _finalize(self):
        "Finalize model training, moving back to the CPU"
        self._model = self._model.to('cpu')
        del self._train_dev

    def _cleanup(self):
        "Clean up data not needed after training"
        del self._loss, self._opt
        del self._rng

    def _fit_iter(self):
        """
        Run one iteration of the recommender training.
        """
        # permute the training data - a permutation over users
        epoch_perm = self._rng.permutation(self.matrix_.nrows)
        # set up a progress bar
        loop = tqdm(range(self.matrix_.nrows))
        for i in loop:
            # create input tensors from the data
            row = epoch_perm[i]
            uv = torch.IntTensor([row])
            iv = torch.arange(self.matrix_.ncols, dtype=torch.int32)
            rv = torch.zeros(len(iv))
            rv[self.matrix_.row_cs(row)] = self.confweight

            uv = uv.to(self._train_dev)
            iv = iv.to(self._train_dev)
            rv = rv.to(self._train_dev)

            # compute scores and loss
            pred = self._model(uv, iv)
            loss = self._loss(pred, rv)

            # update model
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

            loop.set_postfix_str('loss: {:.3f}'.format(loss.item()))

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
        i_cols = self.item_index_.get_indexer(items)
        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable = items[i_cols >= 0]
        i_cols = i_cols[i_cols >= 0]

        u_tensor = torch.from_numpy(np.repeat(u_row, len(i_cols)))
        i_tensor = torch.from_numpy(i_cols)
        if self._train_dev:
            u_tensor = u_tensor.to(self._train_dev)
            i_tensor = i_tensor.to(self._train_dev)

        # get scores
        with torch.no_grad():
            scores = self._model(u_tensor, i_tensor).to('cpu')
        
        # and we can finally put in a series to return
        results = pd.Series(scores, index=scorable)
        return results.reindex(items)  # fill in missing values with nan

    
    def __str__(self):
        return 'TorchMF(features={}, reg={})'.format(self.n_features, self.reg)


    def __getstate__(self):
        state = dict(self.__dict__)
        if '_model' in state:
            del state['_model']
            state['_model_weights_'] = self._model.state_dict()

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_model_weights_' in state:
            self._prepare_model()
            self._model.load_state_dict(self._model_weights_)
            self._model.eval()
            del self._model_weights_