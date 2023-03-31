"""
Implementation of a PyTorch-based implicit matrix factorization recommender.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, replace
from typing import NamedTuple, Optional
from csr import CSR
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
import torch.nn.functional as F
from torch.linalg import vecdot

import seedbank
from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings, sampling
from lenskit import util
from lenskit.algorithms.bias import Bias

# I want a logger for information
_log = logging.getLogger(__name__)


class MFBatch(NamedTuple):
    "Representation of a single batch."

    "The user IDs (B,1)"
    users: torch.Tensor
    "The item IDs (B,2); column 0 is positive, 1 negative"
    items: torch.Tensor

    "The batch size"
    size: int

    def to(self, dev):
        "move this batch to a device"
        return self._replace(
            users=self.users.to(dev),
            items=self.items.to(dev),
        )


@dataclass
class SampleEpochData:
    """
    Permuted data for a single epoch of sampled training.
    """

    data: MFTrainData
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
        
        # we're done, send to torch and return
        return MFBatch(torch.from_numpy(uv), torch.from_numpy(iv), size)


@dataclass
class MFTrainData:
    """
    Class capturing MF training data/context
    """

    # the user-item matrix
    matrix: CSR
    # the user IDs for each element of the CSR
    uinds: np.ndarray

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
        _log.debug('uvec shape: %s', uvec.shape)
        ivec = self.i_embed(item)
        _log.debug('ivec shape: %s', ivec.shape)

        # compute the inner product
        ips = vecdot(uvec, ivec)

        # compute final score
        score = ub + ib + ips
        
        # we're done - return the log-odds (inner score)
        return score


def loss_logistic(X: torch.Tensor):
    """
    Logistic loss funcftion for paired predictions.

    This loss function does not require a separate label tensor, because the
    labels are implicit in the structure. :math:`X` has shape (B, 2), where
    column 0 is scores for positive observations and column 1 is scores for
    negative observations.

    Args:
        X(torch.Tensor):
            A tensor of shape (B, 2) storing the prediction scores (in log
            odds).
    
    Returns:
        torch.Tensor:
            A tensor of shape () with the negative log likelihood for the
            prediction scores.
    """
    # X is the log odds of 1, but we need column 1 to be the log odds of 0. If
    # we multiply column 0 by 1, and 1 by -1, we will get a new tensor where
    # each element is the log odds of the corresponding rating value, not the
    # always log odds of 1.  A tensor of shape (1, 2) will broadcast with (B, 2)
    # and give us what we need.
    mult = torch.Tensor([1, -1]).reshape((1, 2)).to(X.device)
    Xlo = X * mult
    
    # Now logsigmoid will convert log odds to log likelihoods
    Xnll = -F.logsigmoid(Xlo)

    # And now we compute the mean negative log likelihood for this batch
    # The total *observations* is n * 2, but since they are always in pairs,
    # dividing by n will suffice to ensure consistent optimization across batches.
    n = X.shape[0]
    return Xnll.sum() / n


def loss_bpr(X: torch.Tensor):
    """
    BPR loss function for paired predictions.

    This loss function does not require a separate label tensor, because the
    labels are implicit in the structure. :math:`X` has shape (B, 2), where
    column 0 is scores for positive observations and column 1 is scores for
    negative observations.

    Args:
        X(torch.Tensor):
            A tensor of shape (B, 2) storing the prediction scores (in log
            odds).
    
    Returns:
        torch.Tensor:
            A tensor of shape () with the negative log likelihood for the
            prediction scores.
    """
    # For a pair (i, j), we have their scores in columns 0 and 1.
    # The BPR scoring formula is the difference in these scores: i - j
    Xscore = X[:, 0] - X[:, 1]
    
    # Now logsigmoid will convert that score to a log likelihood
    Xnll = -F.logsigmoid(Xscore)

    # And now we compute the mean of the negative log likelihoods for this batch
    n = X.shape[0]
    return Xnll.sum() / n


class TorchSampledMF(Predictor):
    """
    Implementation of implicit-feedback matrix factorization with negative sampling.
    """

    _configured_device = None
    _current_device = None
    _data: Optional[MFTrainData]
    _model: Optional[MFNet]

    def __init__(self, n_features, *, loss='logistic', batch_size=8*1024, lr=0.001, epochs=5, reg=0.01, device=None, rng_spec=None):
        """
        Initialize the Torch MF predictor.

        Args:
            n_features(int):
                The number of latent features (embedding size).
            loss(str):
                The loss function to use. Can be either ``'logistic'`` or ``'bpr'``.
            batch_size(int):
                The number of users to use in each training batch.
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
        self.batch_size = batch_size
        self.loss = loss
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.rng_spec = rng_spec

        self._configured_device = device

    def fit(self, ratings, **kwargs):
        # run the iterations
        timer = util.Stopwatch()
        
        _log.info('[%s] preparing input data set', timer)
        self._prepare_data(ratings)

        dev = self._configured_device
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

        # set up the training data
        self._data = MFTrainData(rmat, rmat.rowinds(), self.batch_size)

        # save the index data for final use
        self.user_index_ = users
        self.item_index_ = items

    def _prepare_model(self, train_dev=None):
        n_users = len(self.user_index_)
        n_items = len(self.item_index_)
        self._rng = seedbank.numpy_rng(self.rng_spec)
        model = MFNet(n_users, n_items, self.n_features)
        self._model = model
        if train_dev:
            _log.info('preparing to train on %s', train_dev)
            self._current_device = train_dev
            # move device to model
            self._model = model.to(train_dev)
            # the model needs to be in training mode
            self._model.train(True)
            # set up loss function
            match self.loss:
                case 'logistic':
                    self._loss = loss_logistic
                case 'bpr':
                    self._loss = loss_bpr
                case _:
                    raise ValueError(f'invalid loss {self.loss}')
            # set up optimizer
            self._opt = AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.reg)

    def _finalize(self):
        "Finalize model training"
        # set the model in evaluation mode (not training)
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
        epoch = self._data.for_epoch(self._rng)
        # set up a progress bar
        loop = tqdm(range(epoch.batch_count))
        for i in loop:
            batch = epoch.batch(i).to(self._current_device)

            # compute scores and loss
            pred = self._model(batch.users, batch.items)

            # loss for this model does not require rating values - they are implicit
            # in the data structure layout (column 0 is r=1, column 1 is r=0)
            loss = self._loss(pred)
            
            # update model
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

            if i % 100 == 99:
                _log.debug('batch %d has NLL %s', i, loss.item())
        
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

        u_tensor = torch.IntTensor([u_row])
        i_tensor = torch.from_numpy(i_cols)
        if self._current_device:
            u_tensor = u_tensor.to(self._current_device)
            i_tensor = i_tensor.to(self._current_device)

        # get scores
        with torch.inference_mode():
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
        if '_current_device' in state:
            # we always go back to CPU in pickling
            del state['_current_device']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_model_weights_' in state:
            self._prepare_model()
            self._model.load_state_dict(self._model_weights_)
            # set the model in evaluation mode (not training)
            self._model.eval()
            del self._model_weights_