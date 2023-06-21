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
import torch.nn.functional as F
from torch.linalg import vecdot

import seedbank
from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings, sampling
from lenskit import util
from lenskit.algorithms.bias import Bias

# I want a logger for information
_log = logging.getLogger(__name__)


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


class TorchLogisticMF(Predictor):
    """
    Implementation of explicit-feedback MF in PyTorch.
    """

    _configured_device = None
    _current_device = None

    def __init__(self, n_features, *, confweight=100, batch_size=16, lr=0.001, epochs=5, reg=0.01, device=None, rng_spec=None):
        """
        Initialize the Torch MF predictor.

        Args:
            n_features(int):
                The number of latent features (embedding size).
            confweight(float):
                The confidence weight for implicit feedback.
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
        self.confweight = confweight
        self.batch_size = batch_size
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
            self._current_device = train_dev
            # move device to model
            self._model = model.to(train_dev)
            # the model needs to be in training mode
            self._model.train(True)
            # set up training features
            self._loss = nn.NLLLoss(torch.Tensor([1, self.confweight]).to(train_dev))
            self._opt = AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.reg)

    def _finalize(self):
        "Finalize model training"
        # set the model in evaluation mode (not training)
        self._model.eval()

    def _cleanup(self):
        "Clean up data not needed after training"
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
        # permute the training data - a permutation over users
        nusers = self.matrix_.nrows
        epoch_perm = self._rng.permutation(nusers)
        bc = math.ceil(nusers / self.batch_size)
        # we use the same item vector for each iteration - allocate ocne
        # this is going to be a matrix: one row for each batch element, col for each item
        # we set up the first row, and then we repeat it
        # play with this code in a python terminal to see how the sizes work
        ni = self.matrix_.ncols
        iv = np.arange(ni, dtype='i4')
        iv = np.repeat(iv.reshape((1, -1)), self.batch_size, axis=0)
        ivt = torch.from_numpy(iv).to(self._current_device)
        ivt = ivt.reshape((self.batch_size, 1, ni))
        # set up a progress bar
        loop = tqdm(range(bc))
        for i in loop:
            # create input tensors from the data
            bs = i * self.batch_size
            be = min(bs + self.batch_size, nusers)
            rows = epoch_perm[bs:be]
            cur_size = len(rows)
            if cur_size < self.batch_size:
                iv = iv[:cur_size, :]
                ivt = torch.from_numpy(iv).to(self._current_device)
                ivt = ivt.reshape((cur_size, 1, ni))
            # convert rows to tensor, and reshape
            # (B, 1) will broadcast with the (B, |I|) item index vector
            uv = torch.IntTensor(rows).reshape((-1, 1, 1))
            # rv = torch.zeros_like(ivt, dtype=torch.float32)
            rv = np.zeros((cur_size, 1, ni), dtype='float32')
            for j, row in enumerate(rows):
                rv[j, 0, self.matrix_.row_cs(row)] = 1

            rv = torch.from_numpy(rv)
            rv = rv.to(self._current_device)
            uv = uv.to(self._current_device)

            # compute scores and loss
            pred = self._model(uv, ivt)
            # preds are log odds of 1 for each item
            # need: log prob 0, log prob 1
            # compute: logsigmoid of score/pred
            lprob1 = F.logsigmoid(pred)
            lprob0 = F.logsigmoid(-pred)
            lprob = torch.concat([lprob0, lprob1], axis=1)
            _log.debug('lprob shape: %s', lprob.shape)
            loss = self._loss(lprob, rv.reshape((cur_size, -1)).to(torch.long))

            # update model
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

            # _log.debug('batch %d has loss %s', i, loss.item())
        
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
