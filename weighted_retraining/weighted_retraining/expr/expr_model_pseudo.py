from typing import Optional, Dict, Any

import torch
import itertools
import numpy as np
from torch import nn, Tensor

from botorch.models import SingleTaskGP
from gpytorch.utils.grid import scale_to_bounds
from botorch.utils.transforms import normalize

import weighted_retraining.weighted_retraining.expr.eq_grammar as eq_gram
from weighted_retraining.weighted_retraining.models_pseudo import BaseVAE, MLPRegressor

masks_t = torch.tensor(eq_gram.masks)
ind_of_ind_t = torch.tensor(eq_gram.ind_of_ind)

MAX_LEN = 15
DIM = eq_gram.D


class EquationMLPRegressor(MLPRegressor):

    def __init__(self, input_dim: int, output_dim: int, *h_dims: int):
        super().__init__(input_dim, output_dim, *h_dims)

    def forward(self, z: Tensor):
        h = super().forward(z)
        return torch.relu(h)


class EquationEncoderTorch(nn.Module):

    def __init__(self, charset_length: int, max_length: int, latent_rep_size: int = 10,
                 dense: int = 100, conv1: int = 2, conv2: int = 3, conv3: int = 4, from_tf: bool = False):
        super(EquationEncoderTorch, self).__init__()

        self.latent_rep_size = latent_rep_size
        self.charset_length = charset_length
        self.max_length = max_length
        self.from_tf = from_tf

        self.dense = dense

        def get_ell_out(ell_in, k_size):
            return ell_in - k_size + 1

        ell = self.max_length
        ell = get_ell_out(ell, conv1)
        ell = get_ell_out(ell, conv2)
        ell = get_ell_out(ell, conv3)

        self.layers = nn.Sequential(
            nn.Conv1d(self.charset_length, conv1, conv1),
            nn.ReLU(),
            nn.BatchNorm1d(conv1, momentum=.01, eps=1e-3),
            nn.Conv1d(conv1, conv2, conv2),
            nn.ReLU(),
            nn.BatchNorm1d(conv2, momentum=.01, eps=1e-3),
            nn.Conv1d(conv2, conv3, conv3),
            nn.ReLU(),
            nn.BatchNorm1d(conv3, momentum=.01, eps=1e-3),
        )
        self.layers_sup = nn.Sequential(nn.Flatten(),
                                        nn.Linear(ell * conv3, dense),
                                        nn.ReLU()
                                        )

        self.fc_mean = nn.Linear(dense, latent_rep_size)
        self.fc_log_std = nn.Linear(dense, latent_rep_size)

    def forward(self, x: Tensor):
        assert x.ndim > 2 and x.shape[-2:] == (self.charset_length, self.max_length), (x.shape,
                                                                                       self.charset_length,
                                                                                       self.max_length)
        h = self.layers(x).transpose(-1, -2)
        if self.from_tf:
            h = h.transpose(-1, -2)
        h = self.layers_sup(h)
        return self.fc_mean(h), self.fc_log_std(h)

    @staticmethod
    def sampling(mu, log_var):
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)  # return z sample

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class EquationDecoderTorch(nn.Module):

    def __init__(self, charset_length: int, max_length: int, latent_rep_size: int = 10,
                 hidden: int = 100):
        super(EquationDecoderTorch, self).__init__()

        self.latent_rep_size = latent_rep_size
        self.charset_length = charset_length
        self.max_length = max_length
        self.hidden_size = hidden

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.latent_rep_size, momentum=.01, eps=1e-3),
            nn.Linear(self.latent_rep_size, self.latent_rep_size),
            nn.ReLU()
        )
        self.gru = nn.Sequential(
            nn.GRU(self.latent_rep_size, hidden, num_layers=3, batch_first=True)
        )

        self.out_fc = nn.Linear(hidden, self.charset_length)

    def forward(self, x: Tensor):
        h = self.fc(x)
        h = h.repeat(*[1 for _ in range(h.ndim - 1)], self.max_length).reshape(
            (*x.shape[:-1], self.max_length, x.shape[-1]))

        out, _ = self.gru(h)  # shape (batch, seq, feature)
        # out = out.permute(*range(1, out.ndim - 1), 0, -1)  # shape (batch, seq, feature)
        out = self.out_fc(out)
        return out

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class EquationVaeTorch(BaseVAE):

    def __init__(self, hparams, charset_length: int, max_length: int = MAX_LEN):
        """
        VAE model for equation reconstruction task

        Args:
            hparams: configurations
            charset_length: number of rules in grammar
            max_length: maximum length of grammar elements sequence
        """
        super().__init__(hparams)

        # Set up encoder and decoder
        self.encoder = EquationEncoderTorch(charset_length=charset_length, max_length=max_length,
                                            latent_rep_size=hparams.latent_dim)

        self.decoder = EquationDecoderTorch(charset_length=charset_length, max_length=max_length,
                                            latent_rep_size=hparams.latent_dim)

        self.target_predictor: Optional[EquationMLPRegressor] = None
        self.gplayer: Optional[SingleTaskGP] = None

        self.pred_loss = nn.MSELoss()
        if self.predict_target:
            self.target_predictor = EquationMLPRegressor(hparams.latent_dim, 1,
                                                         *hparams.target_predictor_hdims)
        self.charset_length = charset_length
        self.max_length = max_length

        # target normalisation constants
        self.training_m = None
        self.training_M = None
        self.validation_m = None
        self.validation_M = None

    def build_target_predictor(self):
        self.target_predictor = EquationMLPRegressor(self.latent_dim, 1,
                                                     *self.target_predictor_hdims)
    
    def build_gp_predictor(self, gplayer: SingleTaskGP, gpkwargs: Dict[str, Any]=None):
        self.gplayer = gplayer
        self.gpkwargs = gpkwargs
        for param in self.gplayer.parameters():
            param.requires_grad = False  
        self.gplayer.eval() 

    def build_bnn_predictor(self, gplayer, gpkwargs: Dict[str, Any]=None):
        self.gplayer = gplayer
        self.gpkwargs = gpkwargs

    def train(self, mode: bool = True):
        super().train(mode)

    def encode_to_params(self, x: Tensor):
        assert x.ndim > 2 and x.shape[-2:] == (self.max_length, self.charset_length), (x.shape,
                                                                                       self.max_length,
                                                                                       self.charset_length)
        mu, logstd = self.encoder(x.transpose(-1, -2))
        return mu, logstd

    def conditional(self, x_true: Tensor, x_pred: Tensor):
        assert x_true.shape[-2:] == x_pred.shape[-2:] == (self.max_length, self.charset_length), (x_true.shape[-2:],
                                                                                                  x_pred.shape[-2:],
                                                                                                  (self.max_length,
                                                                                                   self.charset_length))
        most_likely = torch.argmax(x_true, -1).flatten()
        ix2 = (ind_of_ind_t[most_likely])
        M2_t = masks_t[ix2].reshape(-1, MAX_LEN, DIM).to(x_pred)
        P2_t = torch.exp(x_pred) * M2_t
        P2_t /= torch.sum(P2_t, dim=-1, keepdim=True)
        return P2_t

    def decoder_loss(self, z: Tensor, x_orig: Tensor, reduction='mean'):
        """ return negative Bernoulli log prob """
        x_pred = self.decoder(z)
        probs = self.conditional(x_orig, x_pred)
        ent_loss = self.max_length * nn.BCELoss(reduction=reduction)(probs, x_orig)

        return ent_loss

    def ssdkl_loss(self, z: Tensor, target: Tensor, ulz: Tensor=None, alpha: float=1.0, is_gp: bool = True):
        z = normalize(z, self.gpkwargs['bounds'])
        dist = self.gplayer.posterior(z)
        y_pred = -dist.mean.detach()
        if self.gpkwargs['use_std']:
            y_pred = y_pred * self.gpkwargs['target_std'] + self.gpkwargs['target_mean']
        y_std = dist.variance.detach()
        assert y_pred.shape == target.shape, (y_pred.shape, target.shape)
        ssdkl_loss = (nn.MSELoss(reduce='none')(y_pred, target) / y_std).mean() + torch.log(y_std).mean()
        if ulz is not None:
            ulz = normalize(ulz, self.gpkwargs['bounds'])
            dist_ul = self.gplayer.posterior(ulz)
            y_std_ul = dist_ul.variance.detach()
            ssdkl_loss += alpha * torch.log(y_std_ul).mean()
        return ssdkl_loss

    def ssdkl_loss_bnn(self, z: Tensor, target: Tensor, ulz: Tensor=None, alpha: float=1.0):
        z = normalize(z, self.gpkwargs['bounds'])
        predictions = self.gplayer.predict(z, is_train=True)
        y_pred, y_std = predictions.mean(axis=-1), predictions.std(axis=-1)**2+1e-10

        if self.gpkwargs['use_std']:
            y_pred = y_pred * self.gpkwargs['target_std'] + self.gpkwargs['target_mean']
        assert y_pred.shape == target.shape, (y_pred.shape, target.shape)
        ssdkl_loss = (nn.MSELoss(reduce='none')(y_pred, target) / y_std).mean() + torch.log(y_std).mean()
        if ulz is not None:
            ulz = normalize(ulz, self.gpkwargs['bounds'])
            _, y_std_ul = self.gplayer.predict(ulz, is_train=True)
            ssdkl_loss += alpha * torch.log(y_std_ul).mean()
        return ssdkl_loss

    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        x_pred = self.decoder(z)
        return x_pred

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx, self.training_m, self.training_M)

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx, self.validation_m, self.validation_M)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = BaseVAE.add_model_specific_args(parent_parser)
        return parser

    def configure_optimizers(self):
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
        sched = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.2, patience=20,
                                                                         min_lr=self.hparams.lr / 10),
                 'interval': 'epoch',
                 'monitor': 'loss/val'
                 }
        return dict(optimizer=opt, lr_scheduler=sched)
