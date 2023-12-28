""" code for base VAE model """
import argparse
import math
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as torch_func
from torch import nn, Tensor
from torch.nn import functional as F

from utils.utils_cmd import parse_dict, parse_list

class BaseVAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.save_hyperparameters()
        self.latent_dim: int = hparams.latent_dim
        if not hasattr(hparams, 'predict_target'):  # backward compatibility
            hparams.predict_target = False
        self.predict_target: bool = hparams.predict_target
        if not hasattr(hparams, 'beta_target_pred_loss'):
            hparams.beta_target_pred_loss = 1.
        self.beta_target_pred_loss = hparams.beta_target_pred_loss
        # Register buffers for prior
        self.register_buffer("prior_mu", torch.zeros([self.latent_dim]))
        self.register_buffer("prior_sigma", torch.ones([self.latent_dim]))

        # Create beta
        self.beta = hparams.beta_final
        self.beta_final = hparams.beta_final
        self.beta_annealing = False
        if hparams.beta_start is not None:
            self.beta_annealing = True
            self.beta = hparams.beta_start
            assert (
                hparams.beta_step is not None
                and hparams.beta_step_freq is not None
                and hparams.beta_warmup is not None
            )

        self.logging_prefix = None
        self.log_progress_bar = False

        if not hasattr(hparams, 'use_pseudo'): 
            hparams.use_pseudo = False
        self.use_pseudo: bool = hparams.use_pseudo
        if not hasattr(hparams, 'beta_pseudo_loss'):
            hparams.beta_pseudo_loss = 0.1
        self.beta_pseudo_loss = hparams.beta_pseudo_loss

        if not hasattr(hparams, 'use_ssdkl'):
            hparams.use_ssdkl = False
        self.use_ssdkl: bool = hparams.use_ssdkl
        if not hasattr(hparams, 'beta_ssdkl_loss'):
            hparams.beta_ssdkl_loss = 1
        self.beta_ssdkl_loss = hparams.beta_ssdkl_loss


    @property
    def require_ys(self):
        """ Whether (possibly transformed) target values are required in forward method """
        if self.predict_target or self.use_ssdkl:
            return True

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.register('type', list, parse_list)
        parser.register('type', dict, parse_dict)

        vae_group = parser.add_argument_group("VAE")
        vae_group.add_argument("--latent_dim", type=int, required=True, help='Dimensionality of the latent space')
        vae_group.add_argument("--lr", type=float, default=1e-3, help='Learning rate')
        vae_group.add_argument("--beta_final", type=float, default=1.0, help='Final value for beta')
        vae_group.add_argument(
            "--beta_start",
            type=float,
            default=None,
            help="starting beta value; if None then no beta annealing is used",
        )
        vae_group.add_argument(
            "--beta_step",
            type=float,
            default=None,
            help="multiplicative step size for beta, if using beta annealing",
        )
        vae_group.add_argument(
            "--beta_step_freq",
            type=int,
            default=None,
            help="frequency for beta step, if taking a step for beta",
        )
        vae_group.add_argument(
            "--beta_warmup",
            type=int,
            default=None,
            help="number of iterations of warmup before beta starts increasing",
        )
        vae_group.add_argument(
            "--predict_target",
            action='store_true',
            help="Generative model predicts target value",
        )
        vae_group.add_argument(
            "--target_predictor_hdims",
            type=list,
            default=None,
            help="Hidden dimensions of MLP predicting target values",
        )
        vae_group.add_argument(
            "--beta_target_pred_loss",
            type=float,
            default=1.,
            help="Weight of the target_prediction loss added in the ELBO",
        )

        vae_group = parser.add_argument_group("Pseudo label")
        vae_group.add_argument(
            "--use_pseudo",
            action="store_true",
            help="Wether use the  pseudo label data"
        )
        vae_group.add_argument(
            "--beta_pseudo_loss",
            type=float,
            default=1.,
            help="Weight of the pseudo loss added in the ELBO"
        )
        ssdkl_group = parser.add_argument_group("Deep Kernel Learning")
        ssdkl_group.add_argument(
            "--use_ssdkl",
            action="store_true",
            help="Wether use the  pseudo label data"
        )
        ssdkl_group.add_argument(
            "--beta_ssdkl_loss",
            type=float,
            default=1.,
            help="Weight of the pseudo loss added in the ELBO"
        )
        return parser

    def target_prediction_loss(self, z: Tensor, target: Tensor):
        """ Return MSE loss associated to target prediction

        Args:
            z: latent variable
            target: ground truth score
        """
        y_pred = self.target_predictor(z)
        assert y_pred.shape == target.shape, (y_pred.shape, target.shape)
        pred_loss = self.pred_loss(y_pred, target)
        return pred_loss

    def sample_latent(self, mu, logstd):
        scale_safe = torch.exp(logstd) + 1e-10
        encoder_distribution = torch.distributions.Normal(loc=mu, scale=scale_safe)
        z_sample = encoder_distribution.rsample()
        return z_sample

    def kl_loss(self, mu, logstd, z_sample):
        # Manual formula for kl divergence (more numerically stable!)
        kl_div = 0.5 * (torch.exp(2 * logstd) + mu.pow(2) - 1.0 - 2 * logstd)
        loss = kl_div.sum() / z_sample.shape[0]
        return loss

    def forward(self, *inputs: Tensor, validation: bool = False, m: Optional[float] = None, M: Optional[float] = None):
        """ calculate the VAE ELBO """
        if len(inputs) == 2:
            label_inputs, pseudo_inputs = inputs[0], inputs[1]
            if self.require_ys:
                x, y = label_inputs[:-1], label_inputs[-1]
                if len(label_inputs) == 2:
                    x = x[0]
                elif len(label_inputs) == 1:
                    x, y = label_inputs[0][:-1], label_inputs[0][-1]
            elif len(label_inputs) == 1:
                x = label_inputs[0]
            elif len(label_inputs) == 2:  
                x, y = label_inputs[0], label_inputs[1]
            else:
                x = label_inputs
            if self.require_ys:
                ulx, uly = pseudo_inputs[:-1], pseudo_inputs[-1]
                if len(pseudo_inputs) == 2:
                    ulx = ulx[0]
                elif len(pseudo_inputs) == 1:
                    ulx, uly = pseudo_inputs[0][:-1], pseudo_inputs[0][-1]
            elif len(pseudo_inputs) == 1:
                ulx = pseudo_inputs[0]
            elif len(pseudo_inputs) == 2:
                ulx, uly = pseudo_inputs[0], pseudo_inputs[1]
            else:
                ulx = pseudo_inputs
        else:
            if self.require_ys:
                x, y = inputs[:-1], inputs[-1]
                if len(inputs) == 2:
                    x = x[0]
                elif len(inputs) == 1:
                    x, y = inputs[0][:-1], inputs[0][-1]
            elif len(inputs) == 1:
                x = inputs[0]
            elif len(inputs) == 2:  # e.g. validation step in semi-supervised setup but we have targets that we do not use
                x, y = inputs[0], inputs[1]
            else:
                x = inputs
            ulx = []
            uly = []

        # reparameterization trick
        mu, logstd = self.encode_to_params(x)
        z_sample = self.sample_latent(mu, logstd)

        # KL divergence and reconstruction error
        kl_loss = self.kl_loss(mu, logstd, z_sample)
        reconstruction_loss = self.decoder_loss(z_sample, x)

        # Final loss
        if validation:
            beta = self.beta_final
        else:
            beta = self.beta

        prediction_loss = 0
        if self.predict_target:
            if self.predict_target:
                if y.shape[-1] != 1:
                    y = y.unsqueeze(-1)
            prediction_loss = self.target_prediction_loss(z_sample, target=y)

        pseudo_loss=0
        if self.use_pseudo and len(ulx) > 0:
            # reparameterization trick
            mu_t, logstd_t = self.encode_to_params(ulx)
            z_sample_t = self.sample_latent(mu_t, logstd_t)
            # KL divergence and reconstruction error
            kl_loss_t = self.kl_loss(mu_t, logstd_t, z_sample_t)
            reconstruction_loss_t = self.decoder_loss(z_sample_t, ulx)
            pseudo_loss = reconstruction_loss_t + beta * kl_loss_t 
        

        ssdkl_loss=0
        if self.use_ssdkl and self.gplayer is not None:
            if y.shape[-1] != 1:
                y = y.unsqueeze(-1)
            if len(ulx) > 0:
                ssdkl_loss = self.ssdkl_loss(z_sample, y, z_sample_t)
            else:
                ssdkl_loss = self.ssdkl_loss(z_sample, y)
                        
        loss = 0
        loss = reconstruction_loss \
            + beta * kl_loss \
            + self.beta_target_pred_loss * prediction_loss \
            + self.beta_pseudo_loss * pseudo_loss \
            + self.beta_ssdkl_loss * ssdkl_loss
        
        # Logging
        if self.logging_prefix is not None:
            self.log(
                f"rec/{self.logging_prefix}",
                reconstruction_loss,
                prog_bar=self.log_progress_bar,
            )
            self.log(
                f"kl/{self.logging_prefix}", kl_loss, prog_bar=self.log_progress_bar
            )
            if self.predict_target:
                self.log(
                    f"pred_target/{self.logging_prefix}", prediction_loss, prog_bar=self.log_progress_bar
                )
            if self.use_pseudo:
                self.log(
                    f"pseudo_loss/{self.logging_prefix}", pseudo_loss, prog_bar=self.log_progress_bar
                )
            if self.use_ssdkl:
                self.log(
                    f"ssdkl_loss/{self.logging_prefix}", ssdkl_loss, prog_bar=self.log_progress_bar
                )
            self.log(f"loss/{self.logging_prefix}", loss)
        return loss

    def sample_prior(self, n_samples):
        return torch.distributions.Normal(self.prior_mu, self.prior_sigma).sample(
            torch.Size([n_samples])
        )

    def _increment_beta(self):

        if not self.beta_annealing:
            return

        # Check if the warmup is over and if it's the right step to increment beta
        if (
            self.global_step > self.hparams.beta_warmup
            and self.global_step % self.hparams.beta_step_freq == 0
        ):
            # Multiply beta to get beta proposal
            self.beta = min(self.hparams.beta_final, self.beta * self.hparams.beta_step)

    # Methods to overwrite (ones that differ between specific VAE implementations)
    def encode_to_params(self, x):
        """ encode a batch to it's distributional parameters """
        raise NotImplementedError

    def decoder_loss(self, z: torch.Tensor, x_orig) -> torch.Tensor:
        """ Get the loss of the decoder given a batch of z values to decode """
        raise NotImplementedError

    def ssdkl_loss(self, z: Tensor, target: Tensor, ulz: Tensor=None, alpha: float=1.0):
        raise NotImplementedError
    
    def decode_deterministic(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx, m: Optional[float] = None, M: Optional[float] = None):
        if hasattr(self.hparams, 'cuda') and self.hparams.cuda is not None:
            self.log(f"cuda:{self.hparams.cuda}",
                     pl.core.memory.get_gpu_memory_map()[f'gpu_id: {self.hparams.cuda}/memory.used (MB)'],
                     prog_bar=True)
        self._increment_beta()
        self.log("beta", self.beta, prog_bar=True)
        self.logging_prefix = "train"
        loss = self(*batch, m=m, M=M)
        self.logging_prefix = None
        return loss

    def validation_step(self, batch, batch_idx, m: Optional[float] = None, M: Optional[float] = None):
        if hasattr(self.hparams, 'cuda') and self.hparams.cuda is not None:
            self.log(f"cuda:{self.hparams.cuda}",
                     pl.core.memory.get_gpu_memory_map()[f'gpu_id: {self.hparams.cuda}/memory.used (MB)'],
                     prog_bar=True)
        self.logging_prefix = "val"
        self.log_progress_bar = True
        loss = self(*batch, validation=True, m=m, M=M)
        self.logging_prefix = None
        self.log_progress_bar = False
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
        # No scheduling
        sched = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=.2, patience=1,
                                                                         min_lr=self.hparams.lr),
                 'interval': 'epoch',
                 'monitor': 'loss/val'
                 }
        return dict(optimizer=opt,
                    lr_scheduler=sched)

class UnFlatten(torch.nn.Module):
    """ unflattening layer """

    def __init__(self, filters=1, size=28):
        super().__init__()
        self.filters = filters
        self.size = size

    def forward(self, x):
        return x.view(x.size(0), self.filters, self.size, self.size)


class MLPRegressor(torch.nn.Module):
    """ Simple class for regression """

    def __init__(self, input_dim: int, output_dim: int, *h_dims: int):
        """

        Args:
            input_dim: input dimension
            output_dim: output dimension
            *h_dims: dimensions of the MLP hidden layers
        """
        super(MLPRegressor, self).__init__()
        self.h_dims = list(h_dims)
        layer_dims = [input_dim] + self.h_dims + [output_dim]
        self.layers = torch.nn.ModuleList([nn.Linear(u, v) for u, v in zip(layer_dims[:-1], layer_dims[1:])])

    def forward(self, x: Tensor):
        h = x
        for layer in self.layers[:-1]:
            h = torch_func.relu(layer(h))
        return self.layers[-1](h)
    