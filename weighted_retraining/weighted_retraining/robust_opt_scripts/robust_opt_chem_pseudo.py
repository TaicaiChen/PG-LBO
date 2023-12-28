import argparse
import logging
import os
import gc
import shutil
import subprocess
import sys
import time
import traceback
import copy
from pathlib import Path
from typing import Dict, Any, Optional

import functools
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm, trange
import concurrent.futures
import torch.multiprocessing as mp

import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.constraints import Interval
from gpytorch.utils.errors import NotPSDError
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.priors import GammaPrior, Prior, LogNormalPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel, MaternKernel, Kernel

import botorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize, unnormalize

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.factory import get_termination, get_selection
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

import warnings
warnings.filterwarnings("ignore")

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
if os.path.join(ROOT_PROJECT, 'weighted_retraining') in sys.path:
    sys.path.remove(os.path.join(ROOT_PROJECT, 'weighted_retraining'))
sys.path[0] = ROOT_PROJECT

from weighted_retraining.weighted_retraining.utils import SubmissivePlProgressbar, DataWeighter, print_flush
from weighted_retraining.weighted_retraining.bo_torch.mo_acquisition import bo_mo_loop

from utils.utils_cmd import parse_list, parse_dict
from utils.utils_save import get_storage_root, save_w_pickle, str_dict
from weighted_retraining.weighted_retraining.robust_opt_scripts.utils import is_robust
from weighted_retraining.weighted_retraining.robust_opt_scripts.base import add_common_args
from weighted_retraining.weighted_retraining.bo_torch.utils import put_max_in_bounds
from weighted_retraining.weighted_retraining.bo_torch.gp_torch import (
    gp_torch_train, 
    bo_loop, 
    add_gp_torch_args, 
    gp_fit_test)

# My imports
from weighted_retraining.weighted_retraining.chem.chem_data_pseudo import (
    PseudoWeightedJTNNDataset,
    WeightedJTNNDataset,
    WeightedMolTreeFolder,
    get_rec_x_error,
    append_trainset_torch)
from weighted_retraining.weighted_retraining.chem.jtnn.datautils import tensorize
from weighted_retraining.weighted_retraining.chem.chem_model_pseudo import JTVAE
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet
from weighted_retraining.weighted_retraining.robust_opt_scripts import base as wr_base

logger = logging.getLogger("chem-opt")


def setup_logger(logfile):
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    

def _batch_decode_z_and_props(
        model: JTVAE,
        z: torch.Tensor,
        datamodule: WeightedJTNNDataset,
        invalid_score: float,
        pbar: tqdm = None,
):
    """
    helper function to decode some latent vectors and calculate their properties
    """

    # Progress bar description
    if pbar is not None:
        old_desc = pbar.desc
        pbar.set_description("decoding")

    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            z_batch = z[j: j + batch_size]
            smiles_out = model.decode_deterministic(z_batch)
            if pbar is not None:
                pbar.update(z_batch.shape[0])
        z_decode += smiles_out

    # Now finding properties
    if pbar is not None:
        pbar.set_description("calc prop")

    # Calculate objective function values and choose which points to keep
    # Invalid points get a value of None
    z_prop = [
        invalid_score if s is None else datamodule.train_dataset.prop_func(s)
        for s in z_decode
    ]

    # Now back to normal
    if pbar is not None:
        pbar.set_description(old_desc)

    return z_decode, z_prop


def _batch_decode_z_wo_props(
        model: JTVAE,
        z: torch.Tensor,
        tkwargs,
        batch_size = 512,
):
    # Decode all points in a fixed decoding radius
    rdkit_quiet()
    z_decode = []
    try:
        model.to(**tkwargs)
        for j in tqdm(range(0, len(z), batch_size)):
            with torch.no_grad():
                z_batch = z[j: j + batch_size]
                smiles_out = model.decode_deterministic(z_batch.to(model.device))
            z_decode += smiles_out
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print('decode error:', e)
    return z_decode


def _batch_decode_z_wo_props_for_sample(
        z: torch.Tensor,
        model_path: str,
        vocab,
        tkwargs
):
    # Decode all points in a fixed decoding radius
    rdkit_quiet()
    batch_size = 512
    z_decode = []
    model: JTVAE = JTVAE.load_from_checkpoint(model_path, vocab=vocab, strict=False)
    model.eval()
    model.to(**tkwargs)
    try:
        for j in tqdm(range(0, len(z), batch_size)):
            with torch.no_grad():
                z_batch = z[j: j + batch_size]
                smiles_out = model.decode_deterministic(z_batch.to(model.device))
            z_decode += smiles_out
    except Exception as e:
        print('decode error:', e)
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return z_decode


def _choose_best_rand_points(n_rand_points: int, n_best_points: int, dataset: WeightedMolTreeFolder):
    chosen_point_set = set()

    if len(dataset.data) < n_best_points + n_rand_points:
        n_best_points, n_rand_points = int(n_best_points / (n_best_points + n_rand_points) * len(dataset.data)), int(
            n_rand_points / (n_best_points + n_rand_points) * len(dataset.data))
        n_rand_points += 1 if n_best_points + n_rand_points < len(dataset.data) else 0
    print(f"Take {n_best_points} best points and {n_rand_points} random points")

    # Best scores at start
    targets_argsort = np.argsort(-dataset.data_properties.flatten())
    for i in range(n_best_points):
        chosen_point_set.add(targets_argsort[i])
    candidate_rand_points = np.random.choice(
        len(targets_argsort),
        size=n_rand_points + n_best_points,
        replace=False,
    )
    for i in candidate_rand_points:
        if i not in chosen_point_set and len(chosen_point_set) < (n_rand_points + n_best_points):
            chosen_point_set.add(i)
    assert len(chosen_point_set) == (n_rand_points + n_best_points)
    chosen_points = sorted(list(chosen_point_set))

    return chosen_points


def _encode_mol_trees(model, mol_trees):
    batch_size = 64
    mu_list = []
    with torch.no_grad():
        for i in trange(
            0, len(mol_trees), batch_size, desc="encoding GP points", leave=False
        ):
            batch_slice = slice(i, i + batch_size)
            _, jtenc_holder, mpn_holder = tensorize(
                mol_trees[batch_slice], model.jtnn_vae.vocab, assm=False
            )
            tree_vecs, _, mol_vecs = model.jtnn_vae.encode(jtenc_holder, mpn_holder)
            muT = model.jtnn_vae.T_mean(tree_vecs)
            muG = model.jtnn_vae.G_mean(mol_vecs)
            mu = torch.cat([muT, muG], axis=-1).cpu().numpy()
            mu_list.append(mu)

    # Aggregate array
    mu = np.concatenate(mu_list, axis=0).astype(np.float64)
    return mu


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, gpu, cuda, store_best=False,
                  best_ckpt_path: Optional[str] = None):
    # pl._logger.setLevel(logging.CRITICAL)
    train_pbar = SubmissivePlProgressbar(process_position=1)

    # Create custom saver and logger
    tb_logger = TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/val", )

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = 1.0
    elif int(num_epochs) == num_epochs:
        max_epochs = int(num_epochs)
        limit_train_batches = 1.0
    else:
        raise ValueError(f"invalid num epochs {num_epochs}")

    # Create trainer
    trainer = pl.Trainer(
        gpus=[cuda] if cuda is not None else 0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        checkpoint_callback=True,
        terminate_on_nan=True,
        logger=tb_logger,
        callbacks=[train_pbar, checkpointer],
        gradient_clip_val=20.0 # Model is prone to large gradients
    )

    # Fit model
    trainer.fit(model, datamodule)

    if store_best:
        assert best_ckpt_path is not None
        os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
        shutil.copyfile(checkpointer.best_model_path, best_ckpt_path)


def get_root_path(lso_strategy: str, weight_type, k, r,
                  predict_target, hdims, latent_dim: int, beta_kl_final: float,
                  beta_target_pred_loss: float,
                  acq_func_id: str, acq_func_kwargs: Dict[str, Any],
                  input_wp: bool, random_search_type: Optional[str],
                  pseudo_sampling_type: Optional[str],
                  use_pretrained: bool, pretrained_model_id: str, batch_size: int,
                  n_init_retrain_epochs: float, semi_supervised: Optional[bool], n_init_bo_points: Optional[int],
                  use_pseudo: Optional[bool] = False, pseudo_data_size: int=5000,
                  pseudo_beta: float=0.5, beta_pseudo_loss_final: Optional[float] = 1.0,
                  use_ssdkl: Optional[bool] = False, beta_ssdkl_loss: Optional[float] = 1.0
                  ):
    """ Get result root result path (associated directory will contain results for all seeds)
    Args:
        batch_size: batch size used for vae training
        pretrained_model_id: id of the pretrained model
        lso_strategy: type of optimisation
        weight_type: type of weighting used for retraining
        k: weighting parameter
        r: period of retraining
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        acq_func_id: name of acquisition function
        acq_func_kwargs: acquisition function kwargs
        random_search_type: random search specific strategy
        beta_kl_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        latent_dim: dimension of the latent space
        use_pretrained: Whether or not to use a pretrained VAE model
        n_init_retrain_epochs: number of retraining epochs to do before using VAE model in BO
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        n_init_bo_points: number of initial labelled points considered for BO with semi-supervised training

    Returns:
        path to result dir
    """
    result_path = os.path.join(
        get_storage_root(),
        f"logs/opt/chem/{weight_type}/k_{k}/r_{r}")

    exp_spec = f"paper-mol"
    exp_spec += f'-z_dim_{latent_dim}'
    exp_spec += f"-init_{n_init_retrain_epochs:g}"
    if predict_target:
        assert hdims is not None
        exp_spec += '-predy_' + '_'.join(map(str, hdims))
        exp_spec += f'-b_{float(beta_target_pred_loss):g}'
    exp_spec += f'-bkl_{beta_kl_final}'
    if semi_supervised:
        assert n_init_bo_points is not None, n_init_bo_points
        exp_spec += "-semi_supervised"
        exp_spec += f"-n-init-{n_init_bo_points}"
    if use_pretrained:
        exp_spec += f'_pretrain-{pretrained_model_id}'
    else:
        exp_spec += f'_scratch'
    if batch_size != 32:
        exp_spec += f'_bs-{batch_size}'

    if lso_strategy == 'opt':
        acq_func_spec = ''
        if acq_func_id != 'ExpectedImprovement':
            acq_func_spec += acq_func_id
        acq_func_spec += f"{'_inwp_' if input_wp else str(input_wp)}" \
            # if 'ErrorAware' in acq_func_id and cost_aware_gamma_sched is not None:
        #     acq_func_spec += f"_sch-{cost_aware_gamma_sched}"
        if len(acq_func_kwargs) > 0:
            acq_func_spec += f'_{str_dict(acq_func_kwargs)}'
        # 伪标签数据设置
        if use_pseudo:
            exp_spec += "-use_pseudo"
            exp_spec += f'-b_{float(beta_pseudo_loss_final):g}'
            exp_spec += f'-bs_{int(pseudo_data_size):g}'
            exp_spec += f'-bb_{float(pseudo_beta):g}'
            if pseudo_sampling_type is not None:
                exp_spec += f'-bst_{pseudo_sampling_type}'
        if use_ssdkl:
            exp_spec += "-use_ssdkl"
            exp_spec += f'-b_{float(beta_ssdkl_loss):g}'
        result_path = os.path.join(
            result_path, exp_spec, acq_func_spec
        )

    elif lso_strategy == 'sample':
        raise NotImplementedError('Sample lso strategy not supported')
        # result_path = os.path.join(result_path, exp_spec, f'latent-sample')
    elif lso_strategy == 'random_search':
        base = f'latent-random-search'
        if random_search_type == 'sobol':
            base += '-sobol'
        else:
            assert random_search_type is None, f'{random_search_type} is invalid'
        result_path = os.path.join(result_path, exp_spec, base)
    else:
        raise ValueError(f'{lso_strategy} not supported: try `opt`, `sample`...')
    return result_path


def get_path(lso_strategy: str, weight_type, k, r,
             predict_target, hdims, latent_dim: int, beta_kl_final: float, 
             beta_target_pred_loss: float,
             acq_func_id: str, acq_func_kwargs: Dict[str, Any],
             input_wp: bool, random_search_type: Optional[str],
             pseudo_sampling_type: Optional[str],
             use_pretrained: bool, pretrained_model_id: str, batch_size: int,
             n_init_retrain_epochs: int, seed: float, semi_supervised: Optional[bool], n_init_bo_points: Optional[int],
             use_pseudo: Optional[bool] = False, pseudo_data_size: int=5000, 
             pseudo_beta: float=0.5, beta_pseudo_loss_final: Optional[float] = 1.0,
             use_ssdkl: Optional[bool] = False, beta_ssdkl_loss: Optional[float] = 1.0,
             ):
    """ Get result root result path (associated directory will contain results for all seeds)
    Args:
        batch_size: batch size used for vae training
        pretrained_model_id: id of the pretrained model
        seed: for reproducibility
        lso_strategy: type of optimisation
        weight_type: type of weighting used for retraining
        k: weighting parameter
        r: period of retraining
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        acq_func_id: name of acquisition function
        acq_func_kwargs: acquisition function kwargs
        random_search_type: random search specific strategy
        beta_kl_final: weight of the KL in the ELBO
        beta_target_pred_loss: weight of the target prediction loss added to the ELBO
        latent_dim: dimension of the latent space
        use_pretrained: Whether or not to use a pretrained VAE model
        n_init_retrain_epochs: number of retraining epochs to do before using VAE model in BO
        semi_supervised: whether or not to start BO from VAE trained with unlabelled data
        n_init_bo_points: number of initial labelled points considered for BO
    Returns:
        path to result dir
    """
    result_path = get_root_path(
        lso_strategy=lso_strategy,
        weight_type=weight_type,
        k=k,
        r=r,
        predict_target=predict_target,
        latent_dim=latent_dim,
        hdims=hdims,
        acq_func_id=acq_func_id,
        acq_func_kwargs=acq_func_kwargs,
        input_wp=input_wp,
        random_search_type=random_search_type,
        beta_target_pred_loss=beta_target_pred_loss,
        beta_kl_final=beta_kl_final,
        use_pretrained=use_pretrained,
        n_init_retrain_epochs=n_init_retrain_epochs,
        batch_size=batch_size,
        semi_supervised=semi_supervised,
        n_init_bo_points=n_init_bo_points,
        pretrained_model_id=pretrained_model_id,
        use_pseudo=use_pseudo,
        pseudo_data_size=pseudo_data_size,
        pseudo_beta=pseudo_beta,
        beta_pseudo_loss_final=beta_pseudo_loss_final,
        pseudo_sampling_type=pseudo_sampling_type,
        use_ssdkl=use_ssdkl,
        beta_ssdkl_loss=beta_ssdkl_loss
    )
    result_path = os.path.join(result_path, f'seed{seed}')
    return result_path


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.register('type', list, parse_list)
    parser.register('type', dict, parse_dict)

    parser = add_common_args(parser)
    parser = WeightedJTNNDataset.add_model_specific_args(parser)
    parser = DataWeighter.add_weight_args(parser)
    # parser = wr_base.add_gp_args(parser)
    parser = add_gp_torch_args(parser)

    parser.add_argument(
        '--use_test_set',
        dest="use_test_set",
        action="store_true",
        help="flag to use a test set for evaluating the sparse GP"
    )
    parser.add_argument(
        '--use_full_data_for_gp',
        dest="use_full_data_for_gp",
        action="store_true",
        help="flag to use the full dataset for training the GP"
    )
    parser.add_argument(
        "--input_wp",
        action='store_true',
        help="Whether to apply input warping"
    )
    parser.add_argument(
        "--predict_target",
        action='store_true',
        help="Generative model predicts target value",
    )
    parser.add_argument(
        "--target_predictor_hdims",
        type=list,
        default=None,
        help="Hidden dimensions of MLP predicting target values",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=56,
        help="Hidden dimension the latent space",
    )
    parser.add_argument(
        "--use_pretrained",
        action='store_true',
        help="True if using pretrained VAE model",
    )
    parser.add_argument(
        "--pretrained_model_id",
        type=str,
        default='vanilla',
        help="id of the pretrained VAE model used (should be aligned with the pretrained model file)",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=None,
        help="cuda ID",
    )

    vae_group = parser.add_argument_group("Metric learning")
    vae_group.add_argument(
        "--beta_target_pred_loss",
        type=float,
        default=1.,
        help="Weight of the target_prediction loss added in the ELBO",
    )
    vae_group.add_argument(
        "--beta_final",
        type=float,
        help="Weight of the kl loss in the ELBO",
    )
    vae_group.add_argument(
        "--beta_start",
        type=float,
        default=None,
        help="Weight of the kl loss in the ELBO",
    )
    vae_group.add_argument(
        "--semi_supervised",
        action='store_true',
        help="Start BO from VAE trained with unlabelled data.",
    )
    vae_group.add_argument(
        "--n_init_bo_points",
        type=int,
        default=None,
        help="Number of data points to use at the start of the BO if using semi-supervised training of VAE."
             "(We need at least SOME data to fit the GP(s) etc.)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0007,
        help="learning rate of the VAE training optimizer if needed (e.g. in case VAE from scratch)",
    )
    parser.add_argument(
        "--train-only",
        action='store_true',
        help="Train the JTVAE without running the BO",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="If `train-only`, save the trained model in save_model_path.",
    )
    pseudo_group = parser.add_argument_group("Pseudo data")
    pseudo_group.add_argument(
        "--use_pseudo",
        action="store_true",
        help="Wether use the  pseudo label data"
    )
    pseudo_group.add_argument(
        "--pseudo_data_size",
        type=int, default=5000,
        help="sample how many pseudo label data"
    )
    pseudo_group.add_argument(
        "--beta_pseudo_loss",
        type=float,
        default=0.1,
        help="Weight of the pseudo loss added in the ELBO"
    )
    pseudo_group.add_argument(
        "--beta_pseudo_loss_final",
        type=float,
        default=1.,
        help="Weight of the pseudo loss added in the ELBO"
    )
    pseudo_group.add_argument(
        "--pseudo_sampling_type",
        type=str,
        default=None,
        choices=(None, 'cmaes', 'ga', 'noise', 'random', 'random_bound', 'acq'),
        help="Sampling for pseudo data points",
    )
    pseudo_group.add_argument(
        "--process_num",
        type=int,
        default=5,
        help="The num processes to sampling data points",
    )
    pseudo_group.add_argument(
        "--pseudo_sampling_type_kw",
        type=dict,
        default=None,
        help="The parameter for sampling",
    )
    ssdkl_group = parser.add_argument_group("Deep Kernel Learning")
    ssdkl_group.add_argument(
        "--use_ssdkl",
        action="store_true",
        help="Wether use the Deep Kernel Learning"
    )
    ssdkl_group.add_argument(
        "--beta_ssdkl_loss",
        type=float,
        default=1.,
        help="Weight of the ssdkl loss added in the ELBO"
    )
    args = parser.parse_args()

    args.train_path = os.path.join(ROOT_PROJECT, args.train_path)
    args.val_path = os.path.join(ROOT_PROJECT, args.val_path)
    args.pseudo_train_path = os.path.join(ROOT_PROJECT, args.pseudo_train_path)
    args.pseudo_val_path = os.path.join(ROOT_PROJECT, args.pseudo_val_path)
    args.vocab_file = os.path.join(ROOT_PROJECT, args.vocab_file)
    args.property_file = os.path.join(ROOT_PROJECT, args.property_file)

    if 'ErrorAware' in args.acq_func_id:
        assert 'gamma' in args.acq_func_kwargs
        assert 'eta' in args.acq_func_kwargs
        args.error_aware_acquisition = True
    else:
        args.error_aware_acquisition = False

    if args.pretrained_model_file is None:
        if args.use_pretrained:
            raise ValueError("You should specify the path to the pretrained model you want to use via "
                             "--pretrained_model_file argument")

    # Seeding
    pl.seed_everything(args.seed)

    # create result directory
    result_dir = get_path(
        lso_strategy=args.lso_strategy,
        weight_type=args.weight_type,
        k=args.rank_weight_k,
        r=args.retraining_frequency,
        predict_target=args.predict_target,
        latent_dim=args.latent_dim,
        hdims=args.target_predictor_hdims,
        input_wp=args.input_wp,
        seed=args.seed,
        random_search_type=args.random_search_type,
        beta_target_pred_loss=args.beta_target_pred_loss,
        beta_kl_final=args.beta_final,
        use_pretrained=args.use_pretrained,
        n_init_retrain_epochs=args.n_init_retrain_epochs,
        semi_supervised=args.semi_supervised,
        n_init_bo_points=args.n_init_bo_points,
        pretrained_model_id=args.pretrained_model_id,
        batch_size=args.batch_size,
        acq_func_id=args.acq_func_id,
        acq_func_kwargs=args.acq_func_kwargs,
        use_pseudo=args.use_pseudo,
        pseudo_data_size=args.pseudo_data_size,
        pseudo_beta=args.pseudo_beta,
        beta_pseudo_loss_final=args.beta_pseudo_loss_final,
        pseudo_sampling_type=args.pseudo_sampling_type,
        use_ssdkl=args.use_ssdkl,
        beta_ssdkl_loss=args.beta_ssdkl_loss
    )
    print(f'result dir: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)
    save_w_pickle(args, result_dir, 'args.pkl')
    logs = ''
    exc: Optional[Exception] = None
    try:
        main_aux(args, result_dir=result_dir)
    except Exception as e:
        logs = traceback.format_exc()
        exc = e
    f = open(os.path.join(result_dir, 'logs.txt'), "a")
    f.write('\n' + '--------' * 10)
    f.write(logs)
    f.write('\n' + '--------' * 10)
    f.close()
    if exc is not None:
        raise exc


def main_aux(args, result_dir: str):
    """ main """
    device = args.cuda
    if device is not None:
        torch.cuda.set_device(device)
    tkwargs = {
        "dtype": torch.float,
        "device": torch.device(f"cuda:{device}" if torch.cuda.is_available() and device is not None else "cpu"),
    }
    # Seeding
    pl.seed_everything(args.seed)

    if args.train_only and os.path.exists(args.save_model_path) and not args.overwrite:
        print_flush(f'--- JTVAE already trained in {args.save_model_path} ---')
        return

    # Make results directory
    data_dir = os.path.join(result_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    setup_logger(os.path.join(result_dir, "log.txt"))

    # Load data
    datamodule = WeightedJTNNDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit", n_init_points=args.n_init_bo_points)

    # print python command run
    cmd = ' '.join(sys.argv[1:])
    print_flush(f"{cmd}\n")

    # Load model
    if args.use_pretrained:
        if args.predict_target:
            if 'pred_y' in args.pretrained_model_file:
                # fully supervised training from a model already trained with target prediction
                ckpt = torch.load(args.pretrained_model_file)
                ckpt['hyper_parameters']['hparams'].beta_target_pred_loss = args.beta_target_pred_loss
                ckpt['hyper_parameters']['hparams'].predict_target = True
                ckpt['hyper_parameters']['hparams'].target_predictor_hdims = args.target_predictor_hdims
                torch.save(ckpt, args.pretrained_model_file)
        print(os.path.abspath(args.pretrained_model_file))
        vae: JTVAE = JTVAE.load_from_checkpoint(args.pretrained_model_file, vocab=datamodule.vocab, strict=False)
        vae.beta = vae.hparams.beta_final  # Override any beta annealing
        vae.predict_target = args.predict_target
        vae.hparams.predict_target = args.predict_target
        vae.beta_target_pred_loss = args.beta_target_pred_loss
        vae.hparams.beta_target_pred_loss = args.beta_target_pred_loss
        vae.target_predictor_hdims = args.target_predictor_hdims
        vae.hparams.target_predictor_hdims = args.target_predictor_hdims
        if vae.predict_target and vae.target_predictor is None:
            vae.hparams.target_predictor_hdims = args.target_predictor_hdims
            vae.hparams.predict_target = args.predict_target
            vae.build_target_predictor()
        vae.use_pseudo = args.use_pseudo
        vae.hparams.use_pseudo = args.use_pseudo
        vae.beta_pseudo_loss = args.beta_pseudo_loss
        vae.hparams.beta_pseudo_loss = args.beta_pseudo_loss

        vae.use_ssdkl = args.use_ssdkl
        vae.hparams.use_ssdkl = args.use_ssdkl
        vae.beta_ssdkl_loss = args.beta_ssdkl_loss
        vae.hparams.beta_ssdkl_loss = args.beta_ssdkl_loss
    
    else:
        print("initialising VAE from scratch !")
        vae: JTVAE = JTVAE(hparams=args, vocab=datamodule.vocab)
    vae.eval()

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))
    postfix = dict(
        retrain_left=num_retrain,
        best=float(datamodule.train_dataset.data_properties.max()),
        n_train=len(datamodule.train_dataset.data),
        save_path=result_dir
    )
    VarThreshold = None

    start_num_retrain = 0

    # Set up results tracking
    results = dict(
        opt_points=[],
        opt_point_properties=[],
        opt_model_version=[],
        params=str(sys.argv),
        sample_points=[],
        sample_versions=[],
        sample_properties=[],
        var_threshold_list=[],
    )

    result_filepath = os.path.join(result_dir, 'results.npz')
    if not args.overwrite and os.path.exists(result_filepath):
        with np.load(result_filepath, allow_pickle=True) as npz:
            results = {}
            for k in list(npz.keys()):
                results[k] = npz[k]
                if k != 'params':
                    results[k] = list(results[k])
                else:
                    results[k] = npz[k].item()
        start_num_retrain = results['opt_model_version'][-1] + 1

        prev_retrain_model = args.retraining_frequency * (start_num_retrain - 1)
        num_sampled_points = len(results['opt_points'])
        VarThreshold = results['var_threshold_list'][-1]
        if args.n_init_retrain_epochs == 0 and prev_retrain_model == 0:
            pretrained_model_path = args.pretrained_model_file
        else:
            pretrained_model_path = os.path.join(result_dir, 'retraining', f'retrain_{prev_retrain_model}',
                                                 'checkpoints',
                                                 'last.ckpt')
        print(f"Found checkpoint at {pretrained_model_path}")
        ckpt = torch.load(pretrained_model_path)
        ckpt['hyper_parameters']['hparams'].beta_target_pred_loss = args.beta_target_pred_loss
        if args.predict_target:
            ckpt['hyper_parameters']['hparams'].predict_target = True
            ckpt['hyper_parameters']['hparams'].target_predictor_hdims = args.target_predictor_hdims
        if args.use_pseudo:
            ckpt['hyper_parameters']['hparams'].use_pseudo = True
            ckpt['hyper_parameters']['hparams'].beta_pseudo_loss = args.beta_pseudo_loss
        if args.use_ssdkl:
            ckpt['hyper_parameters']['hparams'].use_ssdkl = True
            ckpt['hyper_parameters']['hparams'].beta_ssdkl_loss = args.beta_ssdkl_loss
        
        torch.save(ckpt, pretrained_model_path)
        print(f"Loading model from {pretrained_model_path}")
        vae.load_from_checkpoint(pretrained_model_path, vocab=datamodule.vocab, strict=False)
        if args.predict_target and not hasattr(vae.hparams, 'predict_target'):
            vae.hparams.target_predictor_hdims = args.target_predictor_hdims
            vae.hparams.predict_target = args.predict_target
        # vae.hparams.cuda = args.cuda
        vae.beta = vae.hparams.beta_final  # Override any beta annealing
        vae.eval()

        # Set up some stuff for the progress bar
        num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency)) - start_num_retrain

        print(f"Append existing points and properties to datamodule...")
        datamodule.append_train_data(
            np.array(results['opt_points']),
            np.array(results['opt_point_properties'])
        )
        postfix = dict(
            retrain_left=num_retrain,
            best=float(datamodule.train_dataset.data_properties.max()),
            n_train=len(datamodule.train_dataset.data),
            initial=num_sampled_points,
            save_path=result_dir
        )
        print(f"Retrain from {result_dir} | Best: {max(results['opt_point_properties'])}")
    start_time = time.time()

    PseudoDatamodule = None
    GPLayer = None
    GPkwargs = None
    GPFile = None
    # Main loop
    with tqdm(
        total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        for ret_idx in range(start_num_retrain, start_num_retrain + num_retrain):
            if (vae.predict_target or vae.use_ssdkl):
                vae.training_m = datamodule.training_m
                vae.training_M = datamodule.training_M
                vae.validation_m = datamodule.validation_m
                vae.validation_M = datamodule.validation_M

            pbar.set_postfix(postfix)
            pbar.set_description("retraining")
            print(result_dir)
            # Decide whether to retrain
            samples_so_far = args.retraining_frequency * ret_idx
           
            # Optionally do retraining
            start_retrain = time.time()
            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                num_epochs = args.n_init_retrain_epochs

            if num_epochs > 0:
                retrain_dir = os.path.join(result_dir, "retraining")
                version = f"retrain_{samples_so_far}"
                if PseudoDatamodule is None:
                    vae.use_pseudo = False
                    vae.hparams.use_pseudo = False

                    if GPLayer is not None:
                        vae.build_gp_predictor(GPLayer, GPkwargs)

                    print('\n retrain: {}, vae.beta_ssdkl_loss: {}'.format(ret_idx, vae.beta_ssdkl_loss))
                    retrain_model(
                        model=vae, 
                        datamodule=datamodule, 
                        save_dir=retrain_dir,
                        version_str=version, 
                        num_epochs=num_epochs, 
                        gpu=args.gpu, 
                        cuda=args.cuda,
                        store_best=args.train_only,
                        best_ckpt_path=args.save_model_path
                    )
                else:
                    vae.use_pseudo = True
                    vae.hparams.use_pseudo = True
                    beta_pseudo_loss=increment_beta_pseudo_loss(
                        vae.beta_pseudo_loss, 
                        args.beta_pseudo_loss_final, 
                        ret_idx, 
                        start_num_retrain, 1, 1.3)
                    vae.beta_pseudo_loss = beta_pseudo_loss
                    vae.hparams.beta_pseudo_loss = beta_pseudo_loss

                    if GPLayer is not None:
                        vae.build_gp_predictor(GPLayer, GPkwargs)

                    print('\n retrain: {}, vae.beta_pseudo_loss: {}, vae.beta_ssdkl_loss: {}'.format(ret_idx, vae.beta_pseudo_loss, vae.beta_ssdkl_loss))
                    retrain_model(
                        model=vae, 
                        datamodule=PseudoDatamodule, 
                        save_dir=retrain_dir,
                        version_str=version, 
                        num_epochs=num_epochs, 
                        gpu=args.gpu, 
                        cuda=args.cuda,
                        store_best=args.train_only,
                        best_ckpt_path=args.save_model_path
                    )
                vae.eval()
                vae.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                if args.train_only:
                    return
            del num_epochs
            print('retrain spend: ', time.time()-start_retrain)

            model = vae

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Draw samples for logs!
            if args.samples_per_model > 0:
                pbar.set_description("sampling")
                with trange(
                        args.samples_per_model, desc="sampling", leave=False
                ) as sample_pbar:
                    sample_x, sample_y = latent_sampling(
                        args, model, datamodule, args.samples_per_model,
                        pbar=sample_pbar
                    )

                # Append to results dict
                results["sample_points"].append(sample_x)
                results["sample_properties"].append(sample_y)
                results["sample_versions"].append(ret_idx)

            # Do querying!
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )
            if args.lso_strategy == "opt":
                gp_dir = os.path.join(result_dir, "gp", f"iter{samples_so_far}")
                os.makedirs(gp_dir, exist_ok=True)
                gp_data_file = os.path.join(gp_dir, "data.npz")
                gp_err_data_file = os.path.join(gp_dir, "data_err.npz")
                retrained_model_path = os.path.join(
                    result_dir, 'retraining', f"retrain_{samples_so_far}",
                    'checkpoints', 'last.ckpt')
                x_new, y_new, pseudodatamodule, varthreshold, gp_model, gpkwargs = latent_optimization(
                    args=args,
                    result_dir=result_dir,
                    samples_so_far=samples_so_far,
                    varthreshold=VarThreshold,
                    GPFile=GPFile,
                    retrained_model_path=retrained_model_path,
                    model=model,
                    datamodule=datamodule,
                    n_inducing_points=args.n_inducing_points,
                    n_best_points=args.n_best_points,
                    n_rand_points=args.n_rand_points,
                    tkwargs=tkwargs,
                    num_queries_to_do=num_queries_to_do,
                    gp_data_file=gp_data_file,
                    gp_run_folder=gp_dir,
                    scale=args.scale,
                    covar_name=args.covar_name,
                    acq_func_id=args.acq_func_id,
                    acq_func_kwargs=args.acq_func_kwargs,
                    acq_func_opt_kwargs=args.acq_func_opt_kwargs,
                    q=1, 
                    num_restarts=args.num_restarts,
                    raw_initial_samples=args.raw_initial_samples,
                    num_MC_sample_acq=args.num_MC_sample_acq,
                    input_wp=args.input_wp,
                    gpu=args.gpu,
                    invalid_score=args.invalid_score,
                    pbar=pbar,
                    postfix=postfix,
                )
            elif args.lso_strategy == "sample":
                x_new, y_new = latent_sampling(
                    args, model, datamodule, num_queries_to_do, pbar=pbar,
                )
            else:
                raise NotImplementedError(args.lso_strategy)
            
            VarThreshold = varthreshold
            # Update dataset
            datamodule.append_train_data(x_new, y_new)

            # Add new results
            results["opt_points"] += list(x_new)
            results["opt_point_properties"] += list(y_new)
            results["opt_model_version"] += [ret_idx] * len(x_new)

            postfix["best"] = max(postfix["best"], float(max(y_new)))
            postfix["n_train"] = len(datamodule.train_dataset.data)

            results["var_threshold_list"] += [VarThreshold]
            pbar.set_postfix(postfix)

            # Save results
            np.savez_compressed(os.path.join(result_dir, "results.npz"), **results)

            # Keep a record of the dataset here
            new_data_file = os.path.join(
                data_dir, f"train_data_iter{samples_so_far + num_queries_to_do}.txt"
            )
            with open(new_data_file, "w") as f:
                f.write("\n".join(datamodule.train_dataset.canonic_smiles))
            
            if args.use_pseudo and pseudodatamodule is not None:
                PseudoDatamodule = PseudoWeightedJTNNDataset()
                PseudoDatamodule.setup(LabelDataModule=datamodule, PseudoDataModule=pseudodatamodule)
            if args.use_ssdkl and gp_model is not None:
                GPLayer = gp_model
                GPkwargs = gpkwargs

    print_flush("=== DONE ({:.3f}s) ===".format(time.time() - start_time))


def latent_optimization(
        args,
        result_dir,
        samples_so_far,
        varthreshold,
        GPFile,
        retrained_model_path,
        model: JTVAE,
        datamodule: WeightedJTNNDataset,
        n_inducing_points: int,
        n_best_points: int,
        n_rand_points: int,
        tkwargs: Dict[str, Any],
        num_queries_to_do: int,
        invalid_score: float,
        gp_data_file: str,
        gp_run_folder: str,
        gpu: bool,
        scale: bool,
        covar_name: str,
        acq_func_id: str,
        acq_func_kwargs: Dict[str, Any],
        acq_func_opt_kwargs: Dict[str, Any],
        q: int,
        num_restarts: int,
        raw_initial_samples: int,
        num_MC_sample_acq: int,
        input_wp: bool,
        pbar=None,
        postfix=None,
):
    ##################################################
    # Prepare GP
    ##################################################

    # First, choose GP points to train!
    dset = datamodule.train_dataset

    chosen_indices = _choose_best_rand_points(n_rand_points=n_rand_points, n_best_points=n_best_points, dataset=dset)
    mol_trees = [dset.data[i] for i in chosen_indices]
    targets = dset.data_properties[chosen_indices]
    chosen_smiles = [dset.canonic_smiles[i] for i in chosen_indices]

    # Next, encode these mol trees
    if gpu:
        model = model.to(**tkwargs)
    latent_points = _encode_mol_trees(model, mol_trees)
    model = model.cpu()  # Make sure to free up GPU memory
    gc.collect()
    torch.cuda.empty_cache()  # Free the memory up for tensorflow
    
    # do not standardize -> we'll normalize in unit cube
    X_train = torch.tensor(latent_points).to(**tkwargs)
    # standardise targets
    y_train = torch.tensor(targets).unsqueeze(1).to(**tkwargs)

    ##################################################
    # Run iterative GP fitting/optimization
    ##################################################
    curr_gp_file = None
    curr_gp_err_file = None

    all_new_opts = []
    all_new_smiles = []
    all_new_props = []
    all_new_err = []

    n_rand_acq = 0  # number of times we have to acquire a random point as bo acquisition crashed

    rand_point_due_bo_fail = []
    # for gp_iter in range(num_queries_to_do):
    gp_iter = 0
    while len(all_new_smiles) < num_queries_to_do:
        # Part 1: fit GP
        # ===============================
        new_gp_file = os.path.join(gp_run_folder, f"gp_train_res{gp_iter:04d}.npz")
        
        iter_seed = int(np.random.randint(10000))

        gp_file = None
        if gp_iter == 0:
            # Add commands for initial fitting
            gp_fit_desc = "GP initial fit"
            # n_perf_measure = 0
            current_n_inducing_points = min(X_train.shape[0], n_inducing_points)
        else:
            gp_fit_desc = "GP incremental fit"
            gp_file = curr_gp_file

        init = gp_iter == 0
        # if semi-supervised training, wait until we have enough points to use as many inducing points as
        # we wanted and re-init GP
        if X_train.shape[0] == n_inducing_points:
            current_n_inducing_points = n_inducing_points
            init = True

        old_desc = None
        # Set pbar status for user
        if pbar is not None:
            old_desc = pbar.desc
            pbar.set_description(gp_fit_desc)

        np.random.seed(iter_seed)

        # To account for outliers
        bounds = torch.zeros(2, X_train.shape[1], **tkwargs)
        bounds[0] = torch.quantile(X_train, .0005, dim=0)
        bounds[1] = torch.quantile(X_train, .9995, dim=0)
        # make sure best sample is within bounds
        target_mean, target_std = y_train.mean(), y_train.std()
        y_train_std = y_train.add(-target_mean).div(target_std)
        bounds = put_max_in_bounds(X_train, y_train_std, bounds)

        delta = .05 * (bounds[1] - bounds[0])
        bounds[0] -= delta
        bounds[1] += delta
        print(f"Using data bound of {bounds}...")

        train_x = normalize(X_train, bounds)
        try:
            gp_model = gp_torch_train(
                train_x=train_x,
                train_y=y_train_std,
                n_inducing_points=current_n_inducing_points,
                tkwargs=tkwargs,
                init=init,
                scale=scale,
                covar_name=covar_name,
                gp_file=gp_file,
                save_file=new_gp_file,
                input_wp=input_wp,
                outcome_transform=None,
                options={'lr': 5e-3, 'maxiter': 1000} if init else {'lr': 5e-3, 'maxiter': 500}
            )
        except (RuntimeError, NotPSDError) as e:  # Random acquisition
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print_flush(f"\t\tNon PSD Error in GP fit. Re-fitting objective GP from scratch...")
            gp_model = gp_torch_train(
                train_x=train_x,
                train_y=y_train_std,
                n_inducing_points=current_n_inducing_points,
                tkwargs=tkwargs,
                init=True,
                scale=scale,
                covar_name=covar_name,
                gp_file=gp_file,
                save_file=new_gp_file,
                input_wp=input_wp,
                outcome_transform=None,
                options={'lr': 5e-3, 'maxiter': 1000}
            )
        curr_gp_file = new_gp_file

        # Part 2: optimize GP acquisition func to query point
        # ===============================
        # Run GP opt script
        if pbar is not None:
            pbar.set_description("optimizing acq func")

        print_flush(f"\t\tPicking new inputs nb. {gp_iter + 1} via optimization...")
        try:  
            print('robust_opt_chem', acq_func_id)
            gp_model.eval()
            res = bo_loop(
                gp_model=gp_model,
                acq_func_id=acq_func_id,
                acq_func_kwargs=acq_func_kwargs,
                acq_func_opt_kwargs=acq_func_opt_kwargs,
                bounds=normalize(bounds, bounds),
                tkwargs=tkwargs,
                q=q,
                num_restarts=num_restarts,
                raw_initial_samples=raw_initial_samples,
                seed=iter_seed,
                num_MC_sample_acq=num_MC_sample_acq,
            )
            z_opt = res
            rand_point_due_bo_fail += [0] * q
        except (RuntimeError, NotPSDError) as e:  # Random acquisition
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print_flush(f"\t\tPicking new inputs nb. {gp_iter + 1} via random sampling...")
            n_rand_acq += q
            z_opt = torch.rand(q, bounds.shape[1]).to(bounds)
            exc = e
            rand_point_due_bo_fail += [1] * q
        
        gc.collect()
        torch.cuda.empty_cache()
        
        z_opt = unnormalize(z_opt.float(), bounds).cpu()
        # Decode point
        model.to(**tkwargs)
        smiles_opt, prop_opt = _batch_decode_z_and_props(
            model,
            torch.as_tensor(z_opt, device=model.device),
            datamodule,
            invalid_score=invalid_score,
            pbar=pbar,
        )
        model.cpu()  
        gc.collect()
        torch.cuda.empty_cache() 

        # Append to new GP data
        latent_points = np.concatenate([latent_points, z_opt], axis=0)
        targets = np.concatenate([targets, prop_opt], axis=0)
        chosen_smiles.append(smiles_opt)

        # Append to overall list
        all_new_opts += z_opt
        all_new_smiles += smiles_opt
        all_new_props += prop_opt

        print_flush(f"\t\tPicked new input with value {all_new_props[-1]}...")

        # Reset pbar description
        if pbar is not None:
            pbar.set_description(old_desc)

            # Update best point in progress bar
            if postfix is not None:
                postfix["best"] = max(postfix["best"], float(max(all_new_props)))
                pbar.set_postfix(postfix)
        X_train, y_train = append_trainset_torch(X_train, y_train, new_inputs=z_opt, 
                                                 new_scores=torch.as_tensor(prop_opt).reshape(-1, 1))
        gp_iter += 1

    gc.collect()
    torch.cuda.empty_cache()

    pseudodatamodule = None
    gpkwargs = {}
    gpkwargs={
        'bounds': bounds,
        'target_mean': target_mean,
        'target_std': target_std
    }
    if args.use_pseudo:
        model.eval()
        gp_model.eval()
        gp_model.to(**tkwargs)

        sampling_start = time.time()
        if args.pseudo_sampling_type in ['cmaes', 'ga', 'noise']:
            seed_dset = datamodule.train_dataset
            seed_chosen_indices = _choose_best_rand_points(
                n_rand_points=args.pseudo_sampling_type_kw['n_rand'], 
                n_best_points=args.pseudo_sampling_type_kw['n_best'], 
                dataset=seed_dset)
            seed_mol_trees = [seed_dset.data[i] for i in seed_chosen_indices]
            seed_targets = seed_dset.data_properties[seed_chosen_indices]
            seed_chosen_smiles = [seed_dset.canonic_smiles[i] for i in seed_chosen_indices]

            model.to(**tkwargs)
            seed_latent_points = _encode_mol_trees(model, seed_mol_trees)
            model.cpu()  # Make sure to free up GPU memory
            gc.collect()
            torch.cuda.empty_cache()  # Free the memory up for tensorflow

            seed_z = seed_latent_points
            seed_y = np.array(seed_targets)
            seed_x = np.array(seed_chosen_smiles)

            if args.pseudo_sampling_type_kw['use_bo']:
                seed_z = np.concatenate([seed_z, np.vstack(all_new_opts)])
                seed_y = np.concatenate([seed_y, np.array(all_new_props)])
                seed_x = np.concatenate([seed_x, np.array(all_new_smiles)])
            
            seed_z = torch.from_numpy(seed_z).to(bounds)
            seed_z = normalize(seed_z, bounds)

            seed_posterior = gp_model.posterior(seed_z)
            seed_mean = seed_posterior.mean.view(-1).detach().cpu().numpy()
            seed_var = seed_posterior.variance.view(-1).detach().cpu().numpy()

            seed_file = os.path.join(result_dir, f"sampling_seed_iter{samples_so_far + num_queries_to_do}")
            np.savez_compressed(
                str(seed_file),
                data=seed_x,
                gp_mean=seed_mean,
                score=seed_y,
                gp_var=seed_var,
            )
            seed_z = seed_z.cpu().numpy()

            if args.pseudo_sampling_type == 'ga':
                varthreshold, heuristic_sampling_z, heuristic_sampling_x, heuristic_sampling_y = latent_sampling_GA(
                    object=gp_model,  
                    model_path=retrained_model_path,
                    vocab=datamodule.vocab,
                    num_queries_to_do=args.pseudo_data_size, 
                    init_points=seed_z, 
                    varthreshold=varthreshold, 
                    sigma=args.pseudo_sampling_type_kw['sigma'],   
                    bounds=bounds, 
                    process_num=args.process_num,
                    tkwargs=tkwargs,
                    use_filter=args.pseudo_sampling_type_kw['use_filter'])
            elif args.pseudo_sampling_type == 'cmaes':
                varthreshold, heuristic_sampling_z, heuristic_sampling_x, heuristic_sampling_y = latent_sampling_CMA_ES(
                    object=gp_model,  
                    model_path=retrained_model_path,
                    vocab=datamodule.vocab,
                    num_queries_to_do=args.pseudo_data_size, 
                    init_points=seed_z, 
                    varthreshold=varthreshold, 
                    sigma=args.pseudo_sampling_type_kw['sigma'],   
                    bounds=bounds,
                    process_num=args.process_num, 
                    tkwargs=tkwargs,
                    use_filter=args.pseudo_sampling_type_kw['use_filter'])
            elif args.pseudo_sampling_type == 'noise':
                varthreshold, heuristic_sampling_z, heuristic_sampling_x, heuristic_sampling_y, heuristic_sampling_var = latent_sampling_Noise(
                    object=gp_model,  
                    model_path=retrained_model_path,
                    vocab=datamodule.vocab,
                    num_queries_to_do=args.pseudo_data_size, 
                    init_points=seed_z, 
                    varthreshold=varthreshold, 
                    sigma=args.pseudo_sampling_type_kw['sigma'],   
                    bounds=bounds, 
                    tkwargs=tkwargs,
                    process_num=args.process_num,
                    use_filter=args.pseudo_sampling_type_kw['use_filter'])
            print("\n heuristic sampling spend time: ", time.time()-sampling_start)

            need_to_rand_sampling = args.pseudo_data_size-len(heuristic_sampling_x)
            print('\n need_to_rand_sampling:', need_to_rand_sampling)
            if need_to_rand_sampling > 0 and need_to_rand_sampling != args.pseudo_data_size:
                rand_sampling_start = time.time()
                rand_sampling_x, rand_sampling_z, rand_sampling_y, rand_sampling_var=latent_sampling_wo_props(
                            object=gp_model, 
                            model=model, 
                            model_path=retrained_model_path, 
                            vocab=datamodule.vocab, 
                            varthreshold=varthreshold, 
                            bounds=bounds,
                            num_queries_to_do=need_to_rand_sampling, 
                            tkwargs=tkwargs,
                            process_num=args.process_num)
                print("\n rand sampling spend time: ", time.time()-rand_sampling_start)
                if rand_sampling_x is not None:
                    sampling_x = np.hstack([heuristic_sampling_x, np.squeeze(rand_sampling_x)])
                    sampling_y = np.hstack([heuristic_sampling_y, np.squeeze(rand_sampling_y)])
                    sampling_var = np.hstack([heuristic_sampling_var, np.squeeze(rand_sampling_var)])
                else:
                    sampling_x = heuristic_sampling_x
                    sampling_y = heuristic_sampling_y
                    sampling_var = heuristic_sampling_var
            else:
                sampling_x = heuristic_sampling_x
                sampling_y = heuristic_sampling_y
                sampling_var = heuristic_sampling_var

        elif args.pseudo_sampling_type == 'random':
            sampling_x, sampling_z, sampling_y, sampling_var=latent_sampling_wo_props(
                object=gp_model, 
                model=model, 
                model_path=retrained_model_path, 
                vocab=datamodule.vocab, 
                num_queries_to_do=args.pseudo_data_size, 
                varthreshold=varthreshold, 
                tkwargs=tkwargs, 
                bounds=bounds, 
                process_num=args.process_num, 
                use_filter=args.pseudo_sampling_type_kw['use_filter'], 
                warm_up=False)
        else:
            raise ValueError(f'{args.pseudo_sampling_type} not supported')

        if sampling_x is not None and len(sampling_x) != 0:
            assert len(sampling_x)==len(sampling_y), (sampling_x.shape, sampling_y.shape)
            sampling_y = sampling_y * target_std.cpu().numpy() + target_mean.cpu().numpy()

            pseudo_data_file = os.path.join(result_dir, f"pseudo_data_iter{samples_so_far + num_queries_to_do}")
            np.savez_compressed(
                str(pseudo_data_file),
                data=sampling_x,
                gp_mean=sampling_y,
                gp_var=sampling_var
            )

            start_build = time.time()
            print("start to build...")
            pseudodatamodule = WeightedJTNNDataset(args, utils.DataWeighter(args), isPseudo=True)
            pseudodatamodule.setup("fit")
            pseudodatamodule.append_pseudo_data(sampling_x, sampling_y)
            print("end to build: ", time.time()-start_build)
        gp_model.cpu()
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    # Update datamodule with ALL data points
    return all_new_smiles, all_new_props, pseudodatamodule, varthreshold, gp_model, gpkwargs


def latent_sampling(args, model, datamodule, num_queries_to_do, pbar=None):
    """ Draws samples from latent space and appends to the dataset """

    z_sample = torch.randn(num_queries_to_do, model.latent_dim, device=model.device)
    z_decode, z_prop = _batch_decode_z_and_props(
        model, z_sample, datamodule, args, pbar=pbar
    )
    return z_decode, z_prop


def latent_sampling_wo_props(object, model, model_path, vocab, num_queries_to_do, varthreshold, tkwargs, bounds, process_num, use_filter=False, warm_up=False):
    z_sample = torch.rand(num_queries_to_do, bounds.shape[1])
    z_sample_unormal = unnormalize(z_sample.to(bounds), bounds)
    if num_queries_to_do >= 1000:
        sampling_x = []
        task_num = len(z_sample_unormal)//process_num + 1
        task_function = functools.partial(
            _batch_decode_z_wo_props_for_sample,
            model_path = model_path,
            vocab = vocab,
            tkwargs = tkwargs,
        )
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=process_num) as executor:
                futures = [executor.submit(task_function, z_sample_unormal[i*task_num:(i+1)*task_num]) for i in range(process_num)]
                for future in futures:
                    result = future.result()
                    sampling_x.extend(result)
        except Exception as exc:
            print(f'Task raised an exception: {exc}')
        finally:
            executor.shutdown()
    else:
        model.eval()
        sampling_x = _batch_decode_z_wo_props(model, z_sample_unormal, tkwargs)
    sampling_x = np.array(sampling_x)
    
    z_sample = z_sample.numpy()
    object.eval()
    object.to(**tkwargs)
    new_posterior = object.posterior(torch.from_numpy(z_sample).to(bounds))
    y_sampling = new_posterior.mean.view(-1).detach().cpu().numpy()
    var_sampling = new_posterior.variance.view(-1).detach().cpu().numpy() 
    object.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return sampling_x[:num_queries_to_do], z_sample[:num_queries_to_do], y_sampling[:num_queries_to_do], var_sampling[:num_queries_to_do]


def latent_sampling_Noise(object, model_path, vocab, num_queries_to_do, init_points, varthreshold, sigma, bounds, tkwargs, process_num, use_filter=False):
    sampling_z, sampling_x, sampling_y, sampling_var = [], [], [], []
    iter_sampling_number = num_queries_to_do // len(init_points) + 1
    warm_up_number = num_queries_to_do // 10 + 1

    object.eval()
    object.to(**tkwargs)

    warm_up_z = []
    if use_filter and varthreshold is None:
        for point in init_points:
            warm_zs = []
            while len(warm_zs) < warm_up_number:
                new_z = np.random.normal(loc=point, scale=sigma, size=(warm_up_number, point.shape[-1]))
                if len(warm_zs) == 0:
                    warm_zs = new_z                
                else:
                    warm_zs = np.concatenate([warm_zs, new_z])
                    warm_zs = np.unique(warm_zs, axis=0)
            if len(warm_up_z) == 0:
                warm_up_z = warm_zs[:warm_up_number]
            else:
                warm_up_z = np.concatenate([warm_up_z, warm_zs[:warm_up_number]])
                warm_up_z = np.unique(warm_up_z, axis=0)
        warm_posterior = object.posterior(torch.from_numpy(warm_up_z).to(**tkwargs))
        warm_var = warm_posterior.variance.view(-1).detach().cpu().numpy()   
        varthreshold = warm_var.mean()

    mean_var_list = []
  
    for point in init_points:
        zs, ys, var = [], [], []
        redo = 0
        while len(zs) < iter_sampling_number:
            new_z = np.random.normal(loc=point, scale=sigma, size=(iter_sampling_number, point.shape[-1]))
            new_posterior = object.posterior(torch.from_numpy(new_z).to(**tkwargs))
            new_y = new_posterior.mean.view(-1).detach().cpu().numpy()
            new_var = new_posterior.variance.view(-1).detach().cpu().numpy()

            mean_var_list.append(new_var.mean())

            if use_filter and varthreshold is not None:
                valid_indx = new_var <= varthreshold
                new_z = new_z[np.squeeze(valid_indx)]
                new_y = new_y[valid_indx]
                new_var = new_var[valid_indx]

                if len(new_z) == 0:
                    redo += 1
                    if redo < 10:
                        continue
                    else:
                        break

            if len(zs) == 0:
                zs = new_z
                ys = new_y
                var = new_var
            else:
                zs = np.concatenate([zs, new_z])
                ys = np.concatenate([ys, new_y])
                var = np.concatenate([var, new_var])
                zs, unique_indx = np.unique(zs, axis=0, return_index=True)
                ys = ys[unique_indx]
                var = var[unique_indx]
        if len(zs) != 0:
            sampling_z.append(zs)
            sampling_y.append(ys)
            sampling_var.append(var)
        else:
            continue
    
    var_mean = np.array(mean_var_list)
    if varthreshold is None: 
        varthreshold = var_mean.mean()
    else:
        varthreshold = 0.9 * varthreshold + 0.1 * var_mean.mean()

    if len(sampling_z) != 0:
        sampling_z = np.vstack(sampling_z)
        sampling_y = np.hstack(sampling_y)
        sampling_var = np.hstack(sampling_var)
        zs=unnormalize(torch.from_numpy(sampling_z).to(**tkwargs), bounds.to(**tkwargs)).squeeze().cpu()
        sampling_x = []
        task_num = len(zs)//process_num + 1
        task_function = functools.partial(
            _batch_decode_z_wo_props_for_sample,
            model_path = model_path,
            vocab = vocab,
            tkwargs = tkwargs,
        )
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=process_num) as executor:
                futures = [executor.submit(task_function, zs[i*task_num:(i+1)*task_num]) for i in range(process_num)]
                for future in futures:
                    result = future.result()
                    sampling_x.extend(result)
        except Exception as exc:
            print(f'Task raised an exception: {exc}')
        finally:
            executor.shutdown()
        sampling_x = np.array(sampling_x)
    object.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return varthreshold, sampling_z[:num_queries_to_do], sampling_x[:num_queries_to_do], sampling_y[:num_queries_to_do], sampling_var[:num_queries_to_do]


def latent_sampling_GA(object, model_path, vocab, num_queries_to_do, init_points, bounds, varthreshold, tkwargs, process_num, sigma=None, use_filter=False):
    object.eval()
    object.to(**tkwargs)

    sampling_z, sampling_x, sampling_y, sampling_var = [], [], [], []
    class SamplingProblem(Problem):
        def __init__(self, n_var, object, xl, xu, tkwargs):
            super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
            self.object = object
            self.tkwargs = tkwargs

        def  _evaluate(self, x, out, *args, **kwargs):
            x_tensor = torch.from_numpy(x).to(**self.tkwargs)
            posterior = self.object.posterior(x_tensor)
            out['F'] = -posterior.mean.view(-1).detach().cpu().numpy()
            out['var'] = posterior.variance.view(-1).detach().cpu().numpy()

    ga_start = time.time()
    prob = SamplingProblem(n_var=init_points.shape[1], object=object, xl=0.0, xu=1.0, tkwargs=tkwargs)
    pop = Population.new("X", init_points)
    Evaluator().eval(prob, pop)
    selection = RandomSelection()
    mutation = PolynomialMutation(prob=1.0, eta=100)
    n_gen = num_queries_to_do // len(init_points) + 1
    termination = get_termination("n_gen", 2*n_gen)

    ga = GA(pop_size=len(init_points),  selection=selection, mutation=mutation)
    ga.setup(problem=prob, sampling=pop, termination=termination)

    mean_var_list = []
    while ga.has_next() and len(sampling_z) < num_queries_to_do:
        p = ga.pop
        if p is None:
            ga.next()
            continue
        zs = p.get('X')
        ys = -p.get('F').squeeze()
        var = p.get('var').squeeze()
        mean_var_list.append(var.mean())
        if use_filter and varthreshold is not None:
            valid_indx = var <= varthreshold
            zs = zs[valid_indx]
            ys = ys[valid_indx]
            var = var[valid_indx]
        if zs is not None and len(zs) != 0 :
            if sampling_z is None or len(sampling_z) == 0:
                sampling_z = zs
                sampling_y = ys 
                sampling_var = var
            else:
                sampling_z = np.concatenate([sampling_z, zs]) 
                sampling_y = np.concatenate([sampling_y, ys])  
                sampling_var = np.concatenate([sampling_var, var])
                sampling_z, unique_indx = np.unique(sampling_z, axis=0, return_index=True)
                sampling_y = sampling_y[unique_indx]
                sampling_var = sampling_var[unique_indx]
        ga.next()
    print("\n ga spend time: ", time.time()-ga_start)
    
    var_mean = np.array(mean_var_list).mean()
    if varthreshold is None:
        varthreshold = var_mean
    else:
        varthreshold = 0.9 * varthreshold + 0.1 * var_mean

    if sampling_z is not None and len(sampling_z) != 0:
        zs=unnormalize(torch.from_numpy(sampling_z).to(**tkwargs), bounds.to(**tkwargs)).cpu()
        sampling_x = []
        task_num = len(zs)//process_num + 1
        task_function = functools.partial(
            _batch_decode_z_wo_props_for_sample,
            model_path = model_path,
            vocab = vocab,
            tkwargs = tkwargs,
        )
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=process_num) as executor:
                futures = [executor.submit(task_function, zs[i*task_num:(i+1)*task_num]) for i in range(process_num)]
                for future in futures:
                    result = future.result()
                    sampling_x.extend(result)
        except Exception as exc:
            print(f'Task raised an exception: {exc}')
        finally:
            executor.shutdown()
        sampling_x = np.array(sampling_x)

    object.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return varthreshold, sampling_z[:num_queries_to_do], sampling_x[:num_queries_to_do], sampling_y[:num_queries_to_do], sampling_var[:num_queries_to_do]


def latent_sampling_CMA_ES(object, model_path, vocab, num_queries_to_do, init_points, bounds, varthreshold, tkwargs, process_num, sigma=None, use_filter=False):
    object.eval()
    object.to(**tkwargs)
    sampling_z, sampling_x, sampling_y, sampling_var = [], [], [], []

    class SamplingProblem(Problem):
        def __init__(self, n_var, object, xl, xu, tkwargs):
            super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
            self.object = object
            self.tkwargs = tkwargs

        def  _evaluate(self, x, out, *args, **kwargs):
            x_tensor = torch.from_numpy(x).to(**self.tkwargs)
            posterior = self.object.posterior(x_tensor.to(**self.tkwargs))
            out['F'] = -posterior.mean.view(-1).detach().cpu().numpy()
            out['var'] = posterior.variance.view(-1).detach().cpu().numpy()
    
    cmaes_start = time.time()
    prob = SamplingProblem(n_var=init_points.shape[1], object=object, xl=0.0, xu=1.0, tkwargs=tkwargs)
    n_gen = num_queries_to_do // len(init_points) + 1

    mean_var_list = []

    for x0 in init_points:
        zs, ys, var = [], [], []
        cmaes = CMAES(x0=x0, sigma=sigma)
        termination = get_termination("n_gen", 2*n_gen)
        cmaes.setup(problem=prob, termination=termination)
        try:
            while cmaes.has_next() and len(zs) < n_gen:
                p = cmaes.opt
                if p is None:
                    cmaes.next()
                    continue
                z = p.get('X')
                y = -p.get('F').squeeze()
                v = p.get('var').squeeze()

                mean_var_list.append(v)
                if use_filter and varthreshold is not None:
                    if v <= varthreshold:
                        zs.append(z)
                        ys.append(y)
                        var.append(v)
                else:    
                    zs.append(z)
                    ys.append(y)
                    var.append(v)
                cmaes.next()  
        except Exception as e:
            print("\n x0:\n", x0)
            print("\n run cmaes to sampling has error: \n", e)
            continue

        if len(zs) != 0:
            zs = np.vstack(zs)
            ys = np.hstack(ys)
            var = np.hstack(var)

        if zs is not None and len(zs) != 0 :
            if sampling_z is None or len(sampling_z) == 0:
                sampling_z = zs
                sampling_y = ys 
                sampling_var = var
            else:
                sampling_z = np.concatenate([sampling_z, zs]) 
                sampling_y = np.concatenate([sampling_y, ys])  
                sampling_var = np.concatenate([sampling_var, var])
                sampling_z, unique_indx = np.unique(sampling_z, axis=0, return_index=True)
                sampling_y = sampling_y[unique_indx]
                sampling_var = sampling_var[unique_indx]
    print("\n ga spend time: ", time.time()-cmaes_start)
    if sampling_z is not None and len(sampling_z) != 0:
        zs=unnormalize(torch.from_numpy(sampling_z).to(**tkwargs), bounds.to(**tkwargs)).cpu()
        sampling_x = []
        task_num = len(zs)//process_num + 1
        task_function = functools.partial(
            _batch_decode_z_wo_props_for_sample,
            model_path = model_path,
            vocab = vocab,
            tkwargs = tkwargs,
        )
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=process_num) as executor:
                futures = [executor.submit(task_function, zs[i*task_num:(i+1)*task_num]) for i in range(process_num)]
                for future in futures:            
                    result = future.result()
                    sampling_x.extend(result)
        except Exception as exc:
            print(f'Task raised an exception: {exc}')
        finally:
            executor.shutdown()
        sampling_x = np.array(sampling_x)

    var_mean = np.array(mean_var_list).mean()
    if varthreshold is None:
        varthreshold = var_mean
    else:
        varthreshold = 0.9 * varthreshold + 0.1 * var_mean

    object.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return varthreshold, sampling_z[:num_queries_to_do], sampling_x[:num_queries_to_do], sampling_y[:num_queries_to_do], sampling_var[:num_queries_to_do]


def increment_beta_pseudo_loss(beta_pseudo_loss, beta_pseudo_loss_final, ret_idx, start_num_retrain, beta_step_freq, beta_step):
    # Check if the warmup is over and if it's the right step to increment beta 
    if (
        ret_idx > start_num_retrain
        and (ret_idx-start_num_retrain) % beta_step_freq == 0
    ):
        beta = min(beta_pseudo_loss_final, beta_pseudo_loss * beta_step)
    else:
        beta = beta_pseudo_loss
    return beta


if __name__ == "__main__":
    mp.set_start_method('spawn')
    rdkit_quiet()
    main()
