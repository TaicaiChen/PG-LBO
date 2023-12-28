# -*- coding: utf-8 -*-
import argparse
import gc
import glob
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pdb
import numpy as np
import pytorch_lightning as pl
import torch
from botorch.models.transforms import Standardize
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.utils.errors import NotPSDError
from torch import Tensor
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.factory import get_termination, get_selection
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.mutation.pm import PolynomialMutation

import warnings
warnings.filterwarnings("ignore")

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from weighted_retraining.weighted_retraining.robust_opt_scripts.utils import is_robust
from weighted_retraining.weighted_retraining.bo_torch.mo_acquisition import bo_mo_loop

from utils.utils_cmd import parse_list
from weighted_retraining.weighted_retraining.partial_train_scripts import partial_train_topology_pseudo

from weighted_retraining.weighted_retraining.bo_torch.gp_torch import (
    bo_loop, add_gp_torch_args, gp_torch_train, gp_fit_test)

from weighted_retraining.weighted_retraining.bo_torch.utils import put_max_in_bounds

from utils.utils_save import str_dict
from weighted_retraining.weighted_retraining.utils import print_flush, DataWeighter

from utils.utils_save import get_storage_root, save_w_pickle
from weighted_retraining.weighted_retraining.topology.topology_data_pseudo import (
    WeightedNumpyDataset, 
    PseudoWeightedNumpyDataset, 
    Pseudo2WeightedNumpyDataset)
from weighted_retraining.weighted_retraining.topology.topology_dataset import (
    score_function,
    get_topology_dataset_path, get_topology_target_path,
    get_topology_binary_dataset_path, get_topology_binary_target_path)

from weighted_retraining.weighted_retraining.topology.topology_utils import (
    get_seed_encodings,
    get_latent_encodings, 
    get_rec_x_error,
    append_trainset_torch)
from weighted_retraining.weighted_retraining.topology.topology_model_pseudo import TopologyVAE
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.robust_opt_scripts import base as wr_base


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, cuda):
    # Make sure logs don't get in the way of progress bars
    pl._logger.setLevel(logging.CRITICAL)
    train_pbar = utils.SubmissivePlProgressbar(process_position=1)

    # Create custom saver and logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/val", )

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = num_epochs
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
        gradient_clip_val=20.0,
    )
    # Fit model
    trainer.fit(model, datamodule)


def get_pretrained_model_path(version: int, k, predict_target: bool,
                              beta_final: float, beta_target_pred_loss: float,
                              n_max_epochs: int,
                              latent_dim: int, use_binary_data: bool,
                              hdims: List[int] = None):
    model_path = os.path.join(
        partial_train_topology_pseudo.topology_get_path(
            k=k,
            predict_target=predict_target,
            hdims=hdims,
            beta_final=beta_final,
            beta_target_pred_loss=beta_target_pred_loss,
            n_max_epochs=n_max_epochs,
            use_binary_data=use_binary_data,
            latent_dim=latent_dim,
        ),
        f'lightning_logs/version_{version}/checkpoints/', 'best.ckpt')
    paths = glob.glob(model_path)
    assert len(paths) > 0, model_path
    return paths[0]


def get_root_path(lso_strategy: str, weight_type, k, r,
                  predict_target, latent_dim: int, hdims,
                  beta_final: float,
                  beta_target_pred_loss: float,
                  acq_func_id: str, acq_func_kwargs: Dict[str, Any], covar_name: str, input_wp,
                  random_search_type: Optional[str],
                  n_max_epochs: int,
                  estimate_rec_error: bool, cost_aware_gamma_sched: Optional[str], use_binary_data: bool,
                  pseudo_sampling_type: Optional[str],
                  semi_supervised: Optional[bool] = False,
                  use_pseudo: Optional[bool] = False, pseudo_data_size: int=5000,
                  pseudo_beta: float=0.5, beta_pseudo_loss_start: Optional[float] = 0.1, 
                  beta_pseudo_loss_final: Optional[float] = 1.0,
                  use_ssdkl: Optional[bool] = False, beta_ssdkl_loss: Optional[float] = 1.0):
    result_path = os.path.join(
        get_storage_root(),
        f"logs/opt/topology/{weight_type}/k_{k}/r_{r}")
    exp_spec = f'topo'
    if latent_dim != 2:
        exp_spec += f'-z_dim_{latent_dim}'
    if predict_target:
        assert hdims is not None
        exp_spec += '-predy_' + '_'.join(map(str, hdims))
        exp_spec += f'-b_{float(beta_target_pred_loss):g}'
    exp_spec += f'-bkl_{beta_final}'
    if use_binary_data:
        exp_spec += "-binary_data"
    if semi_supervised:
        exp_spec += "-semi_supervised"
    if lso_strategy == 'opt':
        acq_func_spec = f"{acq_func_id}_{covar_name}{'_inwp_' if input_wp else str(input_wp)}"
        if 'ErrorAware' in acq_func_id and cost_aware_gamma_sched is not None:
            acq_func_spec += f"_sch-{cost_aware_gamma_sched}"
        if len(acq_func_kwargs) > 0:
            acq_func_spec += f'_{str_dict(acq_func_kwargs)}'
        if is_robust(acq_func_id):
            if estimate_rec_error:
                acq_func_spec += "_rec-est_"
        if use_pseudo:
            exp_spec += "-use_pseudo"
            exp_spec += f'-bls_{float(beta_pseudo_loss_start):g}'
            exp_spec += f'-blf_{float(beta_pseudo_loss_final):g}'
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


def get_path(lso_strategy: str, weight_type, k, r, predict_target,
             latent_dim: int, hdims,
             beta_final: float,  beta_target_pred_loss: float,
             acq_func_id: str, acq_func_kwargs: Dict[str, Any], covar_name: str,
             input_wp, seed, random_search_type: Optional[str], n_max_epochs: int,
             estimate_rec_error: bool, cost_aware_gamma_sched: Optional[str], use_binary_data: bool,
             pseudo_sampling_type: Optional[str], semi_supervised: Optional[bool] = False, 
             use_pseudo: Optional[bool] = False, pseudo_data_size: int=5000, 
             pseudo_beta: float=0.5, beta_pseudo_loss_start: Optional[float] = 0.1, beta_pseudo_loss_final: Optional[float] = 1.0,
             use_ssdkl: Optional[bool] = False, beta_ssdkl_loss: Optional[float] = 1.0):
    result_path = get_root_path(
        lso_strategy=lso_strategy,
        weight_type=weight_type,
        k=k,
        r=r,
        predict_target=predict_target,
        latent_dim=latent_dim,
        hdims=hdims,
        beta_target_pred_loss=beta_target_pred_loss,
        beta_final=beta_final,
        acq_func_id=acq_func_id,
        acq_func_kwargs=acq_func_kwargs,
        covar_name=covar_name,
        input_wp=input_wp,
        random_search_type=random_search_type,
        n_max_epochs=n_max_epochs,
        estimate_rec_error=estimate_rec_error,
        cost_aware_gamma_sched=cost_aware_gamma_sched,
        semi_supervised=semi_supervised,
        use_binary_data=use_binary_data,
        use_pseudo=use_pseudo,
        pseudo_data_size=pseudo_data_size,
        pseudo_beta=pseudo_beta,
        beta_pseudo_loss_start=beta_pseudo_loss_start,
        beta_pseudo_loss_final=beta_pseudo_loss_final,
        pseudo_sampling_type=pseudo_sampling_type,
        use_ssdkl=use_ssdkl,
        beta_ssdkl_loss=beta_ssdkl_loss
    )
    result_path = os.path.join(result_path, f'seed{seed}')
    return result_path


# main
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.register('type', list, parse_list)
    
    parser = WeightedNumpyDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    parser = wr_base.add_common_args(parser)
    parser = add_gp_torch_args(parser)


    opt_group = parser.add_argument_group(title="opt-model")
    opt_group.add_argument(
        '--use_test_set',
        dest="use_test_set",
        action="store_true",
        help="flag to use a test set for evaluating the sparse GP"
    )
    opt_group.add_argument(
        '--use_full_data_for_gp',
        dest="use_full_data_for_gp",
        action="store_true",
        help="flag to use the full dataset for training the GP"
    )
    opt_group.add_argument(
        "--n_decode_attempts",
        type=int,
        default=100,
        help="number of decoding attempts",
    )
    opt_group.add_argument(
        "--cuda",
        type=int,
        default=None,
        help="cuda ID",
    )
    opt_group.add_argument(
        "--data_seed",
        type=int,
        required=True,
        help="Seed that has been used to generate the dataset"
    )
    opt_group.add_argument(
        "--input_wp",
        action='store_true',
        help="Whether to apply input warping"
    )
    opt_group.add_argument(
        "--predict_target",
        action='store_true',
        help="Generative model predicts target value",
    )
    opt_group.add_argument(
        "--target_predictor_hdims",
        type=list,
        default=None,
        help="Hidden dimensions of MLP predicting target values",
    )
    opt_group.add_argument(
        "--latent_dim",
        type=int,
        default=2,
        help="Hidden dimension the latent space",
    )
    opt_group.add_argument(
        "--use_binary_data",
        action='store_true',
        help="Binarize images in the dataset",
    )

    vae_group = parser.add_argument_group("Metric learning")
    vae_group.add_argument(
        "--training_max_epochs",
        type=int,
        required=True,
        help="Total number of training epochs the model has been trained on",
    )
    vae_group.add_argument(
        "--estimate_rec_error",
        action='store_true',
        help="Whether to estimate reconstruction error when new points are acquired",
    )
    vae_group.add_argument(
        "--cost_aware_gamma_sched",
        type=str,
        default=None,
        choices=(None, 'fixed', 'linear', 'reverse_linear', 'exponential', 'reverse_exponential', 'post_obj_var',
                 'post_obj_inv_var', 'post_err_var', 'post_err_inv_var', 'post_min_var', 'post_var_tradeoff',
                 'post_var_inv_tradeoff'),
        help="Schedule for error-aware acquisition function parameter `gamma`",
    )
    vae_group.add_argument(
        "--test_gp_error_fit",
        action='store_true',
        help="test the gp fit on the predicted reconstruction error on a validation set",
    )
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
        "--semi_supervised",
        action='store_true',
        help="Start BO from VAE trained with unlabelled data.",
    )
    vae_group.add_argument(
        "--n_init_bo_points",
        type=int,
        default=None,
        help="Number of data points to use at the start of the BO if using semi-supervised training of the VAE."
             "(We need at least SOME data to fit the GP(s) etc.)",
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
        "--pseudo_beta",
        type=float, default=0.5,
        help="in a batch use how many pseudo label data to train"
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
        choices=(None, 'cmaes', 'ga', 'noise', 'truncnorm'),
        help="Sampling for pseudo data points",
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

    assert is_robust(args.acq_func_id) or (
        not args.estimate_rec_error), 'Acquisition function is not robust so estimation of ' \
                                      'reconstruction error should not be activated'
    if 'ErrorAware' in args.acq_func_id:
        assert 'gamma' in args.acq_func_kwargs
        assert 'eps' in args.acq_func_kwargs
    elif 'MultiObjectiveErrorAware' in args.acq_func_id:
        assert 'gamma' in args.acq_func_kwargs
    else:
        assert args.cost_aware_gamma_sched is None, args.cost_aware_gamma_sched
    
    args.dataset_path = os.path.join(
        ROOT_PROJECT,
        get_topology_binary_dataset_path() if args.use_binary_data else get_topology_dataset_path()
    )

    if args.pretrained_model_file is not None:
        args.pretrained_model_file = os.path.join(get_storage_root(), args.pretrained_model_file)
    elif args.semi_supervised:
        args.pretrained_model_file = get_pretrained_model_path(
            version=args.version,
            k="inf",
            n_max_epochs=args.training_max_epochs,
            predict_target=False,
            latent_dim=args.latent_dim,
            hdims=args.target_predictor_hdims,
            beta_final=args.beta_final,
            beta_target_pred_loss=args.beta_target_pred_loss,
            use_binary_data=args.use_binary_data
        )
    else:
        args.pretrained_model_file = get_pretrained_model_path(
            version=args.version,
            k=args.rank_weight_k,
            n_max_epochs=args.training_max_epochs,
            predict_target=args.predict_target,
            hdims=args.target_predictor_hdims,
            latent_dim=args.latent_dim,
            beta_final=args.beta_final,
            beta_target_pred_loss=args.beta_target_pred_loss,
            use_binary_data=args.use_binary_data,
        )
    print('pretrained model path', args.pretrained_model_file)
    # Seeding
    pl.seed_everything(args.seed)
    result_dir = get_path(
        lso_strategy=args.lso_strategy,
        weight_type=args.weight_type,
        k=args.rank_weight_k,
        r=args.retraining_frequency,
        predict_target=args.predict_target,
        latent_dim=args.latent_dim,
        hdims=args.target_predictor_hdims,
        beta_final=args.beta_final,
        beta_target_pred_loss=args.beta_target_pred_loss,
        acq_func_id=args.acq_func_id,
        acq_func_kwargs=args.acq_func_kwargs,
        covar_name=args.covar_name,
        input_wp=args.input_wp,
        seed=args.seed,
        random_search_type=args.random_search_type,
        n_max_epochs=args.training_max_epochs,
        estimate_rec_error=args.estimate_rec_error,
        cost_aware_gamma_sched=args.cost_aware_gamma_sched,
        semi_supervised=args.semi_supervised,
        use_binary_data=args.use_binary_data,
        use_pseudo=args.use_pseudo,
        pseudo_data_size=args.pseudo_data_size,
        pseudo_beta=args.pseudo_beta,
        beta_pseudo_loss_start=args.beta_pseudo_loss, 
        beta_pseudo_loss_final=args.beta_pseudo_loss_final,
        pseudo_sampling_type=args.pseudo_sampling_type,
        use_ssdkl=args.use_ssdkl,
        beta_ssdkl_loss=args.beta_ssdkl_loss,
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
    f.write(logs)
    f.close()
    if exc is not None:
        raise exc


def main_aux(args, result_dir):
    """ main """
    device = args.cuda
    if device is not None:
        torch.cuda.set_device(device)
    tkwargs = {
        "dtype": torch.float,
        "device": torch.device(f"cuda:{device}" if torch.cuda.is_available() and device is not None else "cpu"),
    }

    # get initial dataset
    aug = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomRotation(90),
        transforms.RandomRotation(180),
        transforms.RandomVerticalFlip(0.5)
    ])

    datamodule = WeightedNumpyDataset(args, DataWeighter(args), transform=aug)
    datamodule.setup("fit", n_init_points=args.n_init_bo_points)

    cmd = ' '.join(sys.argv[1:])
    print_flush(f"{cmd}\n")
    
    assert os.path.exists(args.pretrained_model_file), args.pretrained_model_file
    model: TopologyVAE = TopologyVAE.load_from_checkpoint(args.pretrained_model_file)
    model.hparams.cuda = args.cuda
    model.beta = model.hparams.beta_final  # Override any beta annealing
    model.predict_target = args.predict_target
    model.hparams.predict_target = args.predict_target
    model.beta_target_pred_loss = args.beta_target_pred_loss
    model.hparams.beta_target_pred_loss = args.beta_target_pred_loss
    if model.predict_target and model.target_predictor is None:
        model.target_predictor_hdims = args.target_predictor_hdims
        model.hparams.target_predictor_hdims = args.target_predictor_hdims
        model.build_target_predictor()
    model.use_pseudo = args.use_pseudo
    model.hparams.use_pseudo = args.use_pseudo
    model.beta_pseudo_loss = args.beta_pseudo_loss
    model.hparams.beta_pseudo_loss = args.beta_pseudo_loss
    model.use_ssdkl = args.use_ssdkl
    model.hparams.use_ssdkl = args.use_ssdkl
    model.beta_ssdkl_loss = args.beta_ssdkl_loss
    model.hparams.beta_ssdkl_loss = args.beta_ssdkl_loss

    model.eval()

    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))
    print("retrain number: {}".format(num_retrain))
    postfix = dict(
        retrain_left=num_retrain,
        best=float(datamodule.prop_train.max()),
        n_train=len(datamodule.data_train),
        save_path=result_dir
    )

    VarThreshold = None
    start_num_retrain = 0

    results = dict(
        opt_points=[],
        opt_point_properties=[],
        opt_point_errors=[],
        opt_model_version=[], 
        params=str(sys.argv),
        sample_points=[],
        sample_versions=[],
        sample_properties=[],
        rand_point_due_bo_fail=[], 
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
            if 'rand_point_due_bo_fail' not in results:
                results['rand_point_due_bo_fail'] = [0] * len(results['opt_points'])
        start_num_retrain = results['opt_model_version'][-1] + 1
        prev_retrain_model = args.retraining_frequency * (start_num_retrain - 1)

        num_sampled_points = len(results['opt_points'])

        VarThreshold = results['var_threshold_list'][-1]

        if args.n_init_retrain_epochs == 0 and prev_retrain_model == 0:
            pretrained_model_path = args.pretrained_model_file
        else:
            pretrained_model_path = os.path.join(result_dir, 'retraining', f'retrain_{prev_retrain_model}',
                                                 'checkpoints', 'last.ckpt')

        ckpt = torch.load(pretrained_model_path, map_location=f"cuda:{args.cuda}" if args.cuda else f"cpu")
        ckpt['hyper_parameters']['hparams'].beta_target_pred_loss = args.beta_target_pred_loss
        if args.predict_target:
            ckpt['hyper_parameters']['hparams'].predict_target = True
            ckpt['hyper_parameters']['hparams'].target_predictor_hdims = args.target_predictor_hdims

        if args.use_pseudo:
            ckpt['hyper_parameters']['hparams'].use_pseudo = args.use_pseudo
            ckpt['hyper_parameters']['hparams'].beta_pseudo_loss = args.beta_pseudo_loss

        if args.use_ssdkl:
            ckpt['hyper_parameters']['hparams'].use_ssdkl = args.use_ssdkl
            ckpt['hyper_parameters']['hparams'].beta_ssdkl_loss = args.beta_ssdkl_loss
        
        torch.save(ckpt, pretrained_model_path)
        model.load_from_checkpoint(pretrained_model_path, strict=False)
        if args.predict_target and not hasattr(model.hparams, "predict_target"):
            model.hparams.predict_target = args.predict_target
            model.hparams.target_predictor_hdims = args.target_predictor_hdims
        model.hparams.cuda = args.cuda
        model.beta = model.hparams.beta_final  # Override any beta annealing
        model.eval()


        num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency)) - start_num_retrain

        datamodule.append_train_data(np.array(results['opt_points']),
                                     np.array(results['opt_point_properties']))
        postfix = dict(
            retrain_left=num_retrain,
            best=float(datamodule.prop_train.max()),
            n_train=len(datamodule.train_dataset),
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
            
            if (model.predict_target or model.use_ssdkl):
                model.training_m = datamodule.training_m
                model.training_M = datamodule.training_M
                model.validation_m = datamodule.validation_m
                model.validation_M = datamodule.validation_M

            pbar.set_postfix(postfix)
            pbar.set_description("retraining")
            print(result_dir)

            samples_so_far = args.retraining_frequency * ret_idx

            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                num_epochs = args.n_init_retrain_epochs
            
            retrain_start = time.time()
            if num_epochs > 0:
                retrain_dir = os.path.join(result_dir, "retraining")
                version = f"retrain_{samples_so_far}"
                if PseudoDatamodule is None:
                    model.use_pseudo = False
                    model.hparams.use_pseudo = False
                    if GPLayer is not None:
                        model.build_gp_predictor(GPLayer, GPkwargs)
                    retrain_model(
                        model=model,
                        datamodule=datamodule,
                        save_dir=retrain_dir,
                        version_str=version,
                        num_epochs=num_epochs,
                        cuda=args.cuda
                    )
                else:
                    model.use_pseudo = True
                    model.hparams.use_pseudo = True
                    beta_pseudo_loss=increment_beta_pseudo_loss(
                        model.beta_pseudo_loss, 
                        args.beta_pseudo_loss_final, 
                        ret_idx, 
                        start_num_retrain, 1, 0.065)
                    model.beta_pseudo_loss = beta_pseudo_loss
                    model.hparams.beta_pseudo_loss = beta_pseudo_loss
                    if GPLayer is not None:
                        model.build_gp_predictor(GPLayer, GPkwargs)
                    retrain_model(
                        model=model,
                        datamodule=PseudoDatamodule,
                        save_dir=retrain_dir,
                        version_str=version,
                        num_epochs=num_epochs,
                        cuda=args.cuda
                    )
                model.eval()
            del num_epochs
            print("\n retrain spend time: ", time.time()-retrain_start)

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            if args.samples_per_model > 0:
                pbar.set_description("sampling")
                sample_x, sample_y, _ = latent_sampling(
                    model, num_queries_to_do=args.samples_per_model,
                    use_binary_data=args.use_binary_data,
                    property_key=args.property_key
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
            error_new = None
            if args.lso_strategy == "opt":
                gp_dir = os.path.join(result_dir, "gp", f"iter{samples_so_far}")
                os.makedirs(gp_dir, exist_ok=True)
                gp_data_file = os.path.join(gp_dir, "data.npz")
                x_new, y_new, rand_point_due_bo_fail, error_new, pseudodatamodule, varthreshold, gp_model, gpkwargs = latent_optimization(
                    args=args,
                    result_dir=result_dir,
                    samples_so_far=samples_so_far,
                    varthreshold=VarThreshold,
                    GPFile=GPFile,
                    model=model,
                    datamodule=datamodule,
                    n_inducing_points=args.n_inducing_points,
                    n_best_points=args.n_best_points,
                    n_rand_points=args.n_rand_points,
                    tkwargs=tkwargs,
                    num_queries_to_do=num_queries_to_do,
                    use_test_set=args.use_test_set,
                    use_full_data_for_gp=args.use_full_data_for_gp,
                    gp_data_file=gp_data_file,
                    gp_run_folder=gp_dir,
                    test_gp_error_fit=args.test_gp_error_fit,
                    scale=args.scale,
                    covar_name=args.covar_name,
                    acq_func_id=args.acq_func_id,
                    acq_func_kwargs=args.acq_func_kwargs,
                    acq_func_opt_kwargs=args.acq_func_opt_kwargs,
                    q=1, 
                    num_restarts=args.num_restarts,
                    raw_initial_samples=args.raw_initial_samples,
                    num_MC_sample_acq=args.num_MC_sample_acq,
                    property_key=args.property_key,
                    input_wp=args.input_wp,
                    estimate_rec_error=args.estimate_rec_error,
                    cost_aware_gamma_sched=args.cost_aware_gamma_sched,
                    pbar=pbar,
                    postfix=postfix,
                    semi_supervised=args.semi_supervised,
                    use_binary_data=args.use_binary_data,
                )

            elif args.lso_strategy == "sample":
                x_new, y_new, z_query = latent_sampling(
                    model, num_queries_to_do, args.property_key, use_binary_data=args.use_binary_data,
                )
                rand_point_due_bo_fail = [0] * num_queries_to_do

            elif args.lso_strategy == "random_search":
                x_new, y_new = latent_random_search(
                    model=model, num_queries_to_do=num_queries_to_do, tkwargs=tkwargs,
                    datamodule=datamodule, property_key=args.property_key, filter_unique=False,
                    use_binary_data=args.use_binary_data, random_search_type=args.random_search_type,
                    seed=args.seed, fast_forward=samples_so_far
                )
                rand_point_due_bo_fail = [0] * num_queries_to_do
            else:
                raise NotImplementedError(args.lso_strategy)

            VarThreshold = varthreshold
            # Append new points to dataset
            datamodule.append_train_data(x_new, y_new)
            # Save a new dataset
            new_data_file = os.path.join(result_dir, f"train_data_iter{samples_so_far + num_queries_to_do}")
            np.savez_compressed(
                str(new_data_file),
                data=datamodule.data_train,
                **{args.property_key: datamodule.prop_train},
            )


            results["opt_points"] += list(x_new)
            results["opt_point_properties"] += list(y_new)
            if error_new is not None:
                results['opt_point_errors'] += list(error_new)
            results["opt_model_version"] += [ret_idx] * len(x_new)
            results["rand_point_due_bo_fail"] += rand_point_due_bo_fail

            results["var_threshold_list"] += [varthreshold]
            np.savez_compressed(os.path.join(result_dir, "results.npz"), **results)

            postfix["best"] = max(postfix["best"], float(y_new.max()))
            postfix["n_train"] = len(datamodule.data_train)
            pbar.set_postfix(postfix)
            

            if args.use_pseudo and pseudodatamodule is not None:
                PseudoDatamodule = Pseudo2WeightedNumpyDataset(args)
                PseudoDatamodule.setup(LabelDataModule=datamodule, PseudoDataModule=pseudodatamodule)

            if args.use_ssdkl and gp_model is not None:
                GPLayer = gp_model
                GPkwargs = gpkwargs


    print_flush("=== DONE ({:.3f}s) ===".format(time.time() - start_time))
    np.save(os.path.join(result_dir, "x_hat_star"), datamodule.data_train[datamodule.prop_train.argmax()])


def latent_optimization(
        args,
        result_dir,
        samples_so_far,
        varthreshold,
        GPFile,
        model: TopologyVAE,
        datamodule: WeightedNumpyDataset,
        n_inducing_points: int,
        n_best_points: int,
        n_rand_points: int,
        tkwargs: Dict[str, Any], 
        num_queries_to_do: int,
        use_test_set: bool,
        use_full_data_for_gp: bool,
        gp_data_file: str,
        gp_run_folder: str,
        test_gp_error_fit: bool,
        scale: bool,
        covar_name: str, 
        acq_func_id: str,
        acq_func_kwargs: Dict[str, Any],
        acq_func_opt_kwargs: Dict[str, Any],
        q: int,
        num_restarts: int,
        raw_initial_samples: int,
        num_MC_sample_acq: int,
        property_key: str,
        input_wp: bool,
        estimate_rec_error: bool,
        cost_aware_gamma_sched: Optional[str],
        use_binary_data: bool,
        pbar=None,
        postfix=None,
        semi_supervised: bool = False,
):
    model.to(**tkwargs)
    X_train, y_train, _, _, _, y_mean, _, y_std, train_inds, _ = get_latent_encodings(
        use_test_set=use_test_set,
        use_full_data_for_gp=use_full_data_for_gp,
        model=model,
        data_file=gp_data_file,
        data_imgs=datamodule.data_train,
        data_scores=datamodule.prop_train,
        n_best=n_best_points,
        n_rand=n_rand_points,
        tkwargs=tkwargs,
        return_inds=True
    )

    X_train = torch.tensor(X_train).to(**tkwargs)
    y_train = torch.tensor(y_train).to(**tkwargs)

    do_robust = is_robust(acq_func_id)
    error_train: Optional[Tensor] = None
    if do_robust:
        error_train = get_rec_x_error(model, tkwargs=tkwargs,
                                      xs=torch.from_numpy(datamodule.data_train[train_inds]),
                                      zs=X_train)

        assert error_train.shape == y_train.shape == (len(X_train), 1), (error_train.shape, y_train.shape)

    model.cpu()  
    torch.cuda.empty_cache()

    curr_gp_file = None
    curr_gp_error_file = None

    all_new_imgs = []
    all_new_scores = []
    all_new_errors = [] if do_robust else None
    all_z_opt_noise = []
    all_img_noise = []
    all_score_noise = []

    n_rand_acq = 0  

    rand_point_due_bo_fail = []
    
    gp_iter = 0
    redo_counter = 1

    acq_sampling_z = []
    while len(all_new_imgs) < num_queries_to_do:
        # Part 1: fit GP
        # ===============================

        new_gp_file = os.path.join(gp_run_folder, f"gp_train_res{gp_iter:04d}.npz")
        new_gp_error_file = os.path.join(gp_run_folder, f"gp_train_error_res{gp_iter:04d}.npz")
        iter_seed = int(np.random.randint(10000))

        gp_file = None
        gp_error_file = None
        if gp_iter == 0:
            # Add commands for initial fitting
            gp_fit_desc = "GP initial fit"
            current_n_inducing_points = min(X_train.shape[0], n_inducing_points)
            if GPFile is not None:
                gp_file = GPFile
        else:

            gp_fit_desc = "GP incremental fit"
            gp_file = curr_gp_file
            gp_error_file = curr_gp_error_file
    
        init = (gp_iter == 0 and gp_file is None)

        if X_train.shape[0] == n_inducing_points:
            current_n_inducing_points = n_inducing_points
            init = True

        old_desc = None
        # Set pbar status for user
        if pbar is not None:
            old_desc = pbar.desc
            pbar.set_description(gp_fit_desc)

        np.random.seed(iter_seed)
        torch.manual_seed(iter_seed)

        y_mean, y_std = y_train.mean(), y_train.std()
        y_train_step = (y_train - y_mean) / (y_std + 1e-8)

        bounds = torch.zeros(2, X_train.shape[1], **tkwargs)

        bounds[0] = torch.quantile(X_train, .0005, dim=0)
        bounds[1] = torch.quantile(X_train, .9995, dim=0)
        ybounds = torch.zeros(2, y_train.shape[1], **tkwargs)
        ybounds[0] = torch.quantile(y_train, .0005, dim=0)
        ybounds[1] = torch.quantile(y_train, .9995, dim=0)
        ydelta = .05 * (ybounds[1] - ybounds[0])
        ybounds[0] -= ydelta
        ybounds[1] += ydelta


        y_train_std = y_train.add(-y_train.mean()).div(y_train.std())
        y_train_normalized = normalize(y_train, ybounds)  # minimize
        bounds = put_max_in_bounds(X_train, y_train_std, bounds)

        delta = .05 * (bounds[1] - bounds[0])
        bounds[0] -= delta
        bounds[1] += delta
        print(f"Using data bound of\n {bounds}...")
        
        train_x = normalize(X_train, bounds)

        try:
            gp_model = gp_torch_train(
                train_x=train_x,
                train_y=y_train,
                n_inducing_points=current_n_inducing_points,
                tkwargs=tkwargs,
                init=init,
                scale=scale,
                covar_name=covar_name,
                gp_file=gp_file,
                save_file=new_gp_file,
                input_wp=input_wp,
                outcome_transform=None,
                options={'lr': 5e-3, 'maxiter': 500} if init else {'lr': 5e-3, 'maxiter': 100}
            )
        except (RuntimeError, NotPSDError) as e:  # Random acquisition
            if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                if e.args[0][:7] not in ['symeig_', 'cholesk']:
                    raise
            print_flush(f"\t\tNon PSD Error in GP fit. Re-fitting objective GP from scratch...")
            gp_model = gp_torch_train(
                train_x=train_x,
                train_y=y_train,
                n_inducing_points=current_n_inducing_points,
                tkwargs=tkwargs,
                init=True,
                scale=scale,
                covar_name=covar_name,
                gp_file=gp_file,
                save_file=new_gp_file,
                input_wp=input_wp,
                outcome_transform=None,
                options={'lr': 5e-3, 'maxiter': 500}
            )
        curr_gp_file = new_gp_file
        

        with torch.no_grad():
            y_pred_var = gp_model.posterior(train_x).variance
            yvarbounds = torch.zeros(2, y_train.shape[1], **tkwargs)
            yvarbounds[0] = torch.quantile(y_pred_var, .0005, dim=0)
            yvarbounds[1] = torch.quantile(y_pred_var, .9995, dim=0)
            yvardelta = .05 * (yvarbounds[1] - yvarbounds[0])
            yvarbounds[0] -= yvardelta
            yvarbounds[1] += yvardelta

        if do_robust:
            if estimate_rec_error or init:
                rbounds = torch.zeros(2, error_train.shape[1], **tkwargs)
                rbounds[0] = torch.quantile(error_train, .0005, dim=0)
                rbounds[1] = torch.quantile(error_train, .9995, dim=0)
                rdelta = .05 * (rbounds[1] - rbounds[0])
                rbounds[0] -= rdelta
                rbounds[1] += rdelta
                error_train_normalized = normalize(error_train, rbounds)
                error_train_std = error_train.add(-error_train.mean()).div(error_train.std())
                try:
                    gp_model_error = gp_torch_train(
                        train_x=train_x,
                        train_y=error_train,
                        n_inducing_points=current_n_inducing_points,
                        tkwargs=tkwargs,
                        init=init,
                        scale=scale,
                        covar_name=covar_name,
                        gp_file=gp_error_file,
                        save_file=new_gp_error_file,
                        input_wp=input_wp,
                        outcome_transform=Standardize(m=1),
                        options={'lr': 5e-3, 'maxiter': 500} if init else {'lr': 5e-3, 'maxiter': 100}
                    )
                except (RuntimeError, NotPSDError) as e:
                    if isinstance(e, RuntimeError) and not isinstance(e, NotPSDError):
                        if e.args[0][:7] not in ['symeig_', 'cholesk']:
                            raise
                    print_flush(f"\t\tNon PSD Error in GP fit. Re-fitting error GP from scratch...")
                    print_flush("Error in GP training, restart from scratch...")
                    gp_model_error = gp_torch_train(
                        train_x=train_x,
                        train_y=error_train,
                        n_inducing_points=current_n_inducing_points,
                        tkwargs=tkwargs,
                        init=True,
                        scale=scale,
                        covar_name=covar_name,
                        gp_file=gp_error_file,
                        save_file=new_gp_error_file,
                        input_wp=input_wp,
                        outcome_transform=Standardize(m=1),
                        options={'lr': 5e-3, 'maxiter': 500}
                    )
                curr_gp_error_file = new_gp_error_file

                with torch.no_grad():
                    r_pred_var = gp_model_error.posterior(train_x).variance
                    rvarbounds = torch.zeros(2, error_train.shape[1], **tkwargs)
                    rvarbounds[0] = torch.quantile(r_pred_var, .0005, dim=0)
                    rvarbounds[1] = torch.quantile(r_pred_var, .9995, dim=0)
                    rvardelta = .05 * (rvarbounds[1] - rvarbounds[0])
                    rvarbounds[0] -= rvardelta
                    rvarbounds[1] += rvardelta
        else:
            gp_model_error = None
        
        # Part 2: optimize GP acquisition func to query point
        # ===============================
        if pbar is not None:
            pbar.set_description("optimizing acq func")

        print_flush(f"\n\t\t Picking new inputs nb. {gp_iter + 1} via optimization...")
        optimization_start = time.time()
        try: 
            if do_robust:
                if cost_aware_gamma_sched is not None:
                    # assert isinstance(acq_func_kwargs['gamma'], float) or isinstance(acq_func_kwargs['gamma'], int), acq_func_kwargs['gamma']
                    if 'gamma_start' not in acq_func_kwargs:
                        acq_func_kwargs['gamma_start'] = float(acq_func_kwargs['gamma'])
                    if cost_aware_gamma_sched == 'linear':
                        acq_func_kwargs['gamma'] = (num_queries_to_do - len(all_new_scores)) / num_queries_to_do * \
                                                   acq_func_kwargs['gamma_start']
                    elif cost_aware_gamma_sched == 'reverse_linear':
                        acq_func_kwargs['gamma'] = len(all_new_scores) / num_queries_to_do * acq_func_kwargs[
                            'gamma_start']
                    elif cost_aware_gamma_sched == 'exponential':
                        acq_func_kwargs['gamma'] = 0.75 ** len(all_new_scores) * acq_func_kwargs['gamma_start']
                    elif cost_aware_gamma_sched == 'reverse_exponential':
                        acq_func_kwargs['gamma'] = 0.75 ** (num_queries_to_do - len(all_new_scores)) * acq_func_kwargs[
                            'gamma_start']
                    elif cost_aware_gamma_sched == 'fixed':
                        acq_func_kwargs['gamma'] = acq_func_kwargs['gamma_start']
                    elif cost_aware_gamma_sched == 'post_obj_var' \
                            or cost_aware_gamma_sched == 'post_obj_inv_var' \
                            or cost_aware_gamma_sched == 'post_err_var' \
                            or cost_aware_gamma_sched == 'post_err_inv_var' \
                            or cost_aware_gamma_sched == 'post_min_var' \
                            or cost_aware_gamma_sched == 'post_var_tradeoff' \
                            or cost_aware_gamma_sched == 'post_var_inv_tradeoff':
                        acq_func_kwargs.update({'gamma': cost_aware_gamma_sched})
                    else:
                        raise ValueError(cost_aware_gamma_sched)

                acq_func_kwargs['y_var_bounds'] = yvarbounds
                acq_func_kwargs['r_var_bounds'] = rvarbounds
                print(acq_func_kwargs)

                res = bo_mo_loop(
                    gp_model=gp_model,
                    gp_model_error=gp_model_error,
                    vae_model=model,
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
                    return_best_only=True
                )
                model.eval()
                z_opt = res     
            else:
                print('robust_opt_topology', acq_func_id)
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
                    return_best_only=True
                )
                z_opt = res
            acq_sampling_z.append(res.clone().detach().cpu())
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
        print(f"\n Optimization took {time.time() - optimization_start:.1f}s to finish...")

        model.cpu()
        gp_model.cpu()
        if do_robust:
            gp_model_error.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        z_opt = unnormalize(z_opt, bounds).cpu().detach()
        z_opt = torch.atleast_2d(z_opt)

        assert q == 1, q
        target = np.load(get_topology_binary_target_path()) if use_binary_data else np.load(get_topology_target_path())
        new_img, new_score = _decode_z_props(model=model, z=z_opt, target=target,
                                             filter_unique=False, property_key=property_key)

        if new_img is None or torch.isnan(z_opt).sum() > 0:
            if redo_counter == 3:
                print(f'Invalid! z_opt has Nans or decoded to None at iteration {gp_iter} even after {redo_counter} '
                      f'restarts -> moving on with a randomly picked point...')
                redo_counter = 1
                while new_img is None or torch.isnan(z_opt).sum() > 0:
                    z_opt = torch.rand(q, bounds.shape[1]).to(bounds)
                    new_img, new_score = _decode_z_props(model=model, z=z_opt, target=target,
                                                         filter_unique=False, property_key=property_key)
                rand_point_due_bo_fail += [1] * q
            else:
                print(f'Invalid! z_opt has Nans or decoded to None -> Re-doing iteration {gp_iter}')
                redo_counter += 1
                continue

        all_new_z_opt_noise = []
        all_new_img_noise = []
        all_new_score_noise = []
        checks = [1 if (new_img == img).all() else 0 for img in datamodule.data_train]
        seen = sum(checks) > 0
        
        checks_n = [1 if (new_img == img).all() else 0 for img in all_img_noise]
        seen_n = sum(checks_n) > 0
        
        if seen or seen_n:
            print(f"Img is already in training set, set perturb z_opt & decode until a new img...")
            all_new_z_opt_noise.append(z_opt)
            all_new_img_noise.append(new_img)
            all_new_score_noise.append(new_score)
            noise_level = 1e-1
            for random_trials in range(10):
                z_opt_noise = z_opt + torch.randn_like(z_opt) * noise_level
                new_img_noise = model.decode_deterministic(z=z_opt_noise).detach().squeeze(0).cpu().numpy()
                if new_img_noise is None:
                    print(f"... skipping perturbed point decoded to None ...")
                else:
                    new_score_noise = score_function(new_img_noise, target_image=target)
                    all_new_z_opt_noise.append(z_opt_noise)
                    all_new_img_noise.append(new_img_noise)
                    all_new_score_noise.append(new_score_noise)

                    checks_noise = [1 if (new_img_noise == img).all() else 0 for img in datamodule.data_train]
                    seen_noise = sum(checks_noise) > 0
                    
                    checks_noise_n = [1 if (new_img_noise == img).all() else 0 for img in all_img_noise]
                    seen_noise_n = sum(checks_noise_n) > 0
                    
                    if seen_noise or seen_noise_n:
                        print(f"... {random_trials+1}: img already in training set with score {new_score_noise}...")
                        noise_level *= 1.5
                    else:
                        z_opt = z_opt_noise
                        new_img = new_img_noise
                        new_score = score_function(new_img, target_image=target)
                        print(f'...after {random_trials + 1} perturbations got a new img with score {new_score}')
                        break
            if random_trials == 9:
                if redo_counter == 3:
                    print(f"Moving on anyway after redoing BO step {gp_iter} 3 times.")
                    redo_counter = 1
                else:
                    print(f"...did not find any new img after perturbing for {random_trials + 1} times "
                          f"-> Re-doing BO step {gp_iter} altogether.")
                    redo_counter += 1
                    continue

        all_new_z_opt_noise = [z_opt] if all_new_z_opt_noise == [] else all_new_z_opt_noise
        all_new_img_noise = [new_img] if all_new_img_noise == [] else all_new_img_noise
        all_new_score_noise = [new_score] if all_new_score_noise == [] else all_new_score_noise

        all_new_z_opt_noise = torch.cat(all_new_z_opt_noise).to(**tkwargs)
        all_new_img_noise = np.concatenate(all_new_img_noise)
        all_new_score_noise = np.array(all_new_score_noise).flatten().astype(np.float32)
        
        all_z_opt_noise = torch.cat([torch.tensor(all_z_opt_noise).to(**tkwargs), all_new_z_opt_noise]).to(**tkwargs)
        all_img_noise = np.concatenate([all_img_noise, all_new_img_noise]) if all_img_noise != [] else all_new_img_noise
        all_score_noise = np.concatenate([all_score_noise, all_new_score_noise])
        
        all_new_imgs.append(new_img)
        all_new_scores = np.append(all_new_scores, new_score)
        assert all_new_imgs[-1].shape[-2:] == (40, 40), all_new_imgs[-1].shape

        print_flush(f"\t\tPicked new input with value {all_new_scores[-1]}...")

        # Reset pbar description
        if pbar is not None:
            pbar.set_description(old_desc)
            pbar.update(len(z_opt))

            # Update best point in progress bar
            if postfix is not None:
                postfix["best"] = max(postfix["best"], float(max(all_new_scores)))
                pbar.set_postfix(postfix)

        if do_robust and estimate_rec_error:
            # add estimate new errors
            new_errors = get_rec_x_error(
                model=model,
                xs=torch.from_numpy(all_new_img_noise).to(**tkwargs),
                zs=all_new_z_opt_noise,
                tkwargs=tkwargs
            ).cpu().numpy()
            all_new_errors = np.append(all_new_errors, new_errors[-q:])

            new_errors = torch.from_numpy(new_errors).reshape(-1, 1)
        else:
            new_errors = None

        aux_res_datasets = append_trainset_torch(
            X_train, y_train,
            new_inputs=all_new_z_opt_noise,
            new_scores=torch.from_numpy(all_new_score_noise).reshape(-1, 1),
            y_errors=error_train, new_errors=new_errors
        )


        if new_errors is None:
            X_train, y_train = aux_res_datasets
        else:
            X_train, y_train, error_train = aux_res_datasets

        gp_iter += 1

    pseudodatamodule = None
    gpkwargs = None
    gpkwargs={
        'bounds': bounds,
        'target_mean': y_mean,
        'target_std': y_std,
        'use_std': False,
    }
    if args.use_pseudo:
        model.eval()
        model.to(**tkwargs)
        gp_model.eval()
        gp_model.to(**tkwargs)
        sampling_start = time.time()

        if args.pseudo_sampling_type in ['cmaes', 'ga', 'noise']:
            seed_z, seed_y, seed_inds = get_seed_encodings(
                model=model,
                data_imgs=datamodule.data_train[train_inds],
                data_scores=datamodule.prop_train[train_inds],
                n_best=args.pseudo_sampling_type_kw['n_best'],
                n_rand=args.pseudo_sampling_type_kw['n_rand'],
                tkwargs=tkwargs,
                return_inds=True
            )
            seed_z = torch.tensor(seed_z).to(bounds)
            seed_y = np.squeeze(seed_y)
            seed_x = datamodule.data_train[train_inds][seed_inds]
            seed_z = normalize(seed_z, bounds).cpu().squeeze().numpy()
            if args.pseudo_sampling_type_kw['use_bo']:
                bo_z = normalize(all_z_opt_noise, bounds).cpu().numpy()
                seed_z = np.concatenate([seed_z, bo_z])
                seed_y = np.concatenate([seed_y, all_score_noise])
                seed_x = np.concatenate([seed_x, all_img_noise])
            seed_posterior = gp_model.posterior(torch.from_numpy(seed_z).to(**tkwargs))
            seed_mean = seed_posterior.mean.view(-1).detach().cpu().numpy()
            seed_var = seed_posterior.variance.view(-1).detach().cpu().numpy()
            seed_file = os.path.join(result_dir, f"sampling_seed_iter{samples_so_far + num_queries_to_do}")
            np.savez_compressed(
                str(seed_file),
                data=seed_x,
                gp_mean=seed_mean,
                gp_var=seed_var,
                score=seed_y
            )
            if args.pseudo_sampling_type == 'ga':
                varthreshold, heuristic_sampling_z, heuristic_sampling_x, heuristic_sampling_y, heuristic_sampling_var = latent_sampling_GA(
                    object=gp_model, sigma=args.pseudo_sampling_type_kw['sigma'], init_points=seed_z, 
                    num_queries_to_do=args.pseudo_data_size, model=model, bounds=bounds, varthreshold=varthreshold, tkwargs=tkwargs, use_filter=args.pseudo_sampling_type_kw['use_filter'])
            elif args.pseudo_sampling_type == 'cmaes':
                varthreshold, heuristic_sampling_z, heuristic_sampling_x, heuristic_sampling_y, heuristic_sampling_var = latent_sampling_CMA_ES(
                    object=gp_model, sigma=args.pseudo_sampling_type_kw['sigma'], init_points=seed_z, 
                    num_queries_to_do=args.pseudo_data_size, model=model,  bounds=bounds, varthreshold=varthreshold, tkwargs=tkwargs, use_filter=args.pseudo_sampling_type_kw['use_filter'])
            elif args.pseudo_sampling_type == 'noise':
                varthreshold, heuristic_sampling_z, heuristic_sampling_x, heuristic_sampling_y, heuristic_sampling_var = latent_sampling_Noise(
                    object=gp_model, sigma=args.pseudo_sampling_type_kw['sigma'], init_points=seed_z, 
                    num_queries_to_do=args.pseudo_data_size, model=model, bounds=bounds, varthreshold=varthreshold, tkwargs=tkwargs, use_filter=args.pseudo_sampling_type_kw['use_filter'])
            need_to_rand_sampling = args.pseudo_data_size - len(heuristic_sampling_x)
            print('\n need_to_rand_sampling:', need_to_rand_sampling)
            if need_to_rand_sampling > 0 and need_to_rand_sampling != args.pseudo_data_size:
                rand_sampling_start = time.time()
                _, rand_sampling_x, rand_sampling_z, rand_sampling_y, rand_sampling_var=latent_sampling_wo_props(
                            object=gp_model, varthreshold=varthreshold, model=model, num_queries_to_do=2*need_to_rand_sampling, tkwargs=tkwargs,
                            bounds=bounds, filter_unique=True, use_filter=False, warm_up=False)
                print("\n rand sampling spend time: ", time.time()-rand_sampling_start)
                if rand_sampling_x is not None:
                    sort_index = np.argsort(rand_sampling_y, axis=0)[::-1]
                    sampling_x = np.vstack([heuristic_sampling_x, rand_sampling_x[sort_index[:need_to_rand_sampling]]])
                    sampling_y = np.hstack([heuristic_sampling_y, rand_sampling_y[sort_index[:need_to_rand_sampling]]])
                    sampling_var = np.hstack([heuristic_sampling_var, rand_sampling_var[sort_index[:need_to_rand_sampling]]])
                else:
                    sampling_x = heuristic_sampling_x
                    sampling_y = heuristic_sampling_y
                    sampling_var = heuristic_sampling_var
            else:
                sampling_x = heuristic_sampling_x
                sampling_y = heuristic_sampling_y
                sampling_var = heuristic_sampling_var
        elif args.pseudo_sampling_type == 'random':
            varthreshold, sampling_x, sampling_z, sampling_y, sampling_var = latent_sampling_wo_props(
                object=gp_model,
                varthreshold=varthreshold, 
                model=model, 
                num_queries_to_do=args.pseudo_data_size, 
                bounds=bounds, 
                tkwargs=tkwargs, 
                filter_unique=True,
                use_filter=args.pseudo_sampling_type_kw['use_filter'],
                warm_up=True)
        else:
            raise ValueError(f'{args.pseudo_sampling_type} not supported')        
        print("\n sampling point spend: ", time.time()-sampling_start)
        
        if sampling_x is not None and len(sampling_x) >= 20:
            pseudo_data_file = os.path.join(result_dir, f"pseudo_data_iter{samples_so_far + num_queries_to_do}")
            np.savez_compressed(
                str(pseudo_data_file),
                data=sampling_x,
                gp_mean=sampling_y,
                gp_var=sampling_var
            )

            pseudodatamodule = PseudoWeightedNumpyDataset(
                DataWeighter(args),
                val_frac=0.05,
                batch_size=min(int(args.batch_size*args.pseudo_beta), len(sampling_x)),
                property_key='gp_mean',
                predict_target=args.predict_target,
                use_ssdkl=args.use_ssdkl,
            ) 
            pseudodatamodule.setup(data=(sampling_x, sampling_y), stage="fit")
        
        model.cpu()
        gp_model.cpu()
        if do_robust:
            gp_model_error.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    if n_rand_acq / num_queries_to_do > .2:
        raise ValueError(
            f'Sampled too many random points ({n_rand_acq} / {num_queries_to_do}) due to exceptions in BO such as {exc}')
    elif n_rand_acq > 0:
        print(f'Acquired {n_rand_acq} / {num_queries_to_do} points at random due to bo acquisition failure {exc}')

    return np.array(all_new_imgs).reshape(-1, 40, 40), all_new_scores, rand_point_due_bo_fail, all_new_errors, pseudodatamodule, varthreshold, gp_model, gpkwargs


def latent_sampling(model, num_queries_to_do, property_key: str, use_binary_data: bool, filter_unique=True):
    """ 
    Draws samples from latent space and appends to the dataset 
    Args:
        model: generative model for LSO
        num_queries_to_do: number of queries
        property_key: str 
        use_binary_data: 
    """

    z_sample = torch.randn(num_queries_to_do, model.latent_dim, device=model.device)
    target = np.load(get_topology_binary_target_path()) if use_binary_data else np.load(get_topology_target_path())
    return _batch_decode_z_and_props(model, z_sample, target, property_key=property_key, filter_unique=filter_unique)


def latent_random_search(model: TopologyVAE, num_queries_to_do: int, tkwargs: Dict[str, Any],
                         datamodule: WeightedNumpyDataset, property_key: str, use_binary_data: bool,
                         filter_unique=False, random_search_type=None,
                         seed: int = None, fast_forward: int = None) -> Tuple[np.ndarray, np.ndarray]:
    print_flush(f"\t\tPicking new inputs via {random_search_type if random_search_type is not None else ''} RS...")

    model.eval()
    model.to(**tkwargs)

    bs = 5000
    n_batches = int(np.ceil(len(datamodule.data_train) / bs))
    Xs = [model.encode_to_params(
        torch.from_numpy(datamodule.data_train[i * bs:(i + 1) * bs]).to(**tkwargs).unsqueeze(1)
    )[0].detach().cpu().numpy() for i in tqdm(range(n_batches))]
    X_enc = np.concatenate(Xs, axis=0)
    X_enc = torch.from_numpy(X_enc).to(**tkwargs)

    if random_search_type == 'sobol':
        assert seed is not None, 'Should specify seed for sobol random search'
        soboleng = torch.quasirandom.SobolEngine(dimension=X_enc.shape[1], scramble=True, seed=seed)
        soboleng.fast_forward(fast_forward)

    new_inputs_ = None
    new_scores_ = []

    while new_inputs_ is None or len(new_inputs_) < num_queries_to_do:

        # To account for outliers
        bounds = torch.zeros(2, X_enc.shape[1]).to(X_enc)
        bounds[0] = torch.quantile(X_enc, .0005, dim=0)
        bounds[1] = torch.quantile(X_enc, .9995, dim=0)
        # make sure best sample is within bounds
        bounds = put_max_in_bounds(X_enc, torch.tensor(datamodule.prop_train).unsqueeze(-1).to(X_enc),
                                   bounds)

        # print(f"Data bound of {bounds} found...")
        delta = .05 * (bounds[1] - bounds[0])
        bounds[0] -= delta
        bounds[1] += delta

        if random_search_type is None:
            bounds = bounds.detach().numpy()
            new_latents = np.random.rand(1, model.latent_dim) * (bounds[1] - bounds[0]) + bounds[0]
        elif random_search_type == 'sobol':
            new_latents = unnormalize(soboleng.draw(1).to(bounds), bounds).cpu().numpy()
        else:
            raise ValueError(f'{random_search_type} not supported for random search')

        res_decode = _decode_z_props(model=model,
                                     z=torch.from_numpy(new_latents).to(**tkwargs),
                                     target=np.load(get_topology_binary_target_path()) if use_binary_data else np.load(
                                         get_topology_target_path()),
                                     filter_unique=filter_unique,
                                     property_key=property_key)
        if filter_unique:
            new_inputs, new_scores, z = res_decode
            checks = [1 if (new_inputs == img).all() else 0 for img in datamodule.data_train]
            seen = sum(checks) > 0
            if new_inputs is None or seen:
                continue
            if new_inputs_ is None:
                new_inputs_, inds = np.unique(new_inputs, axis=0, return_index=True)
                if new_inputs_.ndim == 2:
                    new_inputs_ = new_inputs_.reshape(-1, *new_inputs_.shape)
            else:
                new_inputs_, inds = np.unique(np.vstack([new_inputs_, new_inputs]), axis=0, return_index=True)
            new_scores_ = np.concatenate([new_scores_, new_scores])[inds]
            if len(inds) > 0:
                X_enc = torch.cat([X_enc, torch.from_numpy(new_latents[inds]).reshape(-1, model.latent_dim)], 0)
        else:
            new_inputs, new_scores = res_decode
            checks = [1 if (new_inputs == img).all() else 0 for img in datamodule.data_train]
            seen = sum(checks) > 0
            if new_inputs is None or seen:
                continue
            if new_inputs_ is None:
                new_inputs_ = new_inputs.copy()
            else:
                new_inputs_ = np.vstack([new_inputs_, new_inputs])
            new_scores_ = np.concatenate([new_scores_, new_scores])
            X_enc = torch.cat([X_enc, torch.from_numpy(new_latents).reshape(-1, model.latent_dim).to(**tkwargs)], 0)
        if new_inputs_ is not None:
            assert new_inputs_.ndim == (new_scores_.ndim + 2) == 3, (new_inputs_.shape, new_scores_.shape)

    model.cpu()  # Make sure to free up GPU memory
    torch.cuda.empty_cache()
    return new_inputs_[:num_queries_to_do], new_scores_[:num_queries_to_do]


def _decode_z_props(model: TopologyVAE, z: Tensor, target, filter_unique: bool, property_key):
    z_decode = []
    batch_size = 1000
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            img = model.decode_deterministic(z[j: j + batch_size])
            img = img.cpu().numpy()
            z_decode.append(img)
            del img
    # Concatentate all points and convert to numpy
    z_decode = np.concatenate(z_decode, axis=0)
    z_decode = np.around(z_decode)  # convert to int
    z_decode = z_decode[:, 0, ...]  # Correct slicing
    if filter_unique:
        z_decode, uniq_indices = np.unique(
            z_decode, axis=0, return_index=True
        ) 
        z = z.cpu().numpy()[uniq_indices]

    # Calculate objective function values and choose which points to keep
    if property_key == "scores":
        z_prop = score_function(z_decode, target)
    else:
        raise ValueError(property)

    if filter_unique:
        return z_decode, z_prop, z
    else:
        return z_decode, z_prop


def _batch_decode_z_and_props(model, z, target, property_key: str, filter_unique=True):
    z_decode = []
    batch_size = 1000
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            img = model.decode_deterministic(z[j: j + batch_size])
            img = img.cpu().numpy()
            z_decode.append(img)
            del img

    # Concatentate all points and convert to numpy
    z_decode = np.concatenate(z_decode, axis=0)
    z_decode = np.around(z_decode)  # convert to int
    z_decode = z_decode[:, 0, ...]  # Correct slicing

    if filter_unique:
        z_decode, uniq_indices = np.unique(
            z_decode, axis=0, return_index=True
        )  # Unique elements only
        z = z.cpu().numpy()[uniq_indices]

    # Calculate objective function values and choose which points to keep
    if property_key == "scores":
        z_prop = score_function(z_decode, target)
    else:
        raise ValueError(property_key)

    if filter_unique:
        return z_decode, z_prop, z
    else:
        return z_decode, z_prop


def latent_sampling_wo_props(object, varthreshold, model, num_queries_to_do, bounds, tkwargs: Dict[str, Any], filter_unique=True, use_filter=False, warm_up=False):
    object.to(**tkwargs)
    model.to(**tkwargs)

    if use_filter and warm_up and varthreshold is None:
        warm_up_number = num_queries_to_do // 10 + 1
        warm_up_z = torch.randn(warm_up_number, model.latent_dim, device=model.device)
        warm_up_z_norm = normalize(warm_up_z, bounds=bounds)
        warm_posterior = object.posterior(warm_up_z_norm.to(**tkwargs))
        warm_var = warm_posterior.variance.view(-1).detach().cpu().numpy()   
        varthreshold = warm_var.mean()

    z_sample, ys, var = [], [], []
    mean_var_list = []
    redo = 0

    while len(z_sample) < num_queries_to_do:
        new_z = torch.randn(num_queries_to_do, model.latent_dim, device=model.device)
        new_z_norm = normalize(new_z, bounds=bounds)
        new_posterior = object.posterior(new_z_norm.to(**tkwargs))
        new_ys = new_posterior.mean.view(-1).detach().cpu().numpy()
        new_var = new_posterior.variance.view(-1).detach().cpu().numpy() 
        
        mean_var_list.append(new_var.mean())
        
        if use_filter and varthreshold is not None:
            print("varthreshold: ", varthreshold)
            valid_indx = new_var <= varthreshold
            new_z = new_z[valid_indx]
            new_ys = new_ys[valid_indx]
            new_var = new_var[valid_indx]
        
        if len(new_z) != 0:
            if len(z_sample) == 0:
                z_sample = new_z
                ys = new_ys
                var = new_var
            else:
                z_sample = torch.vstack([z_sample, new_z])
                ys = np.hstack([ys, new_ys])
                var = np.hstack([var, new_var])
        else:
            redo += 1
            print('0: ', redo)
            if redo <= 10:
                var_mean = np.array(mean_var_list).mean()
                if varthreshold is None:
                    varthreshold = var_mean
                else:
                    varthreshold = 0.9 * varthreshold + 0.1 * var_mean
                continue
            else:
                break 

    if filter_unique:
        z_decode, _, uniq_indices = _decode_z(model, z_sample, filter_unique=filter_unique)
        z_sample = z_sample[uniq_indices]
        ys = new_ys[uniq_indices]
        var = new_var[uniq_indices]
    else:
        z_decode = _decode_z(model, z_sample, filter_unique=filter_unique)

    object.cpu()
    model.cpu()
    torch.cuda.empty_cache()

    return varthreshold, z_decode, z_sample, ys, var


def latent_random_search_wo_props(object, varthreshold, model: TopologyVAE, num_queries_to_do: int, 
                         tkwargs: Dict[str, Any], datamodule: WeightedNumpyDataset, filter_unique=False,
                         random_search_type=None, use_filter=True, warm_up=False) -> Tuple[np.ndarray, np.ndarray]:

    print_flush(f"\t\tPicking Pseudo data via {random_search_type if random_search_type is not None else ''} RS...")

    model.eval()
    model.to(**tkwargs)
    object.eval()
    object.to(**tkwargs)

    bs = 5000
    n_batches = int(np.ceil(len(datamodule.data_train) / bs))
    Xs = [model.encode_to_params(
        torch.from_numpy(datamodule.data_train[i * bs:(i + 1) * bs]).to(**tkwargs).unsqueeze(1)
    )[0].detach().cpu().numpy() for i in tqdm(range(n_batches))]
    X_enc = np.concatenate(Xs, axis=0)
    X_enc = torch.from_numpy(X_enc).to(**tkwargs)

    if random_search_type == 'sobol':
        # assert seed is not None, 'Should specify seed for sobol random search'
        soboleng = torch.quasirandom.SobolEngine(dimension=X_enc.shape[1], scramble=True)

    if warm_up and use_filter and varthreshold is None:
        warm_up_number = num_queries_to_do // 10 + 1
        if random_search_type is None:
            warm_latents = torch.from_numpy(np.random.rand(warm_up_number, model.latent_dim)).to(bounds)
        elif random_search_type == 'sobol':
            warm_latents = soboleng.draw(warm_up_number).to(bounds)
        else:
            raise ValueError(f'{random_search_type} not supported for random search')

        warm_posterior = object.posterior(warm_latents)
        warm_var = warm_posterior.variance.view(-1).detach().cpu().numpy() 
        varthreshold = warm_var.mean()

    new_inputs_ = None
    new_zs_ = None
    new_ys_ = None
    new_var_ = None
    redo = 0

    mean_var = []

    while new_inputs_ is None or len(new_inputs_) < num_queries_to_do:
        # To account for outliers
        bounds = torch.zeros(2, X_enc.shape[1]).to(X_enc)
        bounds[0] = torch.quantile(X_enc, .0005, dim=0)
        bounds[1] = torch.quantile(X_enc, .9995, dim=0)
        # make sure best sample is within bounds
        bounds = put_max_in_bounds(X_enc, torch.tensor(datamodule.prop_train).unsqueeze(-1).to(X_enc), bounds)
        delta = .05 * (bounds[1] - bounds[0])
        bounds[0] -= delta
        bounds[1] += delta

        if random_search_type is None:
            new_latents_norm = torch.from_numpy(np.random.rand(100, model.latent_dim)).to(bounds)
        elif random_search_type == 'sobol':
            new_latents_norm = soboleng.draw(100).to(bounds)
        else:
            raise ValueError(f'{random_search_type} not supported for random search')

        new_posterior = object.posterior(new_latents_norm)
        new_ys = new_posterior.mean.view(-1).detach().cpu().numpy()
        new_var = new_posterior.variance.view(-1).detach().cpu().numpy() 
        
        mean_var.append(new_var.mean())
        
        if use_filter and varthreshold is not None:
            valid_indx = new_var <= varthreshold
            new_latents_norm = new_latents_norm[valid_indx]
            new_ys = new_ys[valid_indx]
            new_var = new_var[valid_indx]

            if len(new_latents_norm) == 0:
                redo += 1
                print('0: ', redo)
                if redo < 10:
                    continue
                else:
                    break
        
        new_latents = unnormalize(new_latents_norm, bounds)
        res_decode = _decode_z(model=model, z=new_latents.to(**tkwargs), filter_unique=filter_unique)
        
        if filter_unique:
            new_inputs, _, new_inds  = res_decode
            if new_inputs is None:
                continue
            if new_inputs_ is None:
                new_inputs_ = new_inputs.copy()
                if new_inputs_.ndim == 2:
                    new_inputs_ = new_inputs_.reshape(-1, *new_inputs_.shape)
            else:
                new_inputs_, inds = np.unique(np.vstack([new_inputs_, new_inputs]), axis=0, return_index=True)
            if new_zs_ is None:
                new_zs_ = new_latents_norm[new_inds].cpu().numpy().copy()
                new_ys_ = new_ys.copy()
                new_var_ = new_var.copy()
            else:
                new_zs_ = np.vstack([new_zs_, new_latents_norm[new_inds].cpu().numpy()])[inds]
                new_ys_ = np.hstack([new_ys_, new_ys[new_inds]])[inds]
                new_var_ = np.hstack([new_var_, new_var[new_inds]])[inds]
            X_enc = torch.cat([X_enc, new_latents[new_inds].reshape(-1, model.latent_dim).to(**tkwargs)], 0)
        else:
            new_inputs = res_decode
            if new_inputs is None:
                continue
            if new_inputs_ is None:
                new_inputs_ = new_inputs.copy()
                new_zs_ = new_latents_norm.cpu().numpy().copy()
                new_ys_ = new_ys.copy()
                new_var_ = new_var.copy()
            else:
                new_inputs_ = np.vstack([new_inputs_, new_inputs])
                new_zs_ = np.vstack([new_zs_, new_latents_norm.cpu().numpy()])
                new_ys_ = np.hstack([new_ys_, new_ys])
                new_var_ = np.hstack([new_var_, new_var])
            X_enc = torch.cat([X_enc, new_latents.reshape(-1, model.latent_dim).to(**tkwargs)], 0)
        if new_inputs_ is not None:
            assert new_inputs_.ndim == (new_zs_.ndim+1) == 3, (new_inputs_.shape, new_zs_.shape)

    mean_var = np.array(mean_var)
    if varthreshold is None:
        varthreshold = mean_var.mean()
    else:
        varthreshold = 0.9 * varthreshold + 0.1 * mean_var.mean()

    model.cpu()
    torch.cuda.empty_cache()
    if new_inputs_ is None:
        return None, None, None, None
    return new_inputs_[:num_queries_to_do], new_zs_[:num_queries_to_do], new_ys_[:num_queries_to_do], new_var_[:num_queries_to_do]


def latent_sampling_GA(object, init_points, num_queries_to_do, model, bounds, varthreshold, tkwargs, sigma=None, use_filter=True):
    object.eval()
    object.to(**tkwargs)
    model.eval()
    model.to(**tkwargs)

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
        decode_start = time.time()   
        sampling_x, _, unique_indx = _decode_z(model, unnormalize(torch.tensor(sampling_z).to(bounds), bounds).detach(), 
                               filter_unique=True)
        sampling_z = sampling_z[unique_indx]
        sampling_y = sampling_y[unique_indx]
        sampling_var = sampling_var[unique_indx]
        print("\n Decode spend time: ", time.time()-decode_start)
        best_index = np.argsort(sampling_y, axis=0)[::-1]
        sampling_x = sampling_x[best_index]
        sampling_z = sampling_z[best_index]
        sampling_y = sampling_y[best_index]
        sampling_var = sampling_var[best_index]

    object.cpu()
    model.cpu()
    torch.cuda.empty_cache()
    return varthreshold, sampling_z[:num_queries_to_do], sampling_x[:num_queries_to_do], sampling_y[:num_queries_to_do], sampling_var[:num_queries_to_do]


def latent_sampling_Noise(object, sigma, init_points, num_queries_to_do, model, bounds, varthreshold, tkwargs, use_filter=True):
    sampling_z, sampling_x, sampling_y, sampling_var = [], [], [], []
    iter_sampling_number = num_queries_to_do // len(init_points) + 1
    warm_up_number = iter_sampling_number // 10 + 1

    model.eval()
    object.eval()
    
    if use_filter and varthreshold is None:
        warm_up_z = []
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

    print("\n sampling var threshold: ", varthreshold)
    mean_var = []
    for point in init_points:
        zs, ys, var = [], [], []
        redo = 0
        while len(zs) < iter_sampling_number:
            new_z = np.random.normal(loc=point, scale=sigma, size=(iter_sampling_number, point.shape[-1]))
            new_posterior = object.posterior(torch.from_numpy(new_z).to(**tkwargs))
            new_y = new_posterior.mean.view(-1).detach().cpu().numpy()
            new_var = new_posterior.variance.view(-1).detach().cpu().numpy()
            
            mean_var.append(new_var.mean())

            if use_filter and varthreshold is not None:
                valid_indx = new_var <= varthreshold
                new_z = new_z[valid_indx]
                new_y = new_y[valid_indx]
                new_var = new_var[valid_indx]
                
                if len(new_z) == 0:
                    redo += 1
                    print('0: ', redo)
                    if redo < 10:
                        continue
                    else:
                        print("\n point: \n", point)
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
    if len(sampling_z) != 0:
        sampling_z = np.vstack(sampling_z)
        sampling_y = np.hstack(sampling_y)
        sampling_var = np.hstack(sampling_var)

        sampling_x, _, indx = _decode_z(model, unnormalize(torch.tensor(sampling_z).unsqueeze(1).to(bounds), bounds).detach(), filter_unique=True)
        sampling_z = sampling_z[indx]
        sampling_y = sampling_y[indx]
        sampling_var = sampling_var[indx]

        best_index = np.argsort(sampling_y, axis=0)[::-1]
        sampling_z = sampling_z[best_index]
        sampling_x = sampling_x[best_index]
        sampling_y = sampling_y[best_index]
        sampling_var = sampling_var[best_index]
    
    mean_var = np.array(mean_var)
    if varthreshold is None:
        varthreshold = mean_var.mean()
    else:
        varthreshold = 0.9 * varthreshold + 0.1 * mean_var.mean()

    model.cpu()
    torch.cuda.empty_cache()
    return varthreshold, sampling_z[:num_queries_to_do], sampling_x[:num_queries_to_do], sampling_y[:num_queries_to_do], sampling_var[:num_queries_to_do]


def latent_sampling_CMA_ES(object, init_points, num_queries_to_do, model, bounds, varthreshold, tkwargs, sigma=None, use_filter=True):
    object.eval()
    object.to(**tkwargs)
    model.eval()
    model.to(**tkwargs)

    sampling_z, sampling_x, sampling_y, sampling_var = [], [], [], []

    cmaes_start = time.time()
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
    print("\n cmaes spend time: ", time.time()-cmaes_start)

    if sampling_z is not None and len(sampling_z) != 0:
        sampling_x, _, unique_indx = _decode_z(model, unnormalize(torch.tensor(sampling_z).to(bounds), bounds).detach(), filter_unique=True)
        sampling_z = sampling_z[unique_indx]
        sampling_x = sampling_x[unique_indx]
        sampling_y = sampling_y[unique_indx]
        sampling_var = sampling_var[unique_indx]

        best_index = np.argsort(sampling_y, axis=0)[::-1]
        sampling_z = sampling_z[best_index]
        sampling_x = sampling_x[best_index]
        sampling_y = sampling_y[best_index]
        sampling_var = sampling_var[best_index]

    var_mean = np.array(mean_var_list).mean()
    if varthreshold is None:
        varthreshold = var_mean
    else:
        varthreshold = 0.9 * varthreshold + 0.1 * var_mean

    object.cpu()
    model.cpu()
    torch.cuda.empty_cache()
    return varthreshold, sampling_z[:num_queries_to_do], sampling_x[:num_queries_to_do], sampling_y[:num_queries_to_do], sampling_var[:num_queries_to_do]


def _decode_z(model: TopologyVAE, z: Tensor, filter_unique: bool=False):
    """
    helper function to decode some latent vectors and calculate their properties
    """
    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1000
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            img = model.decode_deterministic(z[j: j + batch_size])
            img = img.cpu().numpy()
            z_decode.append(img)
            del img
    # Concatentate all points and convert to numpy
    z_decode = np.concatenate(z_decode, axis=0)
    z_decode = np.around(z_decode)  # convert to int
    z_decode = z_decode[:, 0, ...]  # Correct slicing
    if filter_unique:
        z_decode, uniq_indices = np.unique(
            z_decode, axis=0, return_index=True
        )  # Unique elements only
        z = z.cpu().numpy()[uniq_indices]

    if filter_unique:
        return z_decode, z, uniq_indices
    else:
        return z_decode


def _batch_decode_z(model, z, filter_unique=True):
    """
    helper function to decode some latent vectors and calculate their properties
    """
    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1000
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            img = model.decode_deterministic(z[j: j + batch_size])
            img = img.cpu().numpy()
            z_decode.append(img)
            del img

    # Concatentate all points and convert to numpy
    z_decode = np.concatenate(z_decode, axis=0)
    z_decode = np.around(z_decode)  # convert to int
    z_decode = z_decode[:, 0, ...]  # Correct slicing
    if filter_unique:
        z_decode, uniq_indices = np.unique(
            z_decode, axis=0, return_index=True
        )  # Unique elements only
        z = z.cpu().numpy()[uniq_indices]

    if filter_unique:
        return z_decode, z, uniq_indices
    else:
        return z_decode


def increment_beta_pseudo_loss(beta_pseudo_loss, beta_pseudo_loss_final, ret_idx, start_num_retrain, beta_step_freq, beta_step):
    # Check if the warmup is over and if it's the right step to increment beta 
    if (
        ret_idx > start_num_retrain
        and (ret_idx-start_num_retrain) % beta_step_freq == 0
    ):
        beta = min(beta_pseudo_loss_final, beta_pseudo_loss + beta_step)
    else:
        beta = beta_pseudo_loss
    return beta


if __name__ == "__main__":
    main()
    