#!/bin/bash

seed=0

#-- Dataset composition --#
ignore_percentile=65
good_percentile=5
data_seed=0

#-- Choose dimension of the latent space --#
latent_dim=25

#-- Choose whether to use target prediction --#
predict_target=0
beta_target_pred_loss=1
target_predictor_hdims='[128,128]'
if ((predict_target == 0)); then predict_target=''; else predict_target='--predict_target'; fi

batch_size=1024
max_epochs=300

#-- Choose on which GPU to run --#
cuda=0

# Train expr VAE
k="inf"

cmd="python weighted_retraining/weighted_retraining/partial_train_scripts/partial_train_expr_pseudo.py \
  --seed=$seed  \
  --cuda=$cuda --batch_size $batch_size \
  --latent_dim=$latent_dim \
  --dataset_path=weighted_retraining/data/expr \
  --property_key=scores \
  --max_epochs=$max_epochs \
  --beta_final=.04 --beta_start=1e-6 \
  --beta_warmup=500 --beta_step=1.1 --beta_step_freq=10 \
  --weight_type rank --rank_weight_k $k --data_seed $data_seed \
  --ignore_percentile $ignore_percentile --good_percentile $good_percentile \
  $predict_target --target_predictor_hdims $target_predictor_hdims \
  --beta_target_pred_loss=$beta_target_pred_loss
"
echo $cmd
$cmd
echo $cmd
