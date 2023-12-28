#!/bin/bash

# Meta flags
seed=0

#-- Choose whether to use target prediction --#
# 是否使用 
predict_target=0
beta_target_pred_loss=10
target_predictor_hdims='[128,128]'
if ((predict_target == 0)); then predict_target=''; else predict_target='--predict_target'; fi

#-- Choose whether to use pseudo data --#
# 是否使用 
use_pseudo=0
beta_pseudo_loss=1
if ((use_pseudo == 0)); then use_pseudo=''; else use_pseudo='--use_pseudo'; fi

use_binary_data=1
if (( use_binary_data == 1 )); then use_binary_data='--use_binary_data'; else use_binary_data=''; fi

#-- Choose dimension of the latent space --#
latent_dim=20

#-- For how many epochs do you want to train the model? --#
max_epochs=300

#-- Choose on which GPU to run --#
cuda=4

k="1e-3"

# Train topo VAE
python weighted_retraining/weighted_retraining/partial_train_scripts/partial_train_topology_pseudo.py \
--seed=$seed \
--latent_dim=$latent_dim \
--property_key=scores \
--max_epochs=$max_epochs \
--beta_final=1e-4 \
--beta_start=1e-6 \
--beta_warmup=1000 \
--beta_step=1.1 \
--beta_step_freq=10 \
--batch_size=1024 \
--cuda=$cuda \
--weight_type uniform \
--rank_weight_k $k \
$predict_target --target_predictor_hdims $target_predictor_hdims --beta_target_pred_loss $beta_target_pred_loss \
$use_binary_data
