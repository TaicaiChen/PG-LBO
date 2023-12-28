# PG-LBO
This is the official repository for our AAAI 2024 paper:
> PG-LBO: Enhancing High-Dimensional Bayesian Optimization with Pseudo-Label and Gaussian Process Guidance
## Environment installation
- Install the experiments running environment.
```bash
conda env create -f environment.yml
conda activate pglbo_env
```
- Set result storage path.
```bash
echo '~/LSO-storage/' > ./utils/storage_root_path.txt
```
## Usage guide
### Topology task
#### Set up data
- The data can be downloaded from the following git repo: [Neural Networks for Topology Optimization](https://github.com/ISosnovik/top) then place the unzipped files into your DATA_STORAGE_ROOT/data/topology_data/ (see storage_root_path.txt to check your DATA_STORAGE_ROOT).
#### Run the experiment.
- Train VAE.
```bash
chmod u+x ./weighted_retraining/scripts/models/train-topology-pseudo.sh
./weighted_retraining/scripts/models/train-topology-pseudo.sh
```
- Execute the optimization process.
    - Set parameters for pseudo-label training and GP-guided techniques.
    ```bash
    ....

    #-- Choose whether to use pseudo-label training --#
    use_pseudo='--use_pseudo' #'' represents not use.
    pseudo_data_size=5000
    pseudo_beta=0.5
    beta_pseudo_loss=0.5
    beta_pseudo_loss_final=0.75

    #-- Choose the pseudo data sampling you want to use --#
    pseudo_sampling_type_ind=4
    pseudo_sampling_type=("" "random" "cmaes" "ga" "noise")
    pseudo_sampling_type_kws=("" "{}" "{'n_best':100,'n_rand':0,'sigma':0.25,'use_bo':False}" "{'n_best':100,'n_rand':0,'use_bo':False}" "{'n_best':100,'n_rand':0,'sigma':0.1,'use_bo':False}")
    if ((pseudo_sampling_type_ind == 0)); then pseudo_sampling=''; else pseudo_sampling="--pseudo_sampling_type ${pseudo_sampling_type[$pseudo_sampling_type_ind]}"; fi
    if ((pseudo_sampling_type_ind == 0)); then pseudo_sampling_kw=''; else pseudo_sampling_kw="--pseudo_sampling_type_kw ${pseudo_sampling_type_kws[$pseudo_sampling_type_ind]}"; fi

    #-- Choose whether to use GP guidance --#
    use_ssdkl='--use_ssdkl' # '' represents not use.
    beta_ssdkl_loss=1 # weight of GP guidance loss.

    ....
    ```
    - Execute the script.
    ```bash
    chmod u+x ./weighted_retraining/scripts/robust_opt/robust_opt_topology_pseudo.sh
    ./weighted_retraining/scripts/robust_opt/robust_opt_topology_pseudo.sh
    ```
### Expression task
#### Set up data
- Set up the data with `expr_dataset.py` after downloading and unzipping expression data from [weighted-retraining repo](https://github.com/cambridge-mlg/weighted-retraining/tree/master/assets/data/expr)
```bash
# download necessary files and unzip data
url=https://github.com/cambridge-mlg/weighted-retraining/raw/master/assets/data/expr/
expr_dir="./weighted_retraining/assets/data/expr"
mkdir $expr_dir
for file in eq2_grammar_dataset.zip equation2_15_dataset.txt scores_all.npz;
do
 cmd="wget -P ${expr_dir} ${url}${file}"
 echo $cmd 
 $cmd
done
unzip "$expr_dir/eq2_grammar_dataset.zip" -d "$expr_dir"

# split data and generate datasets used in our BO experiments
python ./weighted_retraining/weighted_retraining/expr/expr_dataset.py \
           --ignore_percentile 65 --good_percentile 5 \
           --seed 0 --save_dir weighted_retraining/data/expr
```
#### Run the experiment.
- Train VAE.
```bash
chmod u+x ./weighted_retraining/scripts/models/train-expr-pseudo.sh
./weighted_retraining/scripts/models/train-expr-pseudo.sh
```
- Execute the optimization process.
    - Set parameters for pseudo-label training and GP-guided techniques.
    ```bash
    ....

    #-- Choose whether to use pseudo-label training --#
    use_pseudo='--use_pseudo' #'' represents not use.
    pseudo_data_size=20000
    pseudo_beta=0.5
    beta_pseudo_loss=0.1
    beta_pseudo_loss_final=0.75

    #-- Choose the pseudo data sampling you want to use --#
    pseudo_sampling_type_ind=4
    pseudo_sampling_type=("" "random" "cmaes" "ga" "noise")
    pseudo_sampling_type_kws=("" "{}" "{'n_best':100,'n_rand':0,'sigma':0.25,'use_bo':False}" "{'n_best':100,'n_rand':0,'use_bo':False}" "{'n_best':100,'n_rand':0,'sigma':0.1,'use_bo':False}")
    if ((pseudo_sampling_type_ind == 0)); then pseudo_sampling=''; else pseudo_sampling="--pseudo_sampling_type ${pseudo_sampling_type[$pseudo_sampling_type_ind]}"; fi
    if ((pseudo_sampling_type_ind == 0)); then pseudo_sampling_kw=''; else pseudo_sampling_kw="--pseudo_sampling_type_kw ${pseudo_sampling_type_kws[$pseudo_sampling_type_ind]}"; fi

    #-- Choose whether to use GP guidance --#
    use_ssdkl='--use_ssdkl' # '' represents not use.
    beta_ssdkl_loss=0.1 # weight of GP guidance loss.

    ....
    ```
    - Execute the script.
    ```bash
    chmod u+x ./weighted_retraining/scripts/robust_opt/robust_opt_expr_pseudo.sh
    ./weighted_retraining/scripts/robust_opt/robust_opt_expr_pseudo.sh
    ```
### Molecule task
#### Set up data.
- To download and build the Zinc250k dataset with black-box functions labels, execute setup-chem.sh
```bash
# download necessary files
url=https://github.com/cambridge-mlg/weighted-retraining/raw/master/assets/data/chem_orig_model
molecule_dir="./weighted_retraining/assets/data/chem_orig_model/"
mkdir $molecule_dir
for file in train.txt val.txt vocab.txt README.md;
do
 wget -P $molecule_dir "$url/$file"
done

# preprocess molecule data for BO experiments
chmod u+x ./weighted_retraining/scripts/data/setup-chem.sh
./weighted_retraining/scripts/data/setup-chem.sh
```
#### Run the experiment.
- get pre-train model
```bash
url=https://github.com/cambridge-mlg/weighted-retraining/raw/master/assets/pretrained_models/chem.ckpt
wget -P ./weighted_retraining/assets/pretrained_models/chem_vanilla/ $url
```
- Execute the optimization process.
    - Set parameters for pseudo-label training and GP-guided techniques.
    ```bash
    ....

    #-- Choose whether to use pseudo-label training --#
    use_pseudo='--use_pseudo' #'' represents not use.
    pseudo_data_size=100000
    pseudo_beta=0.5
    beta_pseudo_loss=0.1
    beta_pseudo_loss_final=0.75
    pseudo_train_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_train_pseudo"
    pseudo_val_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_val_pseudo"

    #-- Choose the pseudo data sampling you want to use --#
    pseudo_sampling_type_ind=4
    pseudo_sampling_type=("" "random" "cmaes" "ga" "noise")
    pseudo_sampling_type_kws=("" "{}" "{'n_best':100,'n_rand':0,'sigma':0.25,'use_bo':False}" "{'n_best':100,'n_rand':0,'use_bo':False}" "{'n_best':100,'n_rand':0,'sigma':0.1,'use_bo':False}")
    if ((pseudo_sampling_type_ind == 0)); then pseudo_sampling=''; else pseudo_sampling="--pseudo_sampling_type ${pseudo_sampling_type[$pseudo_sampling_type_ind]}"; fi
    if ((pseudo_sampling_type_ind == 0)); then pseudo_sampling_kw=''; else pseudo_sampling_kw="--pseudo_sampling_type_kw ${pseudo_sampling_type_kws[$pseudo_sampling_type_ind]}"; fi

    #-- Choose whether to use GP guidance --#
    use_ssdkl='--use_ssdkl' # '' represents not use.
    beta_ssdkl_loss=0.1 # weight of GP guidance loss.

    ....
    ```
    - Execute the script.
    ```bash
    chmod u+x ./weighted_retraining/scripts/robust_opt/robust_opt_chem_pseudo.sh
    ./weighted_retraining/scripts/robust_opt/robust_opt_chem_pseudo.sh
    ```
## Acknowledgements
- Thanks to the authors of the [High-Dimensional Bayesian Optimisation withVariational Autoencoders and Deep Metric Learning](https://github.com/huawei-noah/HEBO/tree/master/T-LBO) for providing their implementations of high-dimensional BO based on weighted-retraining VAE, which we based most of our code on.
