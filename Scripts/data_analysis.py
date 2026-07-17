import numpy as np
import os
import random
from sklearn.model_selection import KFold
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
from pytorch_metric_learning import miners, losses, distances
from DataLoaders import SingleDatasetLoaderLabels, SingleDatasetLoaderNoLabels, DualDatasetLoaderLabels,\
    DualDatasetLoaderNoLabels, init_data_loaders_no_labs_va_te
from Losses import EvalHard
from Utils import set_seed, save_list, load_list, load_base_metadata
from Models import TwoF, ThrCOneTTwoF, SiameseNet
from TrainEval import train_val, test

"""This script implements Machine Learning (ML) models for gait recognition using ForceID-B and four previous large-
scale walking GRF datasets: AIST, Gutenberg, GaitRec and ForceID-A (referred herein as benchmark datasets).
Specifically, ML models are developed (i.e., trained and validated) on data aggregated from the the benchmark datasets
and then evaluated (i.e., tested) on ForceID-B_RS to infer the extent to which defining features of gait are common
between reference data acquired in standard gait analysis contexts and ForceID-B. This script is based on main.py in the
ForceID-Study-2 repo for Duncanson et al. (2024). Please refer to that script if additional information is required on
minor details such as string nomenclature."""

# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. INITIALISE EXPERIMENTAL PARAMETERS AND MODEL HYPER-PARAMETERS - START
# # -------------------------------------------------------------------------------------------------------------------
# Set random seed for reproducibility:
seed = 42
g = set_seed(seed)

# Specify the dataset configuration method (we use five datasets in a cross-dataset protocol):
config_method = 'fiv-cro--'

# Specify the dataset configuration. The last dataset is used for evaluation and the others are used for development:
config = ['fi-all',
          'gr-all',
          'gb-all',
          'ai-all',
          'fi-b-rs']

results_path = './Results'

# For k-fold cross-validation, specify the number of folds:
n_folds = 5

# Specify the neural network architecture from the current set of options: '-----2F' and '3C1T-2F' (see Models module
# for details):
# Summary of architecture string nomenclature (see ForceID-Study-2 -- main.py for details)
# ----------------------------------------------------------------------------------------
# Seven digit code specifying the number of layers of each type.
# C = Convolutional layer
# T = Transformer encoder layer
# F = Fully-connected layer
# ----------------------------------------------------------------------------------------
arch = '-----2F'

# Specify the fixed length of time normalised sequences based on the linear interpolation conducted in
# data_processing.py. The value here must equal the value of the len_interp variable in data_processing.py:
len_interp = 100

# Specify the number of channels in data objects, with each channel representing a force platform directional component.
# The channels and no. channels must be the same across ForceID-B (set-up in data_processing.py) and all benchmark
# datasets. Currently, there are five channels - Fx, Fy, Fz, Cx and Cy:
n_channels = 5

# The single_network variable below was True in the experiments conducted to technically validate ForceID-B. If set
# to False, an experiment is conducted using a Siamese neural network that includes two copies of the specified
# architecture with shared weights. The model takes two inputs and then conducts output feature level fusion using a
# basic operation like addition or concatenation. There are a few operations to choose from by commenting/uncommenting
# within the SiameseNet class in the Models module. This implementation is set up to include data from right and left
# stance sides as the separate inputs as per Duncanson et al. (2021), but could easily be modified to use inputs of
# different nature (e.g., FP1 and FP2) or extended to work with three+ independent inputs (e.g., Fx, Fy and Fz):
single_network = True
if single_network: # TODO: Debug for single_network = False.

    # In this case, right and left stance sides will be concatenated lengthwise, so the sequence length (len_seq) will
    # be twice the interpolation length in terms of number of frames:
    len_seq = len_interp * 2

    if arch == '-----2F':

        # Specify the number of nodes (and associated features) at each layer. The channel and sequence dimensions are
        # flattened, so the number of input features is channels C x sequence length L (currently 5 x 200 = 1000):
        in_features = int(n_channels * len_seq)
        fc1_out = 800
        out_features = 600

        params_mod = [in_features, fc1_out, out_features]

    elif arch == '3C1T-2F':

        # For the first three convolutional layers, a channel represents a convolution filter. ('nc' is shorthand for
        # number of channels and layer 0 is analogous to input):
        nc0 = n_channels
        nc1 = 32
        nc2 = 64
        nc3 = 128

        # Convolution filter hyper-parameters:
        conv_k = 3 # k = kernel size.
        conv_p = 1 # p = padding.

        # Average pooling hyper-parameters:
        pool_k = 2
        pool_s = 2 # s = stride.
        pool_p = 0

        # Transformer encoder hyper-parameters:
        trans_nhead = 4 # Number of heads.
        trans_act = 'gelu' # Activation function.

        # With the above convolution and pooling hyper-parameters, L gets halved after each convolutional layer
        # according to the L equation in the PyTorch Conv1d and AvgPool1d API docs. Changes in the settings above may
        # require changes in the L value below according to the docs. In the current implementation,
        # C = conv_channels_l3 = 128 and L = 25 at the transformer encoder layer. Hence, its internal feed-forward (ff)
        # i.e., fully-connected layer inputs a flattened feature vector of dimensionality/size C x L = 128 x 25 = 3200:
        trans_dim_ff = nc3 * 25

        # For the fully-connected layers after the transformer encoder layer, the channel and sequence dimensions are
        # also flattened for the input (number of input features = C x L = 128 x 25 = 3200). Specify the number of
        # output features for each layer:
        fc1_out = 1600
        out_features = 800

        params_mod = [nc0, nc1, nc2, nc3, conv_k, conv_p,
                      pool_k, pool_s, pool_p, len_seq,
                      trans_nhead, trans_act, trans_dim_ff,
                      fc1_out, out_features]

    else:

        raise Exception('The specified neural network architecture is currently not supported.')

else:

    # With single_network set to False, right and left stance sides will be used as separate inputs, so the sequence
    # length remains unchanged relative to the interpolation length in terms of number of frames:
    len_seq = len_interp

    if arch == '-----2F':

        in_features = int(n_channels * len_seq) # Currently 500.
        fc1_out = 400
        out_features = 300

        params_mod = [in_features, fc1_out, out_features]

    elif arch == '3C1T-2F':

        nc0 = n_channels
        nc1 = 32
        nc2 = 64
        nc3 = 128

        # Convolution filter hyper-parameters:
        conv_k = 3
        conv_p = 1

        # Average pooling hyper-parameters:
        pool_k = 2
        pool_s = 2
        pool_p = 0

        # Transformer encoder hyper-parameters:
        trans_nhead = 4
        trans_act = 'gelu'

        # In the current implementation, C = conv_channels_l3 = 128 and L = 12. Thus, the dimensionality/size of the
        # internal feed-forward layer in the transformer encoder layer is:
        trans_dim_ff = nc3 * 12

        # For the fully-connected layers after the transformer encoder layer, the channel and sequence dimensions are
        # also flattened for the input (number of input features = C x L = 128 x 12 = 1536). Specify the number of
        # output features for each layer:
        fc1_out = 800
        out_features = 400

        params_mod = [nc0, nc1, nc2, nc3, conv_k, conv_p,
                      pool_k, pool_s, pool_p, len_seq,
                      trans_nhead, trans_act, trans_dim_ff,
                      fc1_out, out_features]

    else:

        raise Exception('The specified neural network architecture is currently not supported.')

bs = 512 # Batch size.
epochs = 1000
es_patience = 20 # No. epochs without improvement in loss before early stopping ('es') training and validation.
es_min_delta = 0 # Minimum change in loss that's considered an improvement for the purpose of early stopping.

# Specify additional parameters for the optimiser other than the model parameters to pass as arguments. We use the Adam
# optimiser with the following settings:
lr = 0.0001 # Learning rate.
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0
amsgrad = True # AMSGRAD variant of the optimiser.
params_opt_extra = [lr, betas, eps, weight_decay, amsgrad]

# Specify whether to use the pytorch_metric_learning (ptm) package to calculate the loss. The script is currently set up
# such that it has to be True as per the conditional below. We've retained the ptm variable as a bool to indicate that
# the script can be made to work without the ptm package, i.e., with a custom loss function. A custom loss function
# class would have to be initialized in place of the Exception below:
ptm = True
if ptm:

    # Margin for the triplet margin loss:
    m = 0.3
    # Initialise the loss:
    criterion_tr = losses.TripletMarginLoss(margin=m, swap=False, smooth_loss=True,
                                            distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1))
    # Initialise the miner for sampling hard triplets within each batch during training:
    miner = miners.BatchHardMiner(distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1))
    # If initialising a loss function that requires an optimiser, specify the optimiser here:
    criterion_opt = None

else:

    raise Exception("Script is currently set up to use the pytorch_metric_learning package. If you wish to use a custom"
                    "loss function, please replace this Exception with its class initialisation.")

# We use a custom method of similarity search to calculate loss and accuracy on validation and test sets ('random subset
# search' (Duncanson et al., 2021; Duncanson et al., 2024):
criterion_eval = EvalHard(margin=m)

# Specify whether to use cuda for training, validation and testing. We include this for flexible memory management:
cuda_tr = False
cuda_va = False
cuda_te = False
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. INITIALISE EXPERIMENTAL PARAMETERS AND MODEL HYPER-PARAMETERS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. GET DATA AND BASIC METADATA FROM EACH DATASET - START
# # -------------------------------------------------------------------------------------------------------------------
# Variables representing all datasets vs a single dataset are distinguished by abbreviation (e.g., trial_names vs tns
# and counts_samples vs cs, respectively).
trial_names_by_id_shuff = [] # 'shuff' = shuffled.
trial_names = []
counts_samples = []
sigs_r = [] # r = right stance.
sigs_l = [] # l = left stance.

for idx, dataset in enumerate(config):

    # Load the dataset's trial name identifiers (tns) and the counts of samples per ID (cs):
    tns, _, __, cs = load_base_metadata(dataset)

    # If not the last dataset in the configuration (i.e., if a benchmark dataset to be used for model development):
    if idx != len(config) - 1:

        path_file_trial_names = os.path.join(results_path, 'trial_names_by_id_shuff_%s.txt' % dataset)

        # If the code has not been run and thus the trial_names are yet to be shuffled:
        if not os.path.isfile(path_file_trial_names):

            indices_where_id_changes = np.cumsum(cs)[:-1]
            tns_by_id = np.split(tns, indices_where_id_changes)

            tns_by_id_shuff = []

            for tns_subset in tns_by_id:

                # At least two samples per ID are required for distance metric learning:
                if tns_subset.shape[0] > 1:

                    tns_by_id_shuff.append(random.sample(list(tns_subset), k=tns_subset.shape[0]))

            save_list(os.path.join(results_path, 'trial_names_by_id_shuff_%s.txt' % dataset), tns_by_id_shuff)

        # If the code has been run, the shuffled trial names have already been defined and saved. Thus, simply load here
        # to avoid re-defining and saving each time:
        else:

            tns_by_id_shuff = load_list(os.path.join(results_path, 'trial_names_by_id_shuff_%s.txt' % dataset))

        trial_names_by_id_shuff.append(tns_by_id_shuff)

    trial_names.append(tns)
    counts_samples.append(cs)
    sigs_r.append(np.load('./Datasets/%s/Objects/sigs_pro_r.npy' % dataset, allow_pickle=True))
    sigs_l.append(np.load('./Datasets/%s/Objects/sigs_pro_l.npy' % dataset, allow_pickle=True))

# Concatenate data objects from each dataset along the sample axis. Overwrite the variable to save memory:
sigs_r = np.concatenate((sigs_r[0], sigs_r[1], sigs_r[2], sigs_r[3], sigs_r[4]), axis=0)
sigs_l = np.concatenate((sigs_l[0], sigs_l[1], sigs_l[2], sigs_l[3], sigs_l[4]), axis=0)

# Concatenate trial_names object:
trial_names_cat = np.concatenate(trial_names)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. GET DATA AND BASIC METADATA FROM EACH DATASET - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. DEFINE/LOAD TRIAL_NAMES FOR TRAINING, VALIDATION AND TEST SETS FOR K-FOLD CROSS-VALIDATION - START
# # -------------------------------------------------------------------------------------------------------------------
path_file_trial_names_folds_tr = os.path.join(results_path, 'trial_names_by_id_%s_%s-%s-%s-%s_folds_tr.txt' %
                                              (config_method, config[0], config[1], config[2], config[3]))
path_file_trial_names_folds_va = os.path.join(results_path, 'trial_names_by_id_%s_%s-%s-%s-%s_folds_va.txt' %
                                              (config_method, config[0], config[1], config[2], config[3]))

# If the code has not been run and thus the trial_names for training and validation sets for k-fold cross-validation are
# yet to be defined:
if not os.path.isfile(path_file_trial_names_folds_tr) and not os.path.isfile(path_file_trial_names_folds_va):

    # The trial_names (representing gait samples) for each participant have already been shuffled in each benchmark
    # dataset. We need to aggregate the subsets of trial_names for each participant across the benchmark datasets and
    # then shuffle their order prior to defining training and validation sets. This shuffles participants/IDs to ensure
    # that the training and validation sets each contain a random mix of individuals from each dataset:
    trial_names_by_id_shuff_tr_va = np.array(random.sample(trial_names_by_id_shuff[0] +
                                                           trial_names_by_id_shuff[1] +
                                                           trial_names_by_id_shuff[2] +
                                                           trial_names_by_id_shuff[3],
                                                           k=len(trial_names_by_id_shuff[0]) +
                                                             len(trial_names_by_id_shuff[1]) +
                                                             len(trial_names_by_id_shuff[2]) +
                                                             len(trial_names_by_id_shuff[3])), dtype=object)

    # Allocate participants in the benchmark datasets to training and validation sets for five-fold cross-validation:
    kf = KFold(n_splits=n_folds)

    trial_names_by_id_folds_tr = []
    trial_names_by_id_folds_va = []

    for indices_fold_tr, indices_fold_va in kf.split(trial_names_by_id_shuff_tr_va):

        trial_names_by_id_folds_tr.append(list(trial_names_by_id_shuff_tr_va[indices_fold_tr]))
        trial_names_by_id_folds_va.append(list(trial_names_by_id_shuff_tr_va[indices_fold_va]))

    save_list(path_file_trial_names_folds_tr, trial_names_by_id_folds_tr)
    save_list(path_file_trial_names_folds_va, trial_names_by_id_folds_va)

    # Set variables that are no longer needed to None to save memory and simplify workspace:
    trial_names_by_id_shuff_tr_va = None
    indices_fold_tr = None
    indices_fold_va = None

# If the code has been run, the trial_names for training and validation sets have already been defined and saved. Thus,
# simply load here to avoid re-defining and saving each time:
elif os.path.isfile(path_file_trial_names_folds_tr) and os.path.isfile(path_file_trial_names_folds_va):

    trial_names_by_id_folds_tr = load_list(path_file_trial_names_folds_tr)
    trial_names_by_id_folds_va = load_list(path_file_trial_names_folds_va)

else:

    raise Exception('Trial names objects for training and validation sets should exist together. It seems that one of'
                    'them has been moved or deleted from the results path.')

path_file_trial_names_folds_te = os.path.join(results_path, 'trial_names_%s_%s_folds_te.txt' %
                                              (config_method, config[4]))

# If the code has not been run and thus the trial_names for test sets for k-fold cross-validation are yet to be defined:
if not os.path.isfile(path_file_trial_names_folds_te):

    # The test set is the same on each fold (ForceID-B_RS):
    trial_names_folds_te = [trial_names[4] for i in range(n_folds)]
    save_list(path_file_trial_names_folds_te, trial_names_folds_te)

# If the code has been run, the trial_names for test sets have already been defined and saved. Thus, simply load here to
# avoid re-defining and saving each time:
else:

    trial_names_folds_te = load_list(path_file_trial_names_folds_te)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. DEFINE/LOAD TRIAL_NAMES FOR TRAINING, VALIDATION AND TEST SETS FOR K-FOLD CROSS-VALIDATION - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. RUN THE EXPERIMENT BY PERFORMING K-FOLD CROSS-VALIDATION - START
# # -------------------------------------------------------------------------------------------------------------------
# Initialise lists that will be appended with results from each fold:
times_folds = []
mod_checks_folds = []
losses_tr_folds = []
losses_va_folds = []
losses_te_folds = []
accs_tr_folds = []
accs_va_folds = []
accs_te_folds = []
embs_te_folds = []

# Initialise a label encoder that transforms a set of participant ID labels for a given training, validation or test
# set to start from zero:
le = preprocessing.LabelEncoder()

for idx_fold in range(n_folds):

    print('Fold %s:' % (idx_fold + 1))

    trial_names_tr = np.concatenate(trial_names_by_id_folds_tr[idx_fold])

    # To search the feature space to generate predictions on validation and test sets, this study implemented a
    # difficult method proposed in Duncanson et al. (2023) that was later termed 'random subset search' in Duncanson
    # et al. (2024). This method requires balanced validation and test sets with n samples per individual, where n is
    # the minimum number of samples across all individuals (count_va_min and count_te_min here):
    count_va_min = np.min([len(id_subset) for id_subset in trial_names_by_id_folds_va[idx_fold]])
    trial_names_va = np.concatenate([id_subset[:count_va_min] for id_subset in trial_names_by_id_folds_va[idx_fold]])

    # The test set (ForceID-B_RS) was pre-defined to be balanced in data_processing.py (it contains two random samples
    # from each participant in ForceID-B_CFC with at least two samples). The counts_samples for ForceID-B_RS
    # contains all 2s so count_te_min = 2. Nonetheless, we define it more generally instead of hard-coding:
    count_te_min = np.min(counts_samples[4])
    trial_names_te = trial_names_folds_te[idx_fold]

    # For training, validation and test sets, get z-score normalised signals and associated ID labels. Also, define a
    # weighted random sampler for the training set:
    sigs_sets = []
    labels_sets = []
    samplers_sets = []
    # Initialise scalers to normalise each directional component:
    scalers = [preprocessing.StandardScaler() for idx_channel in range(n_channels)]

    for idx_set, trial_names_set in enumerate([trial_names_tr, trial_names_va, trial_names_te]):

        # Get the indices of the trial names for the tr/va/te set within the global set of trial names (across all
        # datasets):
        indices_trial_names_set = np.concatenate([np.asarray(trial_names_cat == name).nonzero()[0]
                                                  for name in trial_names_set])

        # Use the indices of the tr/va/te set to get the signals for each stance side:
        sigs_set_r = sigs_r[indices_trial_names_set]
        sigs_set_l = sigs_l[indices_trial_names_set]

        # Because signal direction is aligned between stance sides, we can combine signals from each stance side along
        # the sample axis to standardise them together:
        sigs_set = np.concatenate((sigs_set_r, sigs_set_l), axis=0)

        # Set variables that are no longer needed to None to save memory and simplify workspace:
        indices_trial_names_set = None
        sigs_set_r = None
        sigs_set_l = None

        # Standardise each directional component via z-score normalisation:
        sigs_set_stsd = []

        for idx_channel in range(n_channels):

            if idx_set == 0:  # Training set.

                scalers[idx_channel].fit(sigs_set[:, idx_channel])

            channel_stsd = scalers[idx_channel].transform(sigs_set[:, idx_channel])

            # 1) Expand the first axis so that the directional component can later be combined with others along that
            #    axis.
            # 2) Convert to a tensor.
            # 3) Split by stance side:
            channel_stsd_tensor = torch.split(
                torch.tensor(np.expand_dims(channel_stsd, axis=1), dtype=torch.float32),
                int(channel_stsd.shape[0] / 2))

            # Set unneeded variable to None:
            channel_stsd = None

            # Concatenate right and left stance sides along the sequence dimension (respectively). (If single_network =
            # False, the stance sides will be separated again later.):
            channel_stsd_tensor = torch.cat((channel_stsd_tensor[0], channel_stsd_tensor[1]), dim=2)

            sigs_set_stsd.append(channel_stsd_tensor)

        # Get labels for the tr/va/te set:
        labels_temp = np.array([int(name[3:7]) for name in trial_names_set])
        # Fit and transform the labels using the label encoder such that they start from zero. Then convert to a tensor:
        le.fit(labels_temp)
        labels_set = torch.tensor(le.transform(labels_temp)).long()
        # Set temporary variable to None:
        labels_temp = None

        if idx_set == 0:  # Training set.

            # Define a weighted random sampler to account for different numbers of samples per participant in the
            # training set (i.e., class imbalance). The weights are the probabilities of sampling from each participant:
            counts_samples_tr = np.unique(labels_set, return_counts=True)[1]
            weights_ids_tr = 1. / torch.tensor(counts_samples_tr, dtype=torch.float32)
            weights_samples_tr = weights_ids_tr[labels_set]
            samplers_sets.append(WeightedRandomSampler(weights_samples_tr, weights_samples_tr.size(0)))

            # Set unneeded variables to None:
            counts_samples_tr = None
            weights_ids_tr = None
            weights_samples_tr = None

        else:

            # We don't use a sampler for validation or testing:
            samplers_sets.append(None)

        # Concatenate the standardised signals for each directional component along the channel dimension (dim 1) to
        # re-form a single data object for the tr/va/te set:
        sigs_sets.append(torch.cat((sigs_set_stsd[0],
                                    sigs_set_stsd[1],
                                    sigs_set_stsd[2],
                                    sigs_set_stsd[3],
                                    sigs_set_stsd[4]), dim=1))
        labels_sets.append(labels_set)

    # Set unneeded variable to None:
    sigs_set = None

    # Initialise the model, as well as data loaders for training, validation and test sets:
    if arch == '-----2F':

        mod = TwoF(*params_mod)

    else:  # '3C1T-2F'.

        mod = ThrCOneTTwoF(*params_mod)

    if single_network:

        # Note for data loading that training data is loaded with labels, whereas validation and test data are loaded
        # without labels:
        loader_tr, loader_va, loader_te = init_data_loaders_no_labs_va_te(
            data_loader_class_tr=SingleDatasetLoaderLabels,
            data_loader_class_eval=SingleDatasetLoaderNoLabels,
            data_tr=sigs_sets[0],
            labs_tr=labels_sets[0],
            data_va=sigs_sets[1],
            data_te=sigs_sets[2],
            batch_size=bs,
            sampler_tr=samplers_sets[0])

    else:

        # Overwrite the standard model architecture with the Siamese neural network architecture:
        mod = SiameseNet(mod)

        loader_tr, loader_va, loader_te = init_data_loaders_no_labs_va_te(
            data_loader_class_tr=DualDatasetLoaderLabels,
            data_loader_class_eval=DualDatasetLoaderNoLabels,
            data_tr=(sigs_sets[0][:, :, :len_seq], sigs_sets[0][:, :, len_seq:]),
            labs_tr=labels_sets[0],
            data_va=(sigs_sets[1][:, :, :len_seq], sigs_sets[1][:, :, len_seq:]),
            data_te=(sigs_sets[2][:, :, :len_seq], sigs_sets[2][:, :, len_seq:]),
            batch_size=bs,
            sampler_tr=samplers_sets[0])

    count_model_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)

    # Initialise the optimiser:
    opt = optim.Adam(mod.parameters(), *params_opt_extra)

    # Run the training and validation loop:
    tr_va_time, accs_tr, accs_va, losses_tr, losses_va, mod_state_dict, opt_state_dict = \
        train_val(mod=mod,
                  loader_tr=loader_tr,
                  loader_va=loader_va,
                  labels_va=labels_sets[1],
                  criterion_tr=criterion_tr,
                  criterion_eval=criterion_eval,
                  count_samples_min=count_va_min,
                  opt=opt,
                  epochs=epochs,
                  es_patience=es_patience,
                  es_min_delta=es_min_delta,
                  single_network=single_network,
                  ptm=ptm,
                  miner=miner,
                  criterion_opt=criterion_opt,
                  cuda_tr=cuda_tr,
                  cuda_va=cuda_va)

    times_folds.append(tr_va_time)
    accs_tr_folds.append(accs_tr)
    accs_va_folds.append(accs_va)
    losses_tr_folds.append(losses_tr)
    losses_va_folds.append(losses_va)
    mod_checks_folds.append((mod_state_dict, opt_state_dict))

    # Load the best weights from training:
    mod.load_state_dict(mod_state_dict)

    # Set variables from training and validation that are no longer needed to None to save memory for testing:
    loader_tr = None
    loader_va = None
    accs_tr = None
    accs_va = None
    losses_tr = None
    losses_va = None
    mod_state_dict = None
    opt_state_dict = None

    # Test the trained model:
    loss_te, acc_te, embs_te = test(mod=mod,
                                    loader_te=loader_te,
                                    labels_te=labels_sets[2],
                                    criterion_eval=criterion_eval,
                                    count_samples_min=count_te_min,
                                    opt=opt,
                                    single_network=single_network,
                                    cuda=cuda_te)

    losses_te_folds.append(loss_te)
    accs_te_folds.append(acc_te)
    embs_te_folds.append(embs_te.cpu())
    # Embeddings are memory intensive, so set test embeddings to None before running the next cross-validation fold:
    embs_te = None
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. RUN THE EXPERIMENT BY PERFORMING K-FOLD CROSS-VALIDATION - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 5. SAVE RESULTS - START
# # -------------------------------------------------------------------------------------------------------------------
if not single_network:

    # Add a suffix to the arch string to indicate that a Siamese Neural Network (SNN) was used:
    arch = arch + '_SNN'

fname_prefixes = ['losses_tr', 'losses_va', 'losses_te',
                  'accs_tr', 'accs_va', 'accs_te',
                  'embs_te']

for idx, result in enumerate([losses_tr_folds, losses_va_folds, losses_te_folds,
                              accs_tr_folds, accs_va_folds, accs_te_folds,
                              embs_te_folds]):

    if fname_prefixes[idx].__contains__('embs'):

        torch.save(result, os.path.join(results_path, '%s_%s_%s-%s-%s-%s-%s_%s_%s.pth' %
                                        (fname_prefixes[idx], config_method, config[0], config[1],
                                         config[2], config[3], config[4], arch, bs)))

    else:

        save_list(os.path.join(results_path, '%s_%s_%s-%s-%s-%s-%s_%s_%s.txt' %
                               (fname_prefixes[idx], config_method, config[0], config[1],
                                config[2], config[3], config[4], arch, bs)), result)

# The checkpoints from cross-validation (model and optimiser states from each fold) aren't included in the list of
# results to save because they are memory intensive. If desired, can save by uncommenting below:
# torch.save(mod_checks_folds, os.path.join(results_path, 'mod_checks_%s_%s-%s-%s-%s-%s_%s_%s.pth' %
#                                           (config_method, config[0], config[1], config[2], config[3], config[4],
#                                            arch, bs)))
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 5. SAVE RESULTS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 6. GENERATE A FIGURE OF ACCURACY AND LOSS CURVES OVER TRAINING AND VALIDATION - START
# # -------------------------------------------------------------------------------------------------------------------
fig = plt.figure()
gs = fig.add_gridspec(1, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

for idx_fold in range(n_folds):

    if idx_fold == 0:

        ax1.plot(accs_tr_folds[idx_fold], color='#006BA4', lw=1, label='Tr')
        ax1.plot(accs_va_folds[idx_fold], color='#FF800E', lw=1, label='Va')
        ax2.plot(losses_tr_folds[idx_fold], color='#006BA4', lw=1, label='Tr')
        ax2.plot(losses_va_folds[idx_fold], color='#FF800E', lw=1, label='Va')

    else:

        ax1.plot(accs_tr_folds[idx_fold], color='#006BA4')
        ax1.plot(accs_va_folds[idx_fold], color='#FF800E')
        ax2.plot(losses_tr_folds[idx_fold], color='#006BA4')
        ax2.plot(losses_va_folds[idx_fold], color='#FF800E')

ax1.set_ylim(0, 1)
ax2.set_ylim(0, 3)

ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax2.set_yticks(np.arange(0, 3.1, 0.6))

legend_ax1 = ax1.legend(prop={'size': 10}, frameon=False)
legend_ax2 = ax2.legend(prop={'size': 10}, frameon=False)

ax1.tick_params(labelsize=10)
ax2.tick_params(labelsize=10)

ax1.set_ylabel('Accuracy', size=12, labelpad=6)
ax2.set_ylabel('Loss', size=12, labelpad=6)

fig.supxlabel('Epoch', size=12)
fig.tight_layout()

# Save the figure in a specified file type:
ftype = 'pdf' # Could alternatively use 'svg' or 'png', for example.
plt.savefig(os.path.join(results_path, 'accloss_%s_%s-%s-%s-%s-%s_%s_%s.%s' %
                         (config_method, config[0], config[1], config[2],
                          config[3], config[4], arch, bs, ftype)), dpi=1200)
plt.close()
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 6. GENERATE A FIGURE OF ACCURACY AND LOSS CURVES OVER TRAINING AND VALIDATION - END
# # -------------------------------------------------------------------------------------------------------------------
