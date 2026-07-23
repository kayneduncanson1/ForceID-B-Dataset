import random
import os
import pickle
import numpy as np
import torch
from datetime import datetime


def convert_dt_to_iso(dt_str):
    # Parse using two-digit year format:
    dt = datetime.strptime(dt_str, "%y-%m-%d_%H-%M-%S")
    # Convert to ISO 8601 format:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# Function from Maheshkar, S. (September 9, 2024). How to Set Random Seeds in PyTorch and Tensorflow.
# https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    g = torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

    return g


def save_list(filepath, obj):
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)


def load_list(filepath):
    with open(filepath, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


# Extract basic ID, session and trial metadata from the ForceID-B metadata array:
def extract_base_metadata_fib(metadata_arr):

    id_labels = metadata_arr[:, 1]
    ids, counts_trials_by_id = np.unique(id_labels, return_counts=True)
    indices_where_id_changes = np.cumsum(counts_trials_by_id)[:-1]
    # Get ID, session and trial metadata in string format:
    ids_sessions_trials = np.array([f"{el[0]:03}" + "_S%s_T%s" % (el[1], el[2]) for el in metadata_arr[:, 1:4]])
    # Get metadata on a per-ID basis:
    metadata_by_id = np.split(metadata_arr, indices_where_id_changes)

    return id_labels, ids, counts_trials_by_id, indices_where_id_changes, ids_sessions_trials, metadata_by_id


# For a given dataset specified by the string identifier 'dataset_name', load the arrays with basic metadata that were
# defined and saved in the relevant data processing script ('prepare_[dataset].py' in 'ForceID-Study-2' for the
# benchmark datasets and 'data_processing.py' in this repo for ForceID-B_RS). This function is the same as the
# 'get_base_metadata' function in the ForceID-Study-2 repo except for the function name and the capital 'O' for the
# 'Objects' subdirectory:
def load_base_metadata(dataset_name):

    # Trial name identifiers:
    trial_names = np.load('./Datasets/%s/Objects/trial_names.npy' % dataset_name, allow_pickle=True)

    # ID labels:
    labels = np.load('./Datasets/%s/Objects/labels.npy' % dataset_name, allow_pickle=True)

    ids = np.load('./Datasets/%s/Objects/ids.npy' % dataset_name, allow_pickle=True)

    # The number/count of samples per ID (denoted counts_samples_by_id in data_processing.py):
    counts_samples = np.load('./Datasets/%s/Objects/counts_samples.npy' % dataset_name, allow_pickle=True)

    return trial_names, labels, ids, counts_samples


# For a given benchmark dataset specified by the string identifier 'dataset_name', load the metadata arrays that were
# defined and saved in data processing scripts in the ForceID-Study-2 repo ('prepare_[dataset].py'). This function is
# the same as the 'get_full_metadata' function in the ForceID-Study-2 repo except for the function name, the capital
# 'O' for the 'Objects' subdirectories and the order in which the demographic variables are returned:
def load_full_metadata(dataset_name):

    # Trial name identifiers:
    trial_names = np.load('./Datasets/%s/Objects/trial_names.npy' % dataset_name, allow_pickle=True)

    # ID labels:
    labels = np.load('./Datasets/%s/Objects/labels.npy' % dataset_name, allow_pickle=True)

    ids = np.load('./Datasets/%s/Objects/ids.npy' % dataset_name, allow_pickle=True)

    # The number/count of samples per ID. In this implementation, a sample is considered a sequence of two consecutive
    # stance phases from a single walking trial:
    counts_samples = np.load('./Datasets/%s/Objects/counts_samples.npy' % dataset_name, allow_pickle=True)

    # Demographic attributes:
    ages = np.load('./Datasets/%s/Objects/ages.npy' % dataset_name, allow_pickle=True)
    sexes = np.load('./Datasets/%s/Objects/sexes.npy' % dataset_name, allow_pickle=True)
    masses = np.load('./Datasets/%s/Objects/masses.npy' % dataset_name, allow_pickle=True)
    heights = np.load('./Datasets/%s/Objects/heights.npy' % dataset_name, allow_pickle=True)

    # Footwear categories:
    footwear = np.load('./Datasets/%s/Objects/footwear.npy' % dataset_name, allow_pickle=True)

    # Walking speed categories:
    speeds = np.load('./Datasets/%s/Objects/speeds.npy' % dataset_name, allow_pickle=True)

    return trial_names, labels, ids, counts_samples, ages, sexes, masses, heights, footwear, speeds
