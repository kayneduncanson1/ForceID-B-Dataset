import numpy as np
import pandas as pd
import os
from Utils import extract_base_metadata_fib

"""The dataset was originally stored in a directory named 'ID dirs' with the following sub-directory and file structure:
- There was a sub-directory for each participant named with their identification (ID) number in three-digit format
  (e.g., 010 or 209).
- Within each ID sub-directory were files for GRF data, as well as files for metadata (session details, photos and
  videos).
- Each of the 1362 CSV files contained GRF data for a given walking trial, with measurement frames across rows and
  directional components across columns (8 directional components x 3 force platforms = 24 columns).
This script re-formats the dataset into a smaller set of 24 csv files for publication, where each file contains data
from all trials for a given directional component and force platform. The path './Datasets/fi-b-all/ID dirs' at line 36
below is an empty placeholder because the original directory is private. This script is for demonstrative purposes only
as running returns an error."""

# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. GET BASIC METADATA FOR FORCEID-B - START
# # -------------------------------------------------------------------------------------------------------------------
metadata = pd.read_csv('./Datasets/fi-b-all/Spreadsheets/Metadata.csv', usecols=np.arange(40)).values

# The ids_sessions_trials object returned by the function below contains ID, session and trial numbers in string format
# with the same nomenclature as the original csv file names. This enables identifying the data files to read their data:
id_labels, ids, counts_trials_by_id, indices_where_id_changes, ids_sessions_trials, metadata_by_id =\
    extract_base_metadata_fib(metadata)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. GET BASIC METADATA FOR FORCEID-B - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. IN THE ORIGINAL DATASET DIRECTORY, LOOP OVER SUB-DIRECTORIES FOR EACH ID AND CSV FILES FOR INDIVIDUAL
# # TRIALS. READ, REFORMAT AND WRITE DATA TO NEW CSV FILES - START
# # -------------------------------------------------------------------------------------------------------------------
path_dir_ds = './Datasets/fi-b-all/ID dirs'
fp_nos = ['FP1', 'FP2', 'FP3']
channels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy']
n_channels = len(channels)
indices_channels = np.arange(n_channels)
len_sigs_fixed = 20100  # Length to pad the signals. Max length was 20096 which was approximately 20 seconds at 1000 Hz.
lens_sigs = []

# Define the first column of each new dataset csv file to contain row headings for ID, session, trial and frame numbers
# (with the latter starting from 1):
row_headings = np.concatenate((np.array(['PARTICIPANT_ID', 'SESSION_NUMBER', 'TRIAL_NUMBER']),
                               np.arange(1, len_sigs_fixed + 1)))[:, None]

for fp_no in fp_nos:

    dataset = []

    for id_session_trial in ids_sessions_trials:

        path_dir_id = os.path.join(path_dir_ds, id_session_trial[:3])  # id_session_trial[:3] is the three-digit ID.

        for fname in os.listdir(path_dir_id):

            if fname.__contains__('.csv') and fname[:9] == id_session_trial:

                data = pd.read_csv(os.path.join(path_dir_id, fname), usecols=np.arange(1, 25)).values
                lens_sigs.append(data.shape[0])
                pad_width = len_sigs_fixed - data.shape[0]

                if fp_no == 'FP1':

                    sig = data[:, :8]

                elif fp_no == 'FP2':

                    sig = data[:, 8:16]

                elif fp_no == 'FP3':

                    sig = data[:, 16:]

                else:

                    raise Exception("Invalid string entry for force platform number (fp_no). Options are FP1, FP2 and"
                                    "FP3.")

                # sig is shape (L, C), where L = sequence length and C = number of channels.
                sig_padded = np.pad(sig, pad_width=((0, pad_width), (0, 0)), constant_values=np.nan)
                dataset.append(sig_padded)

    dataset = np.array(dataset)
    # dataset is shape (N, L, C), where N = number of trials.

    for idx_channel in indices_channels:

        # To format data from each frame across rows and data from each trial across columns, combine the following:
        # - Transposed slice of the metadata array containing ID, session and trial numbers.
        # - Transposed data array for the channel/component.
        dataset_channel = np.concatenate((metadata[:, 1:4].transpose(), dataset[:, :, idx_channel].transpose()), axis=0)
        excel_grid = np.concatenate((row_headings, dataset_channel), axis=1)
        df = pd.DataFrame(excel_grid)
        df.to_csv('./Datasets/fi-b-all/Spreadsheets/ForceID-B_%s_%s_raw.csv' % (channels[idx_channel], fp_no),
                  header=False,
                  index=False)

lens_sigs = np.array(lens_sigs)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. IN THE ORIGINAL DATASET DIRECTORY, LOOP OVER SUB-DIRECTORIES FOR EACH ID AND CSV FILES FOR INDIVIDUAL
# # TRIALS. READ, REFORMAT AND WRITE DATA TO NEW CSV FILES - END
# # -------------------------------------------------------------------------------------------------------------------
