import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import decimate
from scipy.interpolate import interp1d
from PrePro import butterworth_lowpass
from Utils import set_seed, extract_base_metadata_fib

"""This script defines and processes two subsets of ForceID-B:
1) The ForceID-B_CFC subset contains samples of the GRF and COP during the first two consecutive stance phases from
complete foot contacts ('CFC') in trials with at least two consecutive complete foot contacts. In fig_sigs.py, this
subset is visualised against another subset of ForceID-B containing partial foot contacts from two random IDs, as well
as four previous large-scale walking GRF datasets referred herein as benchmark datasets: AIST, Gutenberg, GaitRec and
ForceID-A.
2) The ForceID-B_RS subset contains two random samples ('RS') from each of the participants in ForceID-B_CFC with at
least two samples. This subset is used in data_analysis.py, where it is 'pre-processed' further for Machine Learning
(ML) and then applied for ML experiments as part of the technical validation."""

# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. INITIALISE PARAMETERS - START
# # -------------------------------------------------------------------------------------------------------------------
# Set random seed for reproducibility:
seed = 42
g = set_seed(seed)

# Specify the data subset to process:
fib_subset_to_process = 'CFC' # Options: 'CFC' and 'RS'.

# Specify the number of random samples that are selected from each ID in the ForceID-B_CFC subset to define the
# ForceID-B_RS subset:
n_rs = 2

# Specify the total number of participants (IDs) across all benchmark datasets to create unique trial identifiers. These
# are later saved and then loaded into data_analysis.py to form a database for ML model development:
n_ids_bm = 1050

# Specify the directional components to be included. The benchmark datasets contain 3D ground reaction force (GRF) and
# 2D center of pressure (COP), so we also include these components:
channels = ['Fx', 'Fy', 'Fz', 'Cx', 'Cy']
n_channels = len(channels)
indices_channels = np.arange(n_channels)

# # The remaining parameters are used in Section 6 to process the selected subset (ForceID-B_CFC/ForceID-B_RS). They are
# # initialised here and soft-coded in Section 6 so that they can be easily adjusted in one place. For more context on
# # the parameters, please click through to their usage locations and see associated comments.

# Vertical GRF (Fz) threshold value for segmenting the portion of each signal pertaining to the stance phase (units:
# newton N):
fz_thresh = 50

# The number of frames over which to calculate the mean unloaded/baseline value of each GRF component to correct GRF
# offset due to signal drift. If changing the value, check that no force platform is loaded throughout the period
# (e.g., by visually inspecting the vertical GRF component as in Section 5):
n_frames_cal = 500

# The GRF is initially segmented to include the approximate stance phase plus a buffer of a specified number of frames
# at each end to avoid edge effects during subsequent interpolation. If changing the setting, check that all signals
# have at least the specified number of frames at each end to avoid returning an error (Section 6):
n_frames_stance_buffer_grf = 20

# The COP is segmented to include the approximate stance phase with an additional portion excluded from each end to
# avoid inaccurate COP measurements at low force values. For example, 0.05 below means 5% of the length of the
# approximate stance phase is excluded from each end, retaining the middle 90%:
portion_to_exclude_cop = 0.05

# Bidirectional Butterworth low-pass filter order:
filt_order = 4

# Bidirectional Butterworth low-pass filter cut-off frequency (units: Hertz Hz):
filt_cutoff = 30

# Bidirectional Butterworth low-pass filter pad length in number of frames. If changing the setting, check that the
# minimum COP sequence length is greater than the filter pad length to avoid returning an error (Section 6):
filt_n_frames_pad = 40

# Original force platform sampling rate (units: Hz):
sampling_rate = 1000

# Down-sampled rate (units: Hz).
sampling_rate_dns = 250

# Number of frames over which to interpolate the signals for the purpose of time normalisation:
len_interp = 100

# We show how to detect the stance side using a simple heuristic based on the left-to-right GRF. Namely, the data are
# assigned a stance side according to whether this directional component is positive or negative at a specified
# percentage of the stance phase. We experimented with values around 35(%) based on preliminary unpublished work that
# contributed to Duncanson et al. (2021). Stance side variables are denoted 'ss':
percent_ss_split = 39
proportion_ss_split = percent_ss_split / 100
frame_ss_split = int(np.round(proportion_ss_split * len_interp))
idx_ss_split = frame_ss_split - 1  # Accounting for zero-indexing.

# Global parameters for plot formatting:
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('tableau-colorblind10')
mpl.rcParams['mathtext.default'] = 'regular'
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. INITIALISE PARAMETERS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. DEFINE FORCEID-B_CFC - START
# # -------------------------------------------------------------------------------------------------------------------
# Get metadata for ForceID-B:
metadata = pd.read_csv('./Datasets/fi-b-all/Spreadsheets/Metadata.csv', usecols=np.arange(40)).values
id_labels, ids, counts_trials_by_id, indices_where_id_changes, ids_sessions_trials, metadata_by_id =\
    extract_base_metadata_fib(metadata)

# Get indices of trials with two consecutive complete foot contacts:
indices_cfc = []

for i in range(metadata.shape[0]):

    # In the metadata spreadsheet, the location of a given foot on the force platform and the region of the foot on the
    # platform are represented in separate columns. Hence, four consecutive columns with 'C' represents two consecutive
    # complete foot contacts on two force platforms:
    if np.all((metadata[i, [27, 28, 31, 32]] == np.array(['C', 'C', 'C', 'C']))) \
            or np.all((metadata[i, [31, 32, 35, 36]] == np.array(['C', 'C', 'C', 'C']))):

        indices_cfc.append(i)

indices_cfc = np.array(indices_cfc)

# Use the indices to get metadata for ForceID-B_CFC:
metadata_cfc = metadata[indices_cfc]
id_labels_cfc, ids_cfc, counts_trials_by_id_cfc, indices_where_id_changes_cfc, ids_sessions_trials_cfc,\
    metadata_by_id_cfc = extract_base_metadata_fib(metadata_cfc)

# Save the metadata:
np.save('./Datasets/fi-b-cfc/Objects/metadata_cfc.npy', metadata_cfc)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. DEFINE FORCEID-B_CFC - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. DEFINE FORCEID-B_RS - START
# # -------------------------------------------------------------------------------------------------------------------
# Get the subset of IDs who had n_rs (or more) samples denoted by '..._n'. Noting that 1 trial = 1 sample in
# ForceID-B_CFC:
indices_ids_n = np.array([id_ for id_, count in enumerate(counts_trials_by_id_cfc) if count >= n_rs])
ids_n = ids_cfc[indices_ids_n]
# Get the indices of the ID labels for these IDs:
indices_id_labels_n = np.concatenate([np.asarray(id_labels_cfc == id_).nonzero()[0] for id_ in ids_n])
# Get metadata for these IDs:
metadata_n = metadata_cfc[indices_id_labels_n]
# _ is used in place of ids_n below because ids_n was defined above:
id_labels_n, _, counts_samples_by_id_n, indices_where_id_changes_n, ids_sessions_trials_n, metadata_by_id_n =\
    extract_base_metadata_fib(metadata_n)

# Get indices of random samples for each ID:
indices_rs_by_id = [np.array(random.sample(list(np.arange(count)), k=n_rs)) for count in counts_samples_by_id_n]

# Initialise counts for the number of participants with random samples from different sessions versus the same session.
# Note: The script is set up such that users can define a different random sample subset to that used in this work by
# changing the n_rs parameter in Section 1 above. However, the conditional in the loop below to get the counts for
# session change vs no session change is hard-coded for n_rs = 2, so the code is commented:
# count_session_change_rs = 0
# count_no_session_change_rs = 0

metadata_by_id_rs = []

for id_, metadata_subset in enumerate(metadata_by_id_n):

    indices_rs = indices_rs_by_id[id_]
    metadata_subset_rs = metadata_subset[indices_rs]
    metadata_by_id_rs.append(metadata_subset_rs)

    # if metadata_subset_rs[0, 2] == metadata_subset_rs[1, 2]:
    #
    #     count_no_session_change_rs += 1
    #
    # elif metadata_subset_rs[0, 2] != metadata_subset_rs[1, 2]:
    #
    #     count_session_change_rs += 1

# percent_session_change_rs = (count_session_change_rs / (count_session_change_rs + count_no_session_change_rs)) * 100

# Get the metadata array for ForceID-B_RS in long form:
metadata_rs = np.concatenate(metadata_by_id_rs)

# Extract basic metadata for ForceID-B_RS. Note that the '_' variable is used below because metadata_by_id_rs was
# already defined above:
id_labels_rs, ids_rs, counts_samples_by_id_rs, indices_where_id_changes_rs, ids_sessions_trials_rs, _ =\
    extract_base_metadata_fib(metadata_rs)

# Set variables that are no longer needed to None to save memory and simplify workspace:
indices_ids_n = None
indices_id_labels_n = None
metadata_n = None
id_labels_n = None
ids_n = None
counts_samples_by_id_n = None
indices_where_id_changes_n = None
metadata_by_id_n = None
indices_rs_by_id = None
_ = None

# Save the metadata:
np.save('./Datasets/fi-b-rs/Objects/metadata_rs.npy', metadata_rs)

# As was done for the benchmark datasets in the ForceID-Study-2 repo from Duncanson et al. (2024), make a trial names
# object with unique trial identifiers. These identifiers must be unique across ForceID-B_RS and all benchmark datasets
# to enable their integration for ML gait recognition experiments (data_analysis.py). The number of IDs across the
# benchmark datasets n_ids_bm = 1050, so we redefine the ID labels in ForceID-B_RS to start from 1051:
metadata_copy = metadata_rs.copy()
metadata_copy[:, 1] = metadata_copy[:, 1] + n_ids_bm + 1

# ForceID-A is denoted 'FI_...' in the trial names, whereas ForceID-B is denoted 'Fi_...'. Ideally, the nomenclature
# would be revised to have a three letter dataset identifier or something, but we have kept it consistent with that used
# in the ForceID-Study-2 repo for now:
trial_names = np.array(['_'.join(['Fi',
                                  format(el[1], '04d'),
                                  format(el[2], '02d'),
                                  format(el[3], '02d'),
                                  'P'])
                        for el in metadata_copy]) # P = self-selected 'preferred' walking speed.

# Save the trial names and other metadata objects that are required for data_analysis.py:
np.save('./Datasets/fi-b-rs/Objects/trial_names.npy', trial_names)
np.save('./Datasets/fi-b-rs/Objects/labels.npy', metadata_copy[:, 1])
np.save('./Datasets/fi-b-rs/Objects/ids.npy', np.unique(metadata_copy[:, 1]))
np.save('./Datasets/fi-b-rs/Objects/counts_samples.npy', counts_samples_by_id_rs)

# Set the copied metadata array to None to save memory:
metadata_copy = None
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. DEFINE FORCEID-B_RS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. READ THE DATASET AND SELECT DATA FOR THE SUBSET - START
# # -------------------------------------------------------------------------------------------------------------------
# From here on, the same process is completed on the ForceID-B_CFC and ForceID-B_RS subsets. Metadata variables for the
# selected subset are re-allocated using a common nomenclature (denoted '..._sel'):
if fib_subset_to_process == 'CFC':

    metadata_sel = metadata_cfc.copy()
    id_labels_sel = id_labels_cfc.copy()
    ids_sel = ids_cfc.copy()
    counts_samples_by_id_sel = counts_trials_by_id_cfc.copy()
    indices_where_id_changes_sel = indices_where_id_changes_cfc.copy()
    ids_sessions_trials_sel = ids_sessions_trials_cfc.copy()
    metadata_by_id_sel = metadata_by_id_cfc.copy()

else: # 'RS'

    metadata_sel = metadata_rs.copy()
    id_labels_sel = id_labels_rs.copy()
    ids_sel = ids_rs.copy()
    counts_samples_by_id_sel = counts_samples_by_id_rs.copy()
    indices_where_id_changes_sel = indices_where_id_changes_rs.copy()
    ids_sessions_trials_sel = ids_sessions_trials_rs.copy()
    metadata_by_id_sel = metadata_by_id_rs.copy()

# Get the indices of the selected subset in the full dataset:
indices_ids_sessions_trials_sel = np.array([np.asarray(ids_sessions_trials == id_session_trial).nonzero()[0]
                                            for id_session_trial in ids_sessions_trials_sel]).flatten()

# Read the specified channels for the selected subset:
sigs_raw_fp1 = [pd.read_csv('./Datasets/fi-b-all/Spreadsheets/ForceID-B_%s_FP1_raw.csv' % channel,
                            usecols=np.arange(1, 1363),
                            skiprows=2).values.transpose()[indices_ids_sessions_trials_sel]
                for channel in channels]
sigs_raw_fp2 = [pd.read_csv('./Datasets/fi-b-all/Spreadsheets/ForceID-B_%s_FP2_raw.csv' % channel,
                            usecols=np.arange(1, 1363),
                            skiprows=2).values.transpose()[indices_ids_sessions_trials_sel]
                for channel in channels]
sigs_raw_fp3 = [pd.read_csv('./Datasets/fi-b-all/Spreadsheets/ForceID-B_%s_FP3_raw.csv' % channel,
                            usecols=np.arange(1, 1363),
                            skiprows=2).values.transpose()[indices_ids_sessions_trials_sel]
                for channel in channels]

sigs_raw_fp1 = np.array(sigs_raw_fp1).transpose((1, 0, 2))
sigs_raw_fp2 = np.array(sigs_raw_fp2).transpose((1, 0, 2))
sigs_raw_fp3 = np.array(sigs_raw_fp3).transpose((1, 0, 2))

# Get signals from the first two force platforms with consecutive complete foot contacts (denoted fpA and fpB):
sigs_raw_fpA = []
sigs_raw_fpB = []
fp_nos = []

for i in range(metadata_sel.shape[0]):

    if np.all((metadata_sel[i, [27, 28, 31, 32]] == np.array(['C', 'C', 'C', 'C']))):

        sigs_raw_fpA.append(sigs_raw_fp1[i])
        sigs_raw_fpB.append(sigs_raw_fp2[i])
        fp_nos.append([1, 2])

    else:

        sigs_raw_fpA.append(sigs_raw_fp2[i])
        sigs_raw_fpB.append(sigs_raw_fp3[i])
        fp_nos.append([2, 3])

sigs_raw_fpA = np.array(sigs_raw_fpA)
sigs_raw_fpB = np.array(sigs_raw_fpB)
sigs_raw_fpA_by_id = np.split(sigs_raw_fpA, indices_where_id_changes_sel)
sigs_raw_fpB_by_id = np.split(sigs_raw_fpB, indices_where_id_changes_sel)
fp_nos = np.array(fp_nos)

# Set data variables that are no longer needed to None to save memory:
sigs_raw_fp1 = None
sigs_raw_fp2 = None
sigs_raw_fp3 = None
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. READ THE DATASET AND SELECT DATA FOR THE SUBSET - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 5. VISUALISE RAW VERTICAL GRF (Fz) MEASUREMENTS TO IDENTIFY ISSUES AND INFORM SIGNAL PROCESSING - START
# # -------------------------------------------------------------------------------------------------------------------
# Plot Fz measurements from all participants colour coded by force platform:
for i in range(sigs_raw_fpA.shape[0]):

    # In the raw dataset, Fz values are negative because the positive Fz axis points downward into a given force
    # platform and the GRF vector points upward. Multiply Fz values at channel index 2 by -1 to make them positive
    # for visualisation:
    plt.plot(sigs_raw_fpA[i, 2] * -1, color='#006BA4')
    plt.plot(sigs_raw_fpB[i, 2] * -1, color='#FF800E')

plt.savefig('./Figures/ForceID-B_%s_Fz_raw.pdf' % fib_subset_to_process, dpi=1200)
# plt.show()

# Plot Fz measurements from each participant in a separate figure:
for idx_id in range(len(sigs_raw_fpA_by_id)):

    subset_fpA = sigs_raw_fpA_by_id[idx_id]
    subset_fpB = sigs_raw_fpB_by_id[idx_id]

    fig = plt.figure(constrained_layout=True)

    for idx_sample in range(subset_fpA.shape[0]):

        plt.plot(subset_fpA[idx_sample, 2] * -1, color='black')
        plt.plot(subset_fpB[idx_sample, 2] * -1, color='black')

    plt.title("Raw Fz measurements from ID %s" % ids_sel[idx_id])
    plt.savefig('./Figures/ForceID-B_%s_Fz_raw_ID%s.pdf' % (fib_subset_to_process, ids_sel[idx_id]), dpi=1200)
    plt.close()

# Signal issues in ForceID-B_CFC (encompassing ForceID-B_RS):
# Trial(s)               | Issue(s)
# --------------------------------------------------------------------------------------------------------------------
# ID 005 S1 T1           | Negative offset in both force platforms. Baseline noise after stance in FPA.
# ID 025 S1 T1 and T2    | Negative offset in FPB.
# ID 028 S1 T1 and T2    | Positive offset in FPA.
# ID 146 S1 T2 and T3    | Positive offset in both force platforms.
# ID 149 S1 T1 and T3    | Positive offset in FPA.
# ID 168 S1 T1           | Positive offset in FPA. Baseline noise and potential offset in FPB. Not sure whether the
#                        | measurements are truly offset because they switch back-and-forth between the offset value
#                        | and zero while the force platform is unloaded.
# --------------------------------------------------------------------------------------------------------------------

# Items flagged for action during signal processing (Section 6):
# Cases of offset to be corrected except the uncertain case of ID 168 S1 T1.
# Cases of baseline noise to be addressed via standard digital filtering.
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 5. VISUALISE RAW VERTICAL GRF (Fz) MEASUREMENTS TO IDENTIFY ISSUES AND INFORM SIGNAL PROCESSING - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 6. PROCESS SIGNALS: TRANSFORM COORDINATE SYSTEM, SEGMENT STANCE PHASE, DOWN-SAMPLE, FILTER, TIME NORMALISE
# # AND DETECT STANCE SIDES - START
# # -------------------------------------------------------------------------------------------------------------------
counts_frames_to_stance_start_raw = []
counts_frames_from_stance_end_raw = []
lens_cop_dns = []
counts_frames_to_stance_start_filt = []
counts_frames_from_stance_end_filt = []
sigs_pro_fps = []

# For all signals from each force platform:
for idx, sigs_raw_fp in enumerate([sigs_raw_fpA, sigs_raw_fpB]):

    # Transform the force platform coordinate system to the following right-handed coordinate system:
    # x = left-to-right axis with the positive direction to the right of the walking direction.
    # y = front-to-back axis with the positive direction forward.
    # z = vertical axis with the positive direction upward, out of a given force platform.

    # The directional components are represented along different channels:
    # Fx = channel 1 (index 0)
    # Fy = channel 2 (index 1)
    # Fz = channel 3 (index 2)
    # Cx = channel 4 (index 3)
    # Cy = channel 5 (index 4)

    # Currently, the first channel representing Fx contains front-to-back GRF measurements, whereas the second channel
    # representing Fy contains left-to-right GRF measurements. Hence, these channels must be swapped:
    sigs_raw_copy = sigs_raw_fp.copy()
    sigs_raw_fp[:, 0] = sigs_raw_copy[:, 1]
    sigs_raw_fp[:, 1] = sigs_raw_copy[:, 0]

    # The third channel representing Fz contains vertical GRF measurements as intended but the axis points downward,
    # into a given force platform. Multiply the values by -1 to flip the axis direction:
    sigs_raw_fp[:, 2] = sigs_raw_copy[:, 2] * -1

    # The x and y axes must also be swapped for COP measurements. Then the antero-posterior COP, Cy, must be multiplied
    # by -1 because it was calculated using Fz from the original coordinate system with z pointing downward rather than
    # upward:
    sigs_raw_fp[:, 3] = sigs_raw_copy[:, 4]
    sigs_raw_fp[:, 4] = sigs_raw_copy[:, 3] * -1

    # Set the copied variable to None to save memory:
    sigs_raw_copy = None

    # For each signal, segment the stance phase, down-sample, filter and time normalise:
    sigs_pro_fp = []

    for i in range(sigs_raw_fp.shape[0]):

        # Correct offset GRF measurements by subtracting the mean of each directional component over n_frames_cal
        # (specified in Section 1). In the following trials, GRF measurements from both force platforms are offset:
        if ids_sessions_trials_sel[i] in ['005_S1_T1',
                                          '146_S1_T2',
                                          '146_S1_T3']:

            sigs_raw_fp[i, :3] = sigs_raw_fp[i, :3] - np.mean(sigs_raw_fp[i, :3, :n_frames_cal], axis=1)[:, None]

        # In the following trials, GRF measurements from FPA are offset:
        elif ids_sessions_trials_sel[i] in ['028_S1_T1',
                                            '028_S1_T2',
                                            '149_S1_T1',
                                            '149_S1_T3',
                                            '168_S1_T1'] and idx == 0:

            sigs_raw_fp[i, :3] = sigs_raw_fp[i, :3] - np.mean(sigs_raw_fp[i, :3, :n_frames_cal], axis=1)[:, None]

        # In the following trials, GRF measurements from FPB are offset:
        elif ids_sessions_trials_sel[i] in ['025_S1_T1',
                                            '025_S1_T2'] and idx == 1:

            sigs_raw_fp[i, :3] = sigs_raw_fp[i, :3] - np.mean(sigs_raw_fp[i, :3, :n_frames_cal], axis=1)[:, None]

        # Get the start and end indices for an approximate stance phase segment based on the Fz threshold value:
        start_1 = np.asarray(sigs_raw_fp[i, 2] > fz_thresh).nonzero()[0][0]
        end_1 = np.asarray(sigs_raw_fp[i, 2] > fz_thresh).nonzero()[0][-1]

        # To check the margins in terms of number of frames on either side of the approximate stance phase in all raw
        # signals, can uncomment the following two lines and comment out the rest of the loop:
        # counts_frames_to_stance_start_raw.append(start_1)
        # counts_frames_from_stance_end_raw.append(sigs_raw_fp[i, 2].shape[0] - end_1)

        # For the GRF, retain n_frames_stance_buffer_grf at each end:
        start_grf = start_1 - n_frames_stance_buffer_grf
        end_grf = end_1 + n_frames_stance_buffer_grf

        # Relative to the length of the approximate stance phase in number of frames, remove a further
        # portion_to_exclude_cop from each end (rounded to the nearest frame):
        start_cop = start_1 + int(np.round(portion_to_exclude_cop * (end_1 - start_1)))
        end_cop = end_1 - int(np.round(portion_to_exclude_cop * (end_1 - start_1)))

        # Based on the start and end indices defined above, segment and then down-sample the GRF and COP:
        grf_dns = decimate(sigs_raw_fp[i, :3, start_grf:end_grf],
                           q=int(sampling_rate / sampling_rate_dns),
                           axis=1)
        cop_dns = decimate(sigs_raw_fp[i, 3:, start_cop:end_cop],
                           q=int(sampling_rate / sampling_rate_dns),
                           axis=1)

        # We checked the lengths of the COP signals to ensure that minimum signal length was greater than filter pad
        # length. This can be done by uncommenting the line below and commenting out the rest of the loop:
        # lens_cop_dns.append(cop_dns.shape[1])

        # Filter the GRF and COP using a bidirectional Butterworth low-pass filter:
        grf_filt = butterworth_lowpass(grf_dns,
                                       order=filt_order,
                                       normal_cutoff=filt_cutoff / (0.5 * sampling_rate_dns),
                                       pad_len=filt_n_frames_pad)
        cop_filt = butterworth_lowpass(cop_dns,
                                       order=filt_order,
                                       normal_cutoff=filt_cutoff / (0.5 * sampling_rate_dns),
                                       pad_len=filt_n_frames_pad)

        # Define indices where the filtered Fz exceeds the Fz threshold value:
        start_2 = np.asarray(grf_filt[2] > fz_thresh).nonzero()[0][0]
        end_2 = np.asarray(grf_filt[2] > fz_thresh).nonzero()[0][-1]

        # To check the margins in terms of number of frames on either side of the approximate stance phase in all
        # down-sampled and filtered signals, can uncomment the following two lines and comment out the rest of the loop.
        # Using our method for time normalisation, the margins must be at least 1 frame to enable interpolation between
        # the frames before and after the vertical GRF crossed the threshold:
        # counts_frames_to_stance_start_filt.append(start_2)
        # counts_frames_from_stance_end_filt.append(grf_filt.shape[1] - end_2)

        # Interpolate between the frames before and after Fz crossed the threshold to determine when it equalled the
        # threshold:
        interp_start = interp1d([grf_filt[2, start_2 - 1],
                                 grf_filt[2, start_2]],
                                [start_2 - 1, start_2],
                                axis=0)
        interp_end = interp1d([grf_filt[2, end_2],
                               grf_filt[2, end_2 + 1]],
                              [end_2, end_2 + 1],
                              axis=0)

        t_start = interp_start(fz_thresh)
        t_end = interp_end(fz_thresh)

        # Interpolate GRFs to contain len_interp number of frames using the time points defined above:
        t_grf = np.arange(0, grf_filt.shape[1], 1)
        interp_func = interp1d(t_grf, grf_filt, axis=1)
        t_new_grf = np.linspace(t_start, t_end, len_interp)
        grf_interp = interp_func(t_new_grf)

        # Interpolate COP coordinates to contain len_interp number of frames:
        t_cop = np.linspace(0, len_interp, cop_filt.shape[1])
        interp_func = interp1d(t_cop, cop_filt, axis=1)
        t_new_cop = np.linspace(0, len_interp, len_interp)
        cop_interp = interp_func(t_new_cop)

        # Set COP coordinates to start at (0, 0):
        cop_zeroed = cop_interp - np.expand_dims(cop_interp[:, 0], axis=1)

        # Concatenate the GRF and COP to reform a single object:
        grf_cop_pro = np.concatenate((grf_interp, cop_zeroed), axis=0)
        sigs_pro_fp.append(grf_cop_pro)

    sigs_pro_fps.append(sigs_pro_fp)

# Overwrite the data object that is split by force platform such that signals from each force platform are concatenated
# along the channel axis:
sigs_pro_fps = np.concatenate((sigs_pro_fps[0], sigs_pro_fps[1]), axis=1)

# Apply the stance side detection heuristic on each sample of bilateral GRF and COP measurements. To re-iterate, the
# signal from each force platform is assigned a stance side according to whether the left-to-right GRF Fx is positive or
# negative at percent_ss_split of the stance phase (frame index idx_ss_split). If they wish, users can experiment with
# different settings of percent_ss_split and validate by running up to Section 8 inclusive:
ss = []
fp_nos_r = []
fp_nos_l = []
sigs_pro_r = []
sigs_pro_l = []

for idx_sample, sample in enumerate(sigs_pro_fps):

    if sample[0, idx_ss_split] >= 0:

        ss.append(['R', 'L'])
        fp_nos_r.append(fp_nos[idx_sample, 0])
        fp_nos_l.append(fp_nos[idx_sample, 1])
        sigs_pro_r.append(sample[:5])
        sigs_pro_l.append(sample[5:])

    else:

        ss.append(['L', 'R'])
        fp_nos_l.append(fp_nos[idx_sample, 0])
        fp_nos_r.append(fp_nos[idx_sample, 1])
        sigs_pro_l.append(sample[:5])
        sigs_pro_r.append(sample[5:])

# Convert lists to arrays:
ss = np.array(ss)
fp_nos_r = np.array(fp_nos_r)
fp_nos_l = np.array(fp_nos_l)
sigs_pro_r = np.array(sigs_pro_r)
sigs_pro_l = np.array(sigs_pro_l)

# Create a data array with signals from each stance side concatenated along the channel axis:
sigs_pro_ss = np.concatenate((sigs_pro_r, sigs_pro_l), axis=1)

# Create versions of data objects that are split by ID:
sigs_pro_fps_by_id = np.split(sigs_pro_fps, indices_where_id_changes_sel)
sigs_pro_ss_by_id = np.split(sigs_pro_ss, indices_where_id_changes_sel)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 6. PROCESS SIGNALS: TRANSFORM COORDINATE SYSTEM, SEGMENT STANCE PHASE, DOWN-SAMPLE, FILTER, TIME NORMALISE
# # AND DETECT STANCE SIDES - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 7. VALIDATE PROCESSED SIGNALS FROM EACH PARTICIPANT VIA VISUAL INSPECTION - START
# # -------------------------------------------------------------------------------------------------------------------
# Plot samples from each participant with signals from each stance phase colour coded by force platform or stance side.
# This serves three main purposes:
# 1) Secondary screen for signal issues that could have been missed in the initial screen in Section 5 or not
#    adequately addressed via signal processing in Section 6.
# 2) Screen for limitations and errors in the signal processing method.
# 3) Assess the effectiveness of the stance side detection heuristic. The Fx subplot includes horizontal and vertical
#    lines that form an intersection point at the percentage of the stance phase where stance side detection was
#    applied.

# If colour coding by force platform (FPA vs FPB):
# for idx_id, subset in enumerate(sigs_pro_fps_by_id):
# If colour coding by stance side:
for idx_id, subset in enumerate(sigs_pro_ss_by_id):

    fig = plt.figure(constrained_layout=True, figsize=(3.5, 7.0))
    gs = fig.add_gridspec(5, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[4, 0])

    for idx_sample, sample in enumerate(subset):

        # Plot each directional component on a separate axis, colour coded by force platform / stance side depending
        # on the option commented above at loop start:
        ax1.plot(sample[0], color='#006BA4')
        ax1.plot(sample[5], color='#FF800E')
        ax2.plot(sample[1], color='#006BA4')
        ax2.plot(sample[6], color='#FF800E')
        ax3.plot(sample[2], color='#006BA4')
        ax3.plot(sample[7], color='#FF800E')
        ax4.plot(sample[3], color='#006BA4')
        ax4.plot(sample[8], color='#FF800E')
        ax5.plot(sample[4], color='#006BA4')
        ax5.plot(sample[9], color='#FF800E')

    # Make the intersection point at the coordinates where the stance detection heuristic was applied:
    ax1.axhline(0, lw=1, color='black', alpha=0.7)
    ax1.axvline(idx_ss_split, lw=1, color='black', alpha=0.7)
    ax1.set_title("ID %s signals before aligning stance sides" % ids_sel[idx_id], fontsize=10)

    # If colour coding by force platform:
    # plt.savefig('./Figures/ForceID-B_%s_fps_ID%s.pdf' % (fib_subset_to_process, ids_sel[idx_id]), dpi=600)
    # If colour coding by stance side:
    plt.savefig('./Figures/ForceID-B_%s_ss_ID%s.pdf' % (fib_subset_to_process, ids_sel[idx_id]), dpi=600)
    # plt.show()
    plt.close()

# Stance side detection edge cases in ForceID-B_CFC (encompassing ForceID-B_RS):
# ID 025 - A few signals near the intersection point but stance side detection method still correct.
# ID 070 - One signal from right stance near the intersection point but stance side detection method still correct.
# ID 142 - One signal from each stance side near the intersection point but stance side detection method still correct.
# ID 209 - One signal from left stance near the intersection point but stance side detection method still correct.
# ID 211 - One signal from left stance near the intersection point but stance side detection method still correct.
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 7. VALIDATE PROCESSED SIGNALS FROM EACH PARTICIPANT VIA VISUAL INSPECTION - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 8. VALIDATE THE STANCE SIDE DETECTION HEURISTIC AGAINST GROUND-TRUTH STANCE SIDE ANNOTATIONS AND THEN
# # TRANSFORM THE FORCE PLATFORM COORDINATE SYSTEM TO A FOOT BIOMECHANICAL COORDINATE SYSTEM - START
# # -------------------------------------------------------------------------------------------------------------------
# Get ground-truth annotations from each foot for each force platform:
ss_true_fp1 = metadata_sel[:, [29, 30]]
ss_true_fp2 = metadata_sel[:, [33, 34]]
ss_true_fp3 = metadata_sel[:, [37, 38]]

# Create a 3D array comprising stance side annotation arrays for each force platform:
ss_true = np.array([ss_true_fp1, ss_true_fp2, ss_true_fp3])

# Get true stance side annotations for FPA and FPB (the first two force platforms with consecutive complete foot
# contacts).
ss_fpa_fpb_true = []
# indices_double_step = []

for i in range(fp_nos.shape[0]):

    # In developing this loop, we validated that there were no cases where a second foot contact was annotated by
    # confirming that the second annotation column was always 'N':
    # ss_true_both_feet = ss_true[fp_nos[i] - 1, i]
    # if np.all(ss_true_both_feet[:, 1] != np.array(['N', 'N'])):
    #     indices_double_step.append(i)

    # Given that the above conditional was always False (i.e., zero cases of double foot contact), we can append only
    # the first stance side annotation column to omit the redundant dimension:
    ss_fpa_fpb_true.append(ss_true[fp_nos[i] - 1, i, 0])

ss_fpa_fpb_true = np.array(ss_fpa_fpb_true)

# Get indices of samples where stance side labels from the heuristic match/mismatch those from video annotation:
ss_matched = np.asarray(ss == ss_fpa_fpb_true).nonzero()
ss_mismatched = np.asarray(ss != ss_fpa_fpb_true).nonzero()

# Now that stance sides have been assigned and validated, we can transform the force platform coordinate system to a
# foot biomechanical coordinate system where x is medio-lateral (pointing medially), y is antero-posterior (pointing
# anterior) and z is vertical (pointing upward) relative to the COP. This requires flipping the x-axis for signals from
# left stance:
sigs_pro_l[:, 0] = sigs_pro_ss[:, 5] * -1 # Fx - medio-lateral GRF.
sigs_pro_l[:, 3] = sigs_pro_ss[:, 8] * -1 # Cx - medio-lateral COP.
# (Indexed sigs_pro_ss to avoid in-place indexed assignment of sigs_pro_l.)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 8. VALIDATE THE STANCE SIDE DETECTION HEURISTIC AGAINST GROUND-TRUTH STANCE SIDE ANNOTATIONS AND THEN
# # TRANSFORM THE FORCE PLATFORM COORDINATE SYSTEM TO A FOOT BIOMECHANICAL COORDINATE SYSTEM - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 9. SAVE DATA OBJECTS FOR EACH STANCE SIDE - START
# # -------------------------------------------------------------------------------------------------------------------
# Save the data objects for each stance side as numpy arrays:
if fib_subset_to_process == 'CFC':

    np.save('./Datasets/fi-b-cfc/Objects/sigs_pro_r.npy', sigs_pro_r)
    np.save('./Datasets/fi-b-cfc/Objects/sigs_pro_l.npy', sigs_pro_l)

else: # 'RS'

    np.save('./Datasets/fi-b-rs/Objects/sigs_pro_r.npy', sigs_pro_r)
    np.save('./Datasets/fi-b-rs/Objects/sigs_pro_l.npy', sigs_pro_l)

# Write the data subset to csv files for publication. Each file contains data for a given directional component and
# stance side. Define the first column of the data subset csv file to contain row headings for ID, session, trial, force
# platform and frame numbers (with the latter starting from 1):
row_headings = np.concatenate((np.array(['PARTICIPANT_ID', 'SESSION_NUMBER', 'TRIAL_NUMBER', 'FORCE_PLATFORM_NUMBER']),
                               np.arange(1, len_interp + 1)))[:, None]

# Specify the stance side categories/labels in the subset. The list only contains 'R' and 'L' because the subset
# contains complete foot contacts only. Note that the order of items matters for the loop below:
ss_categories = ['R', 'L']

for idx_ss, sigs_ss in enumerate([sigs_pro_r, sigs_pro_l]):

    for idx_channel in indices_channels:

        # To format data from each frame across rows and data from each sample across columns, combine the following:
        # - Transposed slice of the metadata array containing ID, session and trial numbers.
        # - Force platform number array for the stance side reshaped from (N,) to (1, N).
        # - Transposed data array for the channel/component.
        if idx_ss == 0:

            # Right stance phases:
            data_channel = np.concatenate((metadata_sel[:, 1:4].transpose(),
                                           fp_nos_r[None, :],
                                           sigs_ss[:, idx_channel].transpose()), axis=0)

        else:

            # Left stance phases:
            data_channel = np.concatenate((metadata_sel[:, 1:4].transpose(),
                                           fp_nos_l[None, :],
                                           sigs_ss[:, idx_channel].transpose()), axis=0)

        excel_grid = np.concatenate((row_headings, data_channel), axis=1)
        df = pd.DataFrame(excel_grid)

        if fib_subset_to_process == 'CFC':

            df.to_csv('./Datasets/fi-b-cfc/Spreadsheets/ForceID-B_%s_%s_%s_pro.csv' % (fib_subset_to_process,
                                                                                       channels[idx_channel],
                                                                                       ss_categories[idx_ss]),
                      header=False,
                      index=False)

        else: # 'RS'

            df.to_csv('./Datasets/fi-b-rs/Spreadsheets/ForceID-B_%s_%s_%s_pro.csv' % (fib_subset_to_process,
                                                                                      channels[idx_channel],
                                                                                      ss_categories[idx_ss]),
                      header=False,
                      index=False)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 9. SAVE DATA OBJECTS FOR EACH STANCE SIDE - END
# # -------------------------------------------------------------------------------------------------------------------
