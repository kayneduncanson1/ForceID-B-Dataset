import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import decimate
from scipy.interpolate import interp1d
from Utils import extract_base_metadata_fib
from PrePro import butterworth_lowpass, normalise_grf_by_body_weight
from Stats import extract_grf_features, aggregate_participant_features, participant_mass_map_from_fib_metadata,\
    bootstrap_difference_and_g, print_feature_sign_diagnostics

"""This script contributes to the technical validation by generating Figure 8/9 in the manuscript and performing
statistical analysis on discrete GRF features.

 Each figure contains three columns of sub-plots:
 1) Means and standard deviations of processed signals in four previous large-scale walking GRF datasets, referred
    herein as benchmark datasets (AIST, Gutenberg, GaitRec and ForceID-A). These signals are from complete foot
    contacts.
 2) Signals in ForceID-B_CFC processed as per the method demonstrated in data_processing.py. These signals are also
    from complete foot contacts.
 3) A subset of ForceID-B containing signals from partial foot contacts for two random participants.

Figure 8 is in the absolute force scale whereas Figure 9 is in the body-weight-normalised force scale."""

# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. INITIALISE PARAMETERS - START
# # -------------------------------------------------------------------------------------------------------------------
# When ``supp_analysis`` is False, this script generates Figure 8. When `supp_analysis`` is True, this script generates
# Figure 9 and performs the statistical analysis. Namely, it extracts five pre-specified GRF features, summarises paired
# right/left stance phases within trials, averages all available trial summaries within participant and outputs
# participant-level mean differences and Hedges' g with bootstrap 95% confidence intervals:
supp_analysis = False

# Parameters used by the body-weight-normalised feature analysis:
gravity = 9.80665  # m/s^2
n_bootstrap = 10000
bootstrap_seed = 42

# Pre-specified windows on the time normalised stance phase. The time axis is 0-100% stance. Fz is positive upwards and
# Fy is positive forwards in the foot biomechanical coordinate system used by data_processing.py:
feature_specs = [
    {"key": "Fz_first_peak", "label": "Fz first peak (5-35% stance)",
     "channel": 2, "start_pct": 5.0, "end_pct": 35.0, "operation": "max"},
    {"key": "Fz_midstance_trough", "label": "Fz midstance trough (35-65% stance)",
     "channel": 2, "start_pct": 35.0, "end_pct": 65.0, "operation": "min"},
    {"key": "Fz_second_peak", "label": "Fz second peak (65-95% stance)",
     "channel": 2, "start_pct": 65.0, "end_pct": 95.0, "operation": "max"},
    {"key": "Fy_braking_minimum", "label": "Fy braking minimum (0-50% stance)",
     "channel": 1, "start_pct": 0.0, "end_pct": 50.0, "operation": "min"},
    {"key": "Fy_propulsive_maximum", "label": "Fy propulsive maximum (50-100% stance)",
     "channel": 1, "start_pct": 50.0, "end_pct": 100.0, "operation": "max"},
]
feature_keys = [spec["key"] for spec in feature_specs]

# The code for part 3 is specifically set up for participant IDs 36 and 151 that were randomly sampled from the
# dataset. It may need to be adapted (e.g., generalised) to work on other participants:
ids_rand = [36, 151]

# The figure is set up to plot each directional component of the 3D ground reaction force (GRF) and 2D center of
# pressure (COP) as these are the components included in the benchmark datasets:
channels = ['Fx', 'Fy', 'Fz', 'Cx', 'Cy']

# Vertical GRF (Fz) threshold value for segmenting the portion of each signal pertaining to the stance phase (units:
# newton N):
fz_thresh = 50

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

# In part 3, signals from partial foot contacts with an approximate foot contact period greater than the length below in
# terms of number of frames will be selected for processing:
len_fc_min = 200

# Number of frames over which to interpolate the signals for the purpose of time normalisation:
len_interp = 100

# Global parameters for plot formatting:
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('tableau-colorblind10')
mpl.rcParams['mathtext.default'] = 'regular'
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. INITIALISE PARAMETERS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. LOAD DATA AND METADATA FROM THE BENCHMARK ('BM') DATASETS - START
# # -------------------------------------------------------------------------------------------------------------------
names_bm = ['AIST', 'Gutenberg', 'GaitRec (healthy)', 'ForceID-A (private)']
dirs_bm = ['ai-all', 'gb-all', 'gr-all', 'fi-all'] # Nomenclature from ForceID-Study-2 repo [2].
sigs_bm_r = []
sigs_bm_l = []
sigs_bm_stance = []
masses_bm = []
labels_bm = []

for dataset in dirs_bm:

    # Each row in the right/left arrays represents the two stance phases from the same trial, i.e., the signals can be
    # paired before trial averaging.
    sigs_r = np.asarray(np.load('./Datasets/%s/Objects/sigs_pro_r.npy' % dataset, allow_pickle=True), dtype=float)
    sigs_l = np.asarray(np.load('./Datasets/%s/Objects/sigs_pro_l.npy' % dataset, allow_pickle=True), dtype=float)
    # masses.npy and labels.npy are trial-level arrays aligned to the paired signal arrays.
    masses = np.asarray(np.load('./Datasets/%s/Objects/masses.npy' % dataset, allow_pickle=True), dtype=float)
    labels = np.asarray(np.load('./Datasets/%s/Objects/labels.npy' % dataset, allow_pickle=True))

    sigs_bm_r.append(sigs_r)
    sigs_bm_l.append(sigs_l)
    sigs_bm_stance.append(np.concatenate((sigs_r, sigs_l)))
    masses_bm.append(masses)
    labels_bm.append(labels)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. LOAD DATA AND METADATA FROM THE BENCHMARK ('BM') DATASETS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. LOAD DATA AND METADATA FROM FORCEID-B_CFC - START
# # -------------------------------------------------------------------------------------------------------------------
# Paired left/right stance signal and metadata arrays were defined and saved in data_processing.py.
sigs_fib_cfc_r = np.load('./Datasets/fi-b-cfc/Objects/sigs_pro_r.npy', allow_pickle=True)
sigs_fib_cfc_l = np.load('./Datasets/fi-b-cfc/Objects/sigs_pro_l.npy', allow_pickle=True)
metadata_fib_cfc = np.load('./Datasets/fi-b-cfc/Objects/metadata_cfc.npy', allow_pickle=True)
id_labels_fib_cfc, ids_fib_cfc, counts_trials_by_id_fib_cfc, indices_where_id_changes_fib_cfc,\
    ids_sessions_trials_fib_cfc, metadata_by_id_fib_cfc = extract_base_metadata_fib(metadata_fib_cfc)

# Concatenate signals from each stance side:
sigs_fib_cfc_stance = np.concatenate((sigs_fib_cfc_r, sigs_fib_cfc_l))
# Generate an ID label array that maps to the stance phase signal array:
id_labels_fib_cfc_stance = np.concatenate((id_labels_fib_cfc, id_labels_fib_cfc))
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. LOAD DATA AND METADATA FROM FORCEID-B_CFC - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. GET METADATA FOR TWO RANDOM PARTICIPANTS IN FORCEID-B (DENOTED 'RAND' SUBSET) - START
# # -------------------------------------------------------------------------------------------------------------------
# First get metadata for ForceID-B at large:
metadata_fib = pd.read_csv('./Datasets/fi-b-all/Spreadsheets/Metadata.csv', usecols=np.arange(40)).values
id_labels_fib, ids_fib, counts_trials_by_id_fib, indices_where_id_changes_fib, ids_sessions_trials_fib,\
    metadata_by_id_fib = extract_base_metadata_fib(metadata_fib)

# Get the subset of metadata for the two random participants:
indices_id_labels_rand = np.array([idx for idx, label in enumerate(id_labels_fib) if label in ids_rand])
metadata_rand = metadata_fib[indices_id_labels_rand]
# _ is used in place of ids_rand below because ids_rand is already defined above:
id_labels_rand, _, counts_trials_by_id_rand, indices_where_id_changes_rand, ids_sessions_trials_rand,\
    metadata_by_id_rand = extract_base_metadata_fib(metadata_rand)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. GET METADATA FOR TWO RANDOM PARTICIPANTS IN FORCEID-B (DENOTED 'RAND' SUBSET) - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 5. GET DATA FOR THE RAND SUBSET - START
# # -------------------------------------------------------------------------------------------------------------------
# Read the specified channels for the rand subset:
sigs_rand_raw_fp1 = [pd.read_csv('./Datasets/fi-b-all/Spreadsheets/ForceID-B_%s_FP1_raw.csv' % channel,
                                 usecols=np.arange(1, 1363),
                                 skiprows=2).values.transpose()[indices_id_labels_rand]
                     for channel in channels]
sigs_rand_raw_fp2 = [pd.read_csv('./Datasets/fi-b-all/Spreadsheets/ForceID-B_%s_FP2_raw.csv' % channel,
                                 usecols=np.arange(1, 1363),
                                 skiprows=2).values.transpose()[indices_id_labels_rand]
                     for channel in channels]
sigs_rand_raw_fp3 = [pd.read_csv('./Datasets/fi-b-all/Spreadsheets/ForceID-B_%s_FP3_raw.csv' % channel,
                                 usecols=np.arange(1, 1363),
                                 skiprows=2).values.transpose()[indices_id_labels_rand]
                     for channel in channels]

sigs_rand_raw_fp1 = np.array(sigs_rand_raw_fp1).transpose((1, 0, 2))
sigs_rand_raw_fp2 = np.array(sigs_rand_raw_fp2).transpose((1, 0, 2))
sigs_rand_raw_fp3 = np.array(sigs_rand_raw_fp3).transpose((1, 0, 2))
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 5. GET DATA FOR THE RAND SUBSET - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 6. FOR EACH TRIAL IN THE RAND SUBSET, SELECT SIGNALS FROM PARTIAL FOOT CONTACTS (DENOTED 'PFC' SUBSET) AND
# # VISUALISE THEIR RAW VERTICAL GRF MEASUREMENTS - START
# # -------------------------------------------------------------------------------------------------------------------
# Get the subset of signals from partial foot contacts (denoted 'PFC') and associated metadata:
sigs_pfc_raw = []
metadata_pfc = []

for i in range(metadata_rand.shape[0]):

    # In the current implementation with the rand subset containing IDs 36 and 151, there are no instances of 'B, F' in
    # the foot contact location field reflecting two partial foot contacts. If including different participants in the
    # rand subset, the code may need to be adapted to consider instances of multiple foot contacts separately (e.g., if
    # metadata_rand[i, ...] not in ['C', 'B, F']):
    if metadata_rand[i, 27] != 'C':

        sigs_pfc_raw.append(sigs_rand_raw_fp1[i])
        metadata_pfc.append(metadata_rand[i])

    if metadata_rand[i, 31] != 'C':

        sigs_pfc_raw.append(sigs_rand_raw_fp2[i])
        metadata_pfc.append(metadata_rand[i])

    if metadata_rand[i, 35] != 'C':

        sigs_pfc_raw.append(sigs_rand_raw_fp3[i])
        metadata_pfc.append(metadata_rand[i])

sigs_pfc_raw = np.array(sigs_pfc_raw)
metadata_pfc = np.array(metadata_pfc)

# Set unneeded variables to None to save memory and simplify workspace:
indices_id_labels_rand = None
metadata_rand = None
id_labels_rand = None
_ = None
counts_trials_by_id_rand = None
indices_where_id_changes_rand = None
ids_sessions_trials_rand = None
metadata_by_id_rand = None
sigs_rand_raw_fp1 = None
sigs_rand_raw_fp2 = None
sigs_rand_raw_fp3 = None

# Plot raw vertical GRF (Fz) measurements from all participants in a single figure to inform signal processing:
for sig in sigs_pfc_raw:

    plt.plot(sig[2] * -1, color='black')

plt.savefig('./Figures/ForceID-B_PFC_Fz_raw.pdf', dpi=1200)
# plt.show()
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 6. FOR EACH TRIAL IN THE RAND SUBSET, SELECT SIGNALS FROM PARTIAL FOOT CONTACTS (DENOTED 'PFC' SUBSET) AND
# # VISUALISE THEIR RAW VERTICAL GRF MEASUREMENTS - END
# # -------------------------------------------------------------------------------------------------------------------

# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 7. SELECT SIGNALS IN THE PFC SUBSET WITH SUFFICIENT FOOT CONTACT DURATION (DENOTED 'VALID' SUBSET) AND THEN
# # PROCESS BY TRANSFORMING COORDINATE SYSTEM, SEGMENTING FOOT CONTACT PERIOD, DOWN-SAMPLING, FILTERING AND TIME
# # NORMALISING - START
# # -------------------------------------------------------------------------------------------------------------------
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
sigs_pfc_raw_copy = sigs_pfc_raw.copy()
sigs_pfc_raw[:, 0] = sigs_pfc_raw_copy[:, 1]
sigs_pfc_raw[:, 1] = sigs_pfc_raw_copy[:, 0]

# The third channel representing Fz contains vertical GRF measurements as intended but the axis points downward,
# into a given force platform. Multiply the values by -1 to flip the axis direction:
sigs_pfc_raw[:, 2] = sigs_pfc_raw[:, 2] * -1

# The x and y axes must also be swapped for COP measurements. Then the antero-posterior COP, Cy, must be multiplied
# by -1 because it was calculated using Fz from the original coordinate system with z pointing downward rather than
# upward:
sigs_pfc_raw[:, 3] = sigs_pfc_raw_copy[:, 4]
sigs_pfc_raw[:, 4] = sigs_pfc_raw_copy[:, 3] * -1

# Set the copied variable to None to save memory:
sigs_pfc_raw_copy = None

# For each signal, segment the foot contact period, down-sample, filter and time normalise:
count_fc_valid = 0 # Number of instances where an approximate foot contact period is >= len_fc_min.
count_fc_invalid = 0 # Number of instances where an approximate foot contact period is < len_fc_min.
counts_frames_to_fc_start_raw = []
counts_frames_from_fc_end_raw = []
lens_cop_dns = []
counts_frames_to_fc_start_filt = []
counts_frames_from_fc_end_filt = []
sigs_valid = []
metadata_valid = []

for idx, sig in enumerate(sigs_pfc_raw):

    indices_fc = np.asarray(sig[2] > fz_thresh).nonzero()[0]
    len_fc = indices_fc.shape[0]

    if len_fc >= len_fc_min:

        count_fc_valid += 1

        # Get the start and end indices for the approximate foot contact period based on the Fz threshold value:
        start_1 = indices_fc[0]
        end_1 = indices_fc[-1]

        # To check the margins in terms of number of frames on either side of the approximate foot contact period in all
        # raw signals, can uncomment the following two lines and comment out the rest of the loop:
        # counts_frames_to_fc_start_raw.append(start_1)
        # counts_frames_from_fc_end_raw.append(sig[2].shape[0] - end_1)

        # For the GRF, retain n_frames_stance_buffer_grf at each end:
        start_grf = start_1 - n_frames_stance_buffer_grf
        end_grf = end_1 + n_frames_stance_buffer_grf

        # Relative to the length of the approximate foot contact period in number of frames, remove a further
        # portion_to_exclude_cop from each end (rounded to the nearest frame):
        start_cop = start_1 + int(np.round(portion_to_exclude_cop * (end_1 - start_1)))
        end_cop = end_1 - int(np.round(portion_to_exclude_cop * (end_1 - start_1)))

        # Based on the start and end indices defined above, segment and then down-sample the GRF and COP:
        grf_dns = decimate(sig[:3, start_grf:end_grf],
                           q=int(sampling_rate / sampling_rate_dns),
                           axis=1)
        cop_dns = decimate(sig[3:, start_cop:end_cop],
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

        # To check the margins in terms of number of frames on either side of the approximate foot contact period in all
        # down-sampled and filtered signals, can uncomment the following two lines and comment out the rest of the loop.
        # Following our method for time normalisation, the margins must be at least 1 frame to enable interpolation
        # between the frames before and after the vertical GRF crossed the threshold:
        # counts_frames_to_fc_start_filt.append(start_2)
        # counts_frames_from_fc_end_filt.append(grf_filt.shape[1] - end_2)

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
        sigs_valid.append(np.concatenate((grf_interp, cop_zeroed), axis=0))
        metadata_valid.append(metadata_pfc[idx])

    else:

        count_fc_invalid += 1

percent_fc_invalid = (count_fc_invalid / sigs_pfc_raw.shape[0]) * 100

sigs_valid = np.array(sigs_valid)
metadata_valid = np.array(metadata_valid)
id_labels_valid = metadata_valid[:, 1]

np.save('./Datasets/fi-b-pfc-valid/Objects/sigs_valid.npy', sigs_valid)
np.save('./Datasets/fi-b-pfc-valid/Objects/metadata_valid.npy', metadata_valid)

# Set unneeded variables to None to save memory and simplify workspace:
sigs_pfc_raw = None
metadata_pfc = None
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 7. SELECT SIGNALS IN THE PFC SUBSET WITH SUFFICIENT FOOT CONTACT DURATION (DENOTED 'VALID' SUBSET) AND THEN
# # PROCESS BY TRANSFORMING COORDINATE SYSTEM, SEGMENTING FOOT CONTACT PERIOD, DOWN-SAMPLING, FILTERING AND TIME
# # NORMALISING - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 8. NORMALISE GRF SIGNALS BY BODY WEIGHT AND CALCULATE FEATURE EFFECT SIZES - START
# # -------------------------------------------------------------------------------------------------------------------
if supp_analysis:

    # Create a participant-level body mass map for ForceID-B. Given that Metadata.csv is trial-level, repeated records
    # within each session are averaged to collapse to individual session records. Then, session records are averaged
    # to obtain a participant mean. Use body mass WITHOUT carried objects (metadata col index 7):
    participant_mass_map = participant_mass_map_from_fib_metadata(metadata_fib)

    # Use the body mass map to get body masses for participants in ForceID-B_CFC and the 'valid' partial contact subset:
    masses_fib_cfc = np.asarray(
        [participant_mass_map.get(str(pid)) for pid in id_labels_fib_cfc], dtype=float
    )
    masses_valid = np.asarray(
        [participant_mass_map.get(str(pid)) for pid in id_labels_valid], dtype=float
    )

    # For the benchmark datasets, normalise GRFs to %BW then extract discrete features from Fy and Fz:
    sigs_norm_bm_r = []
    sigs_norm_bm_l = []
    # Re-define the list of stance phase signal arrays from Section 2 to contain body-weight-normalised signals:
    sigs_bm_stance = []
    participant_features_by_dataset = {}
    trial_features_by_dataset = {}
    trial_counts_by_dataset = {}

    for idx_ds in range(len(dirs_bm)):

        sigs_norm_r = normalise_grf_by_body_weight(sigs_bm_r[idx_ds], masses_bm[idx_ds], gravity)
        sigs_norm_l = normalise_grf_by_body_weight(sigs_bm_l[idx_ds], masses_bm[idx_ds], gravity)

        sigs_norm_bm_r.append(sigs_norm_r)
        sigs_norm_bm_l.append(sigs_norm_l)
        sigs_bm_stance.append(np.concatenate((sigs_norm_r, sigs_norm_l)))

        features_r = extract_grf_features(sigs_norm_r, feature_specs)
        features_l = extract_grf_features(sigs_norm_l, feature_specs)
        trial_features = (features_r + features_l) / 2.0
        participant_features = aggregate_participant_features(trial_features, labels_bm[idx_ds], feature_keys)

        participant_features_by_dataset[names_bm[idx_ds]] = participant_features
        trial_features_by_dataset[names_bm[idx_ds]] = trial_features
        trial_counts_by_dataset[names_bm[idx_ds]] = len(trial_features)

        print_feature_sign_diagnostics(names_bm[idx_ds], trial_features, feature_keys)

    # For ForceID-B_CFC and the 'valid' subset, normalise GRFs to %BW:
    sigs_norm_fib_cfc_r = normalise_grf_by_body_weight(sigs_fib_cfc_r, masses_fib_cfc, gravity)
    sigs_norm_fib_cfc_l = normalise_grf_by_body_weight(sigs_fib_cfc_l, masses_fib_cfc, gravity)
    # Re-define the ForceID-B_CFC stance phase signal array from Section 3 to contain body-weight-normalised signals:
    sigs_fib_cfc_stance = np.concatenate((sigs_norm_fib_cfc_r, sigs_norm_fib_cfc_l))
    # Re-define the 'valid' stance phase signal array from Section 7 to contain body-weight-normalised signals:
    sigs_valid = normalise_grf_by_body_weight(sigs_valid, masses_valid, gravity)

    # For ForceID-B_CFC, extract discrete features from Fy and Fz for comparison with each benchmark dataset:
    features_fib_cfc_r = extract_grf_features(sigs_norm_fib_cfc_r, feature_specs)
    features_fib_cfc_l = extract_grf_features(sigs_norm_fib_cfc_l, feature_specs)
    trial_features_fib_cfc = (features_fib_cfc_r + features_fib_cfc_l) / 2.0
    participant_features_fib_cfc = aggregate_participant_features(
        trial_features_fib_cfc,
        id_labels_fib_cfc,
        feature_keys,
    )

    trial_features_by_dataset['ForceID-B_CFC'] = trial_features_fib_cfc
    trial_counts_by_dataset['ForceID-B_CFC'] = len(trial_features_fib_cfc)

    print_feature_sign_diagnostics('ForceID-B_CFC', trial_features_fib_cfc, feature_keys)

    # Record results in a CSV:
    results = []
    rng = np.random.default_rng(bootstrap_seed)

    for name_bm in names_bm:

        benchmark_participants = participant_features_by_dataset[name_bm]

        for spec in feature_specs:

            x = participant_features_fib_cfc[spec['key']].to_numpy(dtype=float)
            y = benchmark_participants[spec['key']].to_numpy(dtype=float)

            # Use an independently seeded stream per comparison/feature, while
            # ensuring results are reproducible from the script parameters.
            comparison_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            diff, diff_ci, g, g_ci = bootstrap_difference_and_g(
                x, y, n_bootstrap, comparison_seed
            )

            results.append({
                'Comparison': f'ForceID-B_CFC minus {name_bm}',
                'Benchmark dataset': name_bm,
                'Feature': spec['label'],
                'Unit': '%BW',
                'n ForceID-B participants': len(participant_features_fib_cfc),
                'n benchmark participants': len(benchmark_participants),
                'n ForceID-B trials': trial_counts_by_dataset['ForceID-B_CFC'],
                'n benchmark trials': trial_counts_by_dataset[name_bm],
                'ForceID-B participant mean (%BW)': float(np.mean(x)),
                'ForceID-B participant SD (%BW)': float(np.std(x, ddof=1)),
                'Benchmark participant mean (%BW)': float(np.mean(y)),
                'Benchmark participant SD (%BW)': float(np.std(y, ddof=1)),
                'Mean difference (%BW; B - benchmark)': diff,
                'Mean difference 95% CI lower (%BW)': float(diff_ci[0]),
                'Mean difference 95% CI upper (%BW)': float(diff_ci[1]),
                "Hedges' g (B - benchmark)": g,
                "Hedges' g 95% CI lower": float(g_ci[0]),
                "Hedges' g 95% CI upper": float(g_ci[1]),
            })

    results_df = pd.DataFrame(results)
    output_table = './Results/fig9_feature_effect_sizes.csv'
    results_df.to_csv(output_table, index=False, float_format='%.5f')

    print(f"Saved participant-level feature comparisons to {output_table}")

    n_fib_sessions = len(set(zip(metadata_fib_cfc[:, 1].astype(str), metadata_fib_cfc[:, 2].astype(str))))

    print("Feature analysis counts:")
    print(f"  ForceID-B_CFC: {len(trial_features_fib_cfc)} bilateral trials, "
          f"{len(participant_features_fib_cfc)} participants and {n_fib_sessions} sessions.")
    for name_bm in names_bm:
        print(f"  {name_bm}: {len(trial_features_by_dataset[name_bm])} bilateral trials and "
              f"{len(participant_features_by_dataset[name_bm])} participants.")
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 8. NORMALISE GRF SIGNALS BY BODY WEIGHT AND CALCULATE FEATURE EFFECT SIZES - END
# # -------------------------------------------------------------------------------------------------------------------

# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 9. GENERATE THE FIGURE ACROSS THREE COLUMNS OF SUBPLOTS:
# # 1) Mean and standard deviation of processed signals in each benchmark dataset.
# # 2) Processed signals in ForceID-B_CFC.
# # 3) Processed signals in the 'valid' subset (with partial foot contacts from two random participants in ForceID-B).
# # START
# # -------------------------------------------------------------------------------------------------------------------
fig = plt.figure(constrained_layout=True, figsize=(7.0, 7.0))
n_cols = 3
n_rows = len(channels) # Plot each directional component/channel in a separate row.
gs = fig.add_gridspec(n_rows, n_cols)

# Set subplot axes parameters and return the axes:
axs = []

for idx_col in range(n_cols):

    for idx_row in range(n_rows):

        ax = fig.add_subplot(gs[idx_row, idx_col])
        ax.tick_params(labelsize=10)
        ax.set_xlim(0, 100) # Percent foot contact duration.
        ax.set_xticks(np.arange(0, 101, 20))

        if idx_row == 0:

            ax.set_xticklabels([])

            # Force axis limits are kept in the original absolute force scale in Figure 8 mode, and converted to %BW in
            # Figure 9 (supplementary analysis) mode:
            if supp_analysis:

                ax.set_ylim(-21, 21)
                ax.set_yticks(np.arange(-21, 22, 7))

            else:

                ax.set_ylim(-150, 150)
                ax.set_yticks(np.arange(-150, 151, 50))

            if idx_col == 0:

                ax.set_ylabel('$F_x$ (% BW)' if supp_analysis else '$F_x$ (N)', size=11, style='italic')
                ax.set_title("Benchmark datasets\n(complete foot contacts)", size=11)

            elif idx_col == 1:

                ax.set_title("ForceID-B\n(complete foot contacts)", size=11)

            else:

                ax.set_title("ForceID-B\n(partial foot contacts)", size=11)

        elif idx_row == 1:

            ax.set_xticklabels([])

            if supp_analysis:

                ax.set_ylim(-45, 45)
                ax.set_yticks(np.arange(-45, 46, 15))

            else:

                ax.set_ylim(-375, 375)
                ax.set_yticks(np.arange(-375, 376, 125))

            if idx_col == 0:

                ax.set_ylabel('$F_y$ (% BW)' if supp_analysis else '$F_y$ (N)', size=11, style='italic')

        elif idx_row == 2:

            ax.set_xticklabels([])

            if supp_analysis:

                ax.set_ylim(0, 180)
                ax.set_yticks(np.arange(0, 181, 30))

            else:

                ax.set_ylim(0, 1500)
                ax.set_yticks(np.arange(0, 1501, 250))

            if idx_col == 0:

                ax.set_ylabel('$F_z$ (% BW)' if supp_analysis else '$F_z$ (N)', size=11, style='italic')

        elif idx_row == 3:

            ax.set_xticklabels([])
            ax.set_ylim(-0.09, 0.09)
            ax.set_yticks(np.arange(-0.09, 0.091, 0.03))

            if idx_col == 0:

                ax.set_ylabel('$C_x$ (m)', size=11, style='italic')

        else:

            ax.set_ylim(-0.06, 0.30)
            ax.set_yticks(np.arange(-0.06, 0.31, 0.06))

            if idx_col == 0:

                ax.set_ylabel('$C_y$ (m)', size=11, style='italic')

        if idx_col != 0:

            ax.set_yticklabels([])

        axs.append(ax)

colors = ['#C85200', '#5F9ED1', '#ABABAB', '#595959']

# 1) Plot the mean and standard deviation of processed signals in each benchmark dataset:
for ds in range(len(sigs_bm_stance)):

    for c in range(len(channels)):

        mean = np.mean(sigs_bm_stance[ds][:, c], axis=0)
        std = np.std(sigs_bm_stance[ds][:, c], axis=0)

        # Label the first subplot:
        if c == 0:

            axs[c].plot(mean, lw=1, alpha=1, c=colors[ds], label=names_bm[ds])

        else:

            axs[c].plot(mean, lw=1, alpha=1, c=colors[ds])

        axs[c].plot(mean - std, lw=1, alpha=1, c=colors[ds], ls='--')
        axs[c].plot(mean + std, lw=1, alpha=1, c=colors[ds], ls='--')

# 2) Plot processed signals in ForceID-B_CFC. This subsection is broken down into separate loops so that signals from
#    participants not in the rand/valid subset (colour grey) are plotted before signals from each of the two
#    participants in the rand/valid subset (overlaid, colour blue/orange). There are three loops with one for each
#    subgroup, meaning that this section is hard-coded on the assumption that the 'rand' subset (and thus 'valid'
#    subset) contains two participants. (We acknowledge the repetition - this section could be improved.):
count = 0
color = 'lightgray'
alpha = 0.8
plot_label = 'Others'

for idx, sig in enumerate(sigs_fib_cfc_stance):

    if id_labels_fib_cfc_stance[idx] not in ids_rand:

        count += 1

        if count == 1:

            axs[5].plot(sig[0], lw=0.5, color=color, alpha=alpha, label=plot_label)

        else:

            axs[5].plot(sig[0], lw=0.5, color=color, alpha=alpha)

        # Plot the remaining directional components/channels:
        axs[6].plot(sig[1], lw=0.5, color=color, alpha=alpha)
        axs[7].plot(sig[2], lw=0.5, color=color, alpha=alpha)
        axs[8].plot(sig[3], lw=0.5, color=color, alpha=alpha)
        axs[9].plot(sig[4], lw=0.5, color=color, alpha=alpha)

count = 0
color = '#006BA4'
alpha = 1
plot_label = 'ID %s' % ids_rand[0]

for idx, sig in enumerate(sigs_fib_cfc_stance):

    if id_labels_fib_cfc_stance[idx] == ids_rand[0]:  # ID 36 in the current implementation.

        count += 1

        if count == 1:

            axs[5].plot(sig[0], lw=0.5, color=color, alpha=alpha, label=plot_label)

        else:

            axs[5].plot(sig[0], lw=0.5, color=color, alpha=alpha)

        # Plot the remaining directional components/channels:
        axs[6].plot(sig[1], lw=0.5, color=color, alpha=alpha)
        axs[7].plot(sig[2], lw=0.5, color=color, alpha=alpha)
        axs[8].plot(sig[3], lw=0.5, color=color, alpha=alpha)
        axs[9].plot(sig[4], lw=0.5, color=color, alpha=alpha)

count = 0
color = '#FF800E'
alpha = 1
plot_label = 'ID %s' % ids_rand[1]

for idx, sig in enumerate(sigs_fib_cfc_stance):

    if id_labels_fib_cfc_stance[idx] == ids_rand[1]:  # ID 151 in the current implementation.

        count += 1

        if count == 1:

            axs[5].plot(sig[0], lw=0.5, color=color, alpha=alpha, label=plot_label)

        else:

            axs[5].plot(sig[0], lw=0.5, color=color, alpha=alpha)

        # Plot the remaining directional components/channels:
        axs[6].plot(sig[1], lw=0.5, color=color, alpha=alpha)
        axs[7].plot(sig[2], lw=0.5, color=color, alpha=alpha)
        axs[8].plot(sig[3], lw=0.5, color=color, alpha=alpha)
        axs[9].plot(sig[4], lw=0.5, color=color, alpha=alpha)

# Also plot the mean and standard deviation of each directional component/channel:
for c in range(len(channels)):

    mean = np.mean(sigs_fib_cfc_stance[:, c], axis=0)
    std = np.std(sigs_fib_cfc_stance[:, c], axis=0)

    # Index axes 5-9 inclusive to plot in the middle column:
    axs[c + 5].plot(mean, lw=1, alpha=1, c='black')
    axs[c + 5].plot(mean - std, lw=1, alpha=1, c='black', ls='--')
    axs[c + 5].plot(mean + std, lw=1, alpha=1, c='black', ls='--')

# 3) Plot processed signals in the 'valid' subset. This loop assumes that the subset contains two IDs:
for idx, sig in enumerate(sigs_valid):

    for c in range(len(channels)):

        if id_labels_valid[idx] == ids_rand[0]: # ID 36 in the current implementation.

            axs[c + 10].plot(sig[c], lw=0.5, alpha=1, color='#006BA4')

        else:  # ID 151 in the current implementation.

            axs[c + 10].plot(sig[c], lw=0.5, alpha=1, color='#FF800E')

# The segment below creates a legend that was post-edited in Adobe Illustrator for better positioning:
# handles_ax0, labels_ax0 = axs[0].get_legend_handles_labels()
# handles_ax5, labels_ax5 = axs[5].get_legend_handles_labels()
# handles = np.concatenate((handles_ax0, handles_ax5))
# labels = np.concatenate((labels_ax0, labels_ax5))
# axs[0].legend(handles,
#               labels,
#               prop={'size': 8},
#               loc='lower right',
#               ncol=2,
#               borderpad=0.3,
#               labelspacing=0.2,
#               handlelength=1.5,
#               handletextpad=0.6,
#               columnspacing=0.8)

fig.supxlabel("Percent foot contact duration", size=11)
figure_path = './Figures/fig 9 (sigs BW).pdf' if supp_analysis else './Figures/fig 8 (sigs).pdf'
plt.savefig(figure_path, dpi=1200)
print(f"Saved figure to {figure_path}")
# plt.show()
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 9. GENERATE THE FIGURE ACROSS THREE COLUMNS OF SUBPLOTS:
# # 1) Mean and standard deviation of processed signals in each benchmark dataset.
# # 2) Processed signals in ForceID-B_CFC.
# # 3) Processed signals in the 'valid' subset (with partial foot contacts from two random participants in ForceID-B).
# # END
# # -------------------------------------------------------------------------------------------------------------------
