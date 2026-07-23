import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from Utils import extract_base_metadata_fib, load_full_metadata

"""This script generates Figure 4 in the manuscript showing the distributions of age, sex, mass and height in ForceID-B
compared to four previous large-scale walking GRF datasets (AIST, Gutenberg, GaitRec and ForceID-A), referred herein
as benchmark datasets."""

# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. INITIALISE PARAMETERS - START
# # -------------------------------------------------------------------------------------------------------------------
gaitrec_to_plot = 'gr-all' # Options: 'gr-all' (barefoot and shod trials) or 'gr-sho' (shod trials only).

plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('tableau-colorblind10')
mpl.rcParams['mathtext.default'] = 'regular'
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. INITIALISE PARAMETERS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. GET DEMOGRAPHIC METADATA FOR FORCEID-B - START
# # -------------------------------------------------------------------------------------------------------------------
# Get metadata for ForceID-B:
metadata_fib = pd.read_csv('./Datasets/fi-b-all/Spreadsheets/Metadata.csv', usecols=np.arange(40)).values
id_labels_fib, ids_fib, counts_trials_by_id_fib, indices_where_id_changes_fib, ids_sessions_trials_fib,\
    metadata_by_id_fib = extract_base_metadata_fib(metadata_fib)

# Get demographic attributes for each participant:
ages_by_id_fib = np.split(metadata_fib[:, 4], indices_where_id_changes_fib)
sexes_by_id_fib = np.split(metadata_fib[:, 5], indices_where_id_changes_fib)
masses_by_id_fib = np.split(metadata_fib[:, 7], indices_where_id_changes_fib)
heights_by_id_fib = np.split(metadata_fib[:, 8], indices_where_id_changes_fib)

# Get a representative value of each demographic attribute for each participant. Age, sex and height measurements are
# from session 1 because these demographics were only measured in session 1 (values were duplicated across subsequent
# session rows in the metadata spreadsheet). Body mass measurements are averaged. Height is nan for IDs 30 and 142,
# whereas mass is nan for ID 142 session 1:
idx_id_30 = np.asarray(ids_fib == 30).nonzero()[0][0]
idx_id_142 = np.asarray(ids_fib == 142).nonzero()[0][0]

ages_fib = []
sexes_fib = []
masses_fib = []
heights_fib = []

for idx_id in range(ids_fib.shape[0]):

    # Get entries for age, sex and mass. Include only non-nan mass values:
    ages_fib_subset = ages_by_id_fib[idx_id]
    sexes_fib_subset = sexes_by_id_fib[idx_id]
    masses_fib_subset = masses_by_id_fib[idx_id][~np.isnan(masses_by_id_fib[idx_id].astype(float))]

    # Get age and sex measurements from the first session via index 0, as well as the average body mass rounded to one
    # decimal place:
    age_fib = ages_fib_subset[0]
    sex_fib = sexes_fib_subset[0]
    mass_fib = np.round(np.mean(masses_fib_subset), decimals=1)

    ages_fib.append(age_fib)
    sexes_fib.append(sex_fib)
    masses_fib.append(mass_fib)

    # Given that IDs 30 and 142 have nan height, only get height for other participants:
    if idx_id not in [idx_id_30, idx_id_142]:

        # Get entries for height:
        heights_fib_subset = heights_by_id_fib[idx_id]

        # Get height measurements from the first session via index 0:
        height_fib = heights_fib_subset[0]

        heights_fib.append(height_fib)

ages_fib = np.array(ages_fib)
sexes_fib = np.array(sexes_fib)
masses_fib = np.array(masses_fib)
heights_fib = np.array(heights_fib)

# Convert sexes from string to binary:
indices_females_fib = np.asarray(sexes_fib == 'Female').nonzero()[0]
indices_males_fib = np.asarray(sexes_fib == 'Male').nonzero()[0]
sexes_fib[indices_females_fib] = 0
sexes_fib[indices_males_fib] = 1
sexes_fib = sexes_fib.astype(int)

# Get summary stats of demographics to report in the manuscript text:
age_fib_mean = np.mean(ages_fib)
mass_fib_mean = np.mean(masses_fib)
height_fib_mean = np.mean(heights_fib)

age_fib_std = np.std(ages_fib)
mass_fib_std = np.std(masses_fib)
height_fib_std = np.std(heights_fib)

count_females_fib = indices_females_fib.shape[0]
count_males_fib = indices_males_fib.shape[0]
percent_male_fib = (count_males_fib / sexes_fib.shape[0]) * 100
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. GET DEMOGRAPHIC METADATA FOR FORCEID-B - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. GET DEMOGRAPHIC METADATA FOR THE BENCHMARK DATASETS - START
# # -------------------------------------------------------------------------------------------------------------------
# This method differs to that applied above to get demographic metadata for ForceID-B due to minor differences in
# metadata spreadsheet format. This method is re-purposed from the ForceID-Study-2 repo by Duncanson et al. (2024):
ages_bm = [] # bm = benchmark.
sexes_bm = []
masses_bm = []
heights_bm = []

ages_bm_mean = []
masses_bm_mean = []
heights_bm_mean = []

ages_bm_std = []
masses_bm_std = []
heights_bm_std = []

counts_females_bm = []
counts_males_bm = []

for dataset in ['fi-all', 'gr-all', 'gb-all', 'ai-all']:

    # Get metadata:
    trial_names, id_labels, ids, counts_trials_by_id,\
        ages_temp, sexes_temp, masses_temp, heights_temp,\
        footwear, speeds = load_full_metadata(dataset)

    indices_where_id_changes = np.cumsum(counts_trials_by_id)[:-1]

    # Get demographic measurements from session 1 for all but the first participant:
    ages_by_id = []
    sexes_by_id = []
    masses_by_id = []
    heights_by_id = []

    for idx in indices_where_id_changes:

        # The measurement from session 1 can be found at the index where a participant's subset begins (i.e., idx):
        ages_by_id.append(ages_temp[idx])
        sexes_by_id.append(sexes_temp[idx])
        masses_by_id.append(masses_temp[idx])
        heights_by_id.append(heights_temp[idx])

    # Then include demographic measurements from session 1 for the first participant:
    ages = np.array([ages_temp[0]] + ages_by_id)
    sexes = np.array([sexes_temp[0]] + sexes_by_id)
    masses = np.array([masses_temp[0]] + masses_by_id)
    heights = np.array([heights_temp[0]] + heights_by_id)

    # Exclude nan values:
    # - Age is nan for 30 participants in Gutenberg and 1 participant in AIST.
    # - Sex is nan for 3 participants in Gutenberg.
    # - Height is nan for 24 participants in Gutenberg.
    ages = ages[~np.isnan(ages)]
    sexes = sexes[~np.isnan(sexes)]
    heights = heights[~np.isnan(heights)]

    ages_bm.append(ages)
    sexes_bm.append(sexes)
    masses_bm.append(masses)
    heights_bm.append(heights)

    ages_bm_mean.append(np.mean(ages))
    masses_bm_mean.append(np.mean(masses))
    heights_bm_mean.append(np.mean(heights))

    ages_bm_std.append(np.std(ages))
    masses_bm_std.append(np.std(masses))
    heights_bm_std.append(np.std(heights))

    count_females = np.asarray(sexes == 0).nonzero()[0].shape[0]
    count_males = np.asarray(sexes == 1).nonzero()[0].shape[0]

    counts_females_bm.append(count_females)
    counts_males_bm.append(count_males)

if gaitrec_to_plot == 'gr-sho':

    ages_gr_sho = []
    sexes_gr_sho = []
    masses_gr_sho = []
    heights_gr_sho = []

    for idx_id in range(ages_bm[1].shape[0]):

        # Exclude participants who did not walk shod (indices obtained from 'prepare_gaitrec.py' in the
        # ForceID-Study-2 repository):
        if idx_id not in [92, 95, 131]:

            ages_gr_sho.append(ages_bm[1][idx_id])
            sexes_gr_sho.append(sexes_bm[1][idx_id])
            masses_gr_sho.append(masses_bm[1][idx_id])
            heights_gr_sho.append(heights_bm[1][idx_id])

    # Overwrite original demographic arrays for the GaitRec dataset:
    ages_bm[1] = np.array(ages_gr_sho)
    sexes_bm[1] = np.array(sexes_gr_sho)
    masses_bm[1] = np.array(masses_gr_sho)
    heights_bm[1] = np.array(heights_gr_sho)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. GET DEMOGRAPHIC METADATA FOR THE BENCHMARK DATASETS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. GENERATE THE FIGURE SHOWING THE DISTRIBUTIONS OF DEMOGRAPHIC ATTRIBUTES - START
# # -------------------------------------------------------------------------------------------------------------------
fig = plt.figure(constrained_layout=True, figsize=(7.0, 4.85))
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

color_fib = '#C85200'

# Plot age distributions:
sns.kdeplot(ages_bm[3], ax=ax1, label='AIST')
sns.kdeplot(ages_bm[2], ax=ax1, label='Gutenberg')
sns.kdeplot(ages_bm[1], ax=ax1, label='GaitRec')
sns.kdeplot(ages_bm[0], ax=ax1, label='ForceID-A')
sns.kdeplot(ages_fib, ax=ax1, label='ForceID-B', color=color_fib)

# Plot height distributions:
sns.kdeplot(heights_bm[3], ax=ax2, label='AIST')
sns.kdeplot(heights_bm[2], ax=ax2, label='Gutenberg')
sns.kdeplot(heights_bm[1], ax=ax2, label='GaitRec')
sns.kdeplot(heights_bm[0], ax=ax2, label='ForceID-A')
sns.kdeplot(heights_fib, ax=ax2, label='ForceID-B', color=color_fib)

# Plot mass distributions:
sns.kdeplot(masses_bm[3], ax=ax3, label='AIST')
sns.kdeplot(masses_bm[2], ax=ax3, label='Gutenberg')
sns.kdeplot(masses_bm[1], ax=ax3, label='GaitRec')
sns.kdeplot(masses_bm[0], ax=ax3, label='ForceID-A')
sns.kdeplot(masses_fib, ax=ax3, label='ForceID-B', color=color_fib)

# Plot sex distributions in a bar chart:
counts_female = [counts_females_bm[3],
                 counts_females_bm[2],
                 counts_females_bm[1],
                 counts_females_bm[0],
                 count_females_fib]
counts_male = [counts_males_bm[3],
               counts_males_bm[2],
               counts_males_bm[1],
               counts_males_bm[0],
               count_males_fib]
# Hard-coded that sex was nan (i.e., unknown) for 3 participants in Gutenberg:
counts_sex_unk = [0, 3, 0, 0, 0]
ds_labels = ['AI', 'GB', 'GR', 'FI-A', 'FI-B']
x = np.arange(len(ds_labels))  # the x-axis label locations
width = 0.15  # the width of the bars
rects1 = ax4.bar(x - width, counts_female, width, label='Female')
rects2 = ax4.bar(x, counts_male, width, label='Male')
rects3 = ax4.bar(x + width, counts_sex_unk, width, label='Unknown')

# Set axis parameters:
ax1.set_xlim(0, 100)
ax1.set_xticks(np.arange(0, 101, 20))

ax2.set_xlim(1.2, 2.2)
ax2.set_xticks(np.arange(1.2, 2.21, 0.2))

ax3.set_xlim(0, 170)
ax3.set_xticks(np.arange(0, 171, 34))

ax4.set_xticks(x, ds_labels)

ax1.set_ylim(0, 0.1)
ax1.set_yticks(np.arange(0, 0.11, 0.02))

ax2.set_ylim(0, 5)
ax2.set_yticks(np.arange(0, 6, 1))

ax3.set_ylim(0, 0.05)
ax3.set_yticks(np.arange(0, 0.06, 0.01))

ax4.set_ylim(0, 250)
ax4.set_yticks(np.arange(0, 251, 50))

ax1.set_xlabel("Age (y)", size=11, labelpad=5)
ax2.set_xlabel("Height (m)", size=11, labelpad=5)
ax3.set_xlabel("Mass (kg)", size=11, labelpad=5)
ax4.set_xlabel('Dataset', size=11, labelpad=5)

ax1.set_ylabel("Density", size=11, labelpad=5)
ax2.set_ylabel("Density", size=11, labelpad=5)
ax3.set_ylabel("Density", size=11, labelpad=5)
ax4.set_ylabel('Count', size=11, labelpad=0)

ax1.tick_params(labelsize=11)
ax2.tick_params(labelsize=11)
ax3.tick_params(labelsize=11)
ax4.tick_params(labelsize=10)

ax1.legend(prop={'size': 9})
ax4.legend(prop={'size': 9})

# Save and/or show the figure:
plt.savefig('./Figures/fig 4 (demographics).pdf', dpi=1200)
# plt.show()
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. GENERATE THE FIGURE SHOWING THE DISTRIBUTIONS OF DEMOGRAPHIC ATTRIBUTES - END
# # -------------------------------------------------------------------------------------------------------------------
