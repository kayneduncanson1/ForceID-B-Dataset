import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

"""This script analyses the distribution of foot contacts on a given force platform to present in Figure 7 of the
manuscript (foot contact heatmap). It also generates an accessory figure comprising bar charts showing the distributions
of foot contacts on each force platform independently."""

# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. GET FOOT CONTACT LOCATION METADATA - START
# # -------------------------------------------------------------------------------------------------------------------
metadata = pd.read_csv('./Datasets/fi-b-all/Spreadsheets/Metadata.csv', usecols=np.arange(40)).values
fc_locs_fp1 = metadata[:, 27]
fc_locs_fp2 = metadata[:, 31]
fc_locs_fp3 = metadata[:, 35]

# If the entry for one force platform field was nan, then the entry for the other force platform fields was also nan.
# In other words, the same non-nan indices apply to all force platform fields:
fc_locs_nan_mask = ~pd.isna(fc_locs_fp1)
fc_locs_fp1_not_nan = fc_locs_fp1[fc_locs_nan_mask]
fc_locs_fp2_not_nan = fc_locs_fp2[fc_locs_nan_mask]
fc_locs_fp3_not_nan = fc_locs_fp3[fc_locs_nan_mask]
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. GET FOOT CONTACT LOCATION METADATA - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. GET THE DISTRIBUTION OF FOOT CONTACTS ACROSS FORCE PLATFORMS - START
# # -------------------------------------------------------------------------------------------------------------------
fc_locs = np.concatenate((fc_locs_fp1_not_nan, fc_locs_fp2_not_nan, fc_locs_fp3_not_nan))
categories, counts_by_category = np.unique(fc_locs, return_counts=True)
indices_sort_counts_by_category = np.argsort(counts_by_category)[::-1]
counts_by_category_sorted = counts_by_category[indices_sort_counts_by_category]
categories_sorted = categories[indices_sort_counts_by_category]

# Get the percentage of partial foot contacts in the dataset to report in the manuscript text. Note that 'B, F' and
# 'L, F' are double foot contacts where each foot lands partially, so the counts for these categories were doubled:
percent_pfc = ((1188 + 1142 + 212 + 3) / (1396 + 1188 + 1142 + 212 + 3)) * 100

# We show the counts for the five most common foot contact categories in the heatmap figure:
counts_by_category_for_heatmap = counts_by_category_sorted[:5]
categories_for_heatmap = categories_sorted[:5]

# Create plasma colour map:
cmap = mpl.cm.get_cmap('plasma')

# Normalise the counts for foot contact categories relative to a total of 1500 counts:
counts_by_category_for_heatmap_norm = np.array([el / 1500
                                                for el in counts_by_category_for_heatmap])

# Convert the normalised counts to RGB codes according to the plasma colour map. These RGB codes were used to fill the
# foot illustrations in Adobe Illustrator:
rgbs = np.round(np.array([cmap(percent)[:3] for percent in counts_by_category_for_heatmap_norm]) * 255)

# Generate and save a plasma colour bar to include in the figure. (The figure was created in Adobe Illustrator using the
# RGB codes and colour bar.):
fig, ax = plt.subplots()
norm = plt.Normalize(0, 1500)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax, ticks=np.arange(0, 1501, 300))
cbar.ax.tick_params(labelsize=11)
plt.savefig('./Figures/fig 7 (foot strikes) cbar.pdf', dpi=1200)
# plt.show()
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. GET THE DISTRIBUTION OF FOOT CONTACTS ACROSS FORCE PLATFORMS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. (ACCESSORY) GET THE DISTRIBUTION OF FOOT CONTACTS ON EACH FORCE PLATFORM INDEPENDENTLY - START
# # -------------------------------------------------------------------------------------------------------------------
counts_by_category_to_plot = []
categories_to_plot = []

for fc_locs_fp in [fc_locs_fp1_not_nan, fc_locs_fp2_not_nan, fc_locs_fp3_not_nan]:

    counts_by_category_fp = []

    for category in categories:

        counts_by_category_fp.append(np.asarray(fc_locs_fp == category).nonzero()[0].shape[0])

    counts_by_category_fp = np.array(counts_by_category_fp)

    indices_sort_counts_by_category_fp = np.argsort(counts_by_category_fp)[::-1]
    counts_by_category_fp_sorted = counts_by_category_fp[indices_sort_counts_by_category_fp]
    categories_fp_sorted = categories[indices_sort_counts_by_category_fp]
    counts_by_category_to_plot.append(counts_by_category_fp_sorted[:5])
    categories_to_plot.append(categories_fp_sorted[:5])
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. (ACCESSORY) GET THE DISTRIBUTION OF FOOT CONTACTS ON EACH FORCE PLATFORM INDEPENDENTLY - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. (ACCESSORY) GENERATE A FIGURE WITH HORIZONTAL BAR CHARTS OF FOOT CONTACT DISTRIBUTIONS - START
# # -------------------------------------------------------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('tableau-colorblind10')
mpl.rcParams['mathtext.default'] = 'regular'

fig = plt.figure(constrained_layout=True, figsize=(7.0, 4.85))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# First, plot the distribution across all force platforms as a reference:
ax1_y_pos = np.arange(categories_for_heatmap.shape[0])
ax1.barh(ax1_y_pos, counts_by_category_for_heatmap, height=0.8)
ax1.set_yticks(ax1_y_pos, labels=categories_for_heatmap)
ax1.tick_params(labelsize=10)
ax1.invert_yaxis()
ax1.set_xlim(0, 1500)
ax1.set_xticks(np.arange(0, 1501, 300))
ax1.set_title("Across force platforms", size=11)

# Force platform 1 (first force platform in the walking direction):
ax2_y_pos = np.arange(categories_to_plot[0].shape[0])
ax2.barh(ax2_y_pos, counts_by_category_to_plot[0], height=0.8)
ax2.set_yticks(ax2_y_pos, labels=categories_to_plot[0])
ax2.tick_params(labelsize=10)
ax2.invert_yaxis()
ax2.set_xlim(0, 600)
ax2.set_xticks(np.arange(0, 601, 120))
ax2.set_title("Force platform 1", size=11)

# Force platform 2:
ax3_y_pos = np.arange(categories_to_plot[1].shape[0])
ax3.barh(ax3_y_pos, counts_by_category_to_plot[1], height=0.8)
ax3.set_yticks(ax3_y_pos, labels=categories_to_plot[1])
ax3.tick_params(labelsize=10)
ax3.invert_yaxis()
ax3.set_xlim(0, 600)
ax3.set_xticks(np.arange(0, 601, 120))
ax3.set_title("Force platform 2", size=11)

# Force platform 3:
ax4_y_pos = np.arange(categories_to_plot[2].shape[0])
ax4.barh(ax4_y_pos, counts_by_category_to_plot[2], height=0.8)
ax4.set_yticks(ax4_y_pos, labels=categories_to_plot[2])
ax4.tick_params(labelsize=10)
ax4.invert_yaxis()
ax4.set_xlim(0, 600)
ax4.set_xticks(np.arange(0, 601, 120))
ax4.set_title("Force platform 3", size=11)

fig.supxlabel("Count", size=11)
fig.supylabel("Foot contact location", size=11)
plt.savefig('./Figures/fig extra (foot strikes).pdf', dpi=1200)
# plt.show()
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. (ACCESSORY) GENERATE A FIGURE WITH HORIZONTAL BAR CHARTS OF FOOT CONTACT DISTRIBUTIONS - END
# # -------------------------------------------------------------------------------------------------------------------
