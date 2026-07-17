import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from Utils import extract_base_metadata_fib

"""This script generates the figure with horizontal bar charts showing the distributions of sessions, trials, objects,
object masses, footwear and clothing in ForceID-B."""

# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. GET THE DISTRIBUTIONS OF SESSIONS AND TRIALS - START
# # -------------------------------------------------------------------------------------------------------------------
# Get metadata:
metadata = pd.read_csv('./Datasets/fi-b-all/Spreadsheets/Metadata.csv', usecols=np.arange(40)).values
id_labels, ids, counts_trials_by_id, indices_where_id_changes, ids_sessions_trials, metadata_by_id =\
    extract_base_metadata_fib(metadata)

# Get the counts of sessions per participant and trials per session:
counts_sessions_by_id = []
counts_trials_by_session = []

for metadata_subset in metadata_by_id:

    unique_session_numbers, counts_trials_by_session_subset = np.unique(metadata_subset[:, 2], return_counts=True)
    counts_sessions_by_id.append(unique_session_numbers.shape[0])
    counts_trials_by_session.append(counts_trials_by_session_subset)

counts_sessions_by_id = np.array(counts_sessions_by_id)
counts_trials_by_session = np.concatenate(counts_trials_by_session)

# Get the indices of trials where the session changes for later use:
indices_where_session_changes = np.cumsum(counts_trials_by_session)[:-1]
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 1. GET THE DISTRIBUTIONS OF SESSIONS AND TRIALS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. GET THE DISTRIBUTION OF OBJECT MASS - START
# # -------------------------------------------------------------------------------------------------------------------
masses = metadata[:, 6:8]
masses_by_session = np.split(masses, indices_where_session_changes)

# Get mass measurements from sessions where the participant had at least one object and their mass was measured with vs
# without objects (i.e., both mass entries were non-nan):
masses_not_nan = []

for mass_subset in masses_by_session:

    # Check for nans in the first row representing the first trial because mass entries were copied across rows:
    if not pd.isna(mass_subset[0]).any():

        masses_not_nan.append(mass_subset[0])

masses_not_nan = np.array(masses_not_nan)

# Calculate the absolute mass of the object(s) as the difference between body mass with vs without the object(s):
masses_objects = masses_not_nan[:, 0] - masses_not_nan[:, 1]

# Calculate the relative mass of the object(s) compared to body mass as a percentage:
masses_objects_pc_bm = ((masses_not_nan[:, 0] - masses_not_nan[:, 1]) / masses_not_nan[:, 1]) * 100
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 2. GET THE DISTRIBUTION OF OBJECT MASS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. GET ANNOTATIONS AND ANNOTATION CATEGORIES (I.E., LABELS) FOR CLOTHING, OBJECTS AND FOOTWEAR - START
# # -------------------------------------------------------------------------------------------------------------------
# Get annotations:
clothing_top = metadata[:, 10] # top = top clothing items.
clothing_bot = metadata[:, 11] # bot = bottom clothing items.
clothing_ttb = metadata[:, 12] # ttb = top-to-bottom clothing items.
objects = metadata[:, 13:15]
footwear = metadata[:, 15]

# There are nans for clothing, so we want to get non-nan entries:
clothing_not_nan_top = clothing_top[~pd.isna(clothing_top)]
clothing_not_nan_bot = clothing_bot[~pd.isna(clothing_bot)]
clothing_not_nan_ttb = clothing_ttb[~pd.isna(clothing_ttb)]

# Specify categories:
categories_clothing_top = np.array(["T-shirt",
                                    "Polo shirt",
                                    "Dress shirt",
                                    "Singlet",
                                    "Blouse",
                                    "Sports bra",
                                    "Blazer",
                                    "Jacket",
                                    "Coat",
                                    "Sweater/Sweatshirt/Jumper/Cardigan/Hoodie",
                                    "Other"])
categories_clothing_bot = np.array(["Leggings/tights",
                                    "Jeans",
                                    "Dress pants",
                                    "Cargo pants",
                                    "Chinos",
                                    "Tracksuit/sweat pants",
                                    "Skirt - short",
                                    "Skirt - long",
                                    "Shorts",
                                    "Other"])
categories_clothing_ttb = np.array(["Romper",
                                    "Jumpsuit/playsuit",
                                    "Dress - short",
                                    "Dress - long",
                                    "Dungaree/overalls",
                                    "Other"])
categories_objects = np.array(['Backpack',
                               'Handbag',
                               'Purse',
                               'Bumbag',
                               'Fanny pack',
                               'Mobile phone',
                               'Shopping bag',
                               'Gym bag',
                               'Tote bag',
                               'Drawstring bag',
                               'Briefcase',
                               'Laptop',
                               'Book(s)',
                               'Water bottle',
                               'Other',
                               'No object'])
categories_footwear = np.array(["Athletic",
                                "Women's ankle boot (flat)",
                                "Women's ankle boot (heel)",
                                "Men's ankle boot (flat)",
                                "Men's ankle boot (heel)",
                                "Flat canvas (slip-on/laced)",
                                "Men's business",
                                "Women's business",
                                "Ballet flat",
                                "Sandal",
                                "Loafer",
                                "Rubber boot",
                                "Five finger",
                                "Steel capped boot",
                                "Flip-flops (thongs)",
                                "Other"])
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 3. GET ANNOTATIONS AND ANNOTATION CATEGORIES (I.E., LABELS) FOR CLOTHING, OBJECTS AND FOOTWEAR - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. ACROSS SESSIONS AND INDIVIDUAL TRIALS, GET THE NUMBER OF INSTANCES OF NO OBJECTS AND TWO OBJECTS, AS WELL
# # AS THE ANNOTATIONS OF INDIVIDUAL OBJECT CATEGORIES IN INSTANCES OF ONE OBJECT - START
# # -------------------------------------------------------------------------------------------------------------------
# There will be two object distribution bar charts. One will show the object distribution across trials and the other
# will show the object distribution across sessions. In both cases, instances of one object from each category will be
# compared to instances of no objects and two objects. First we analyse objects at the level of individual trials by
# using boolean masks to distinguish trials with no objects vs one object vs two objects:
no_objects_carried = np.asarray(objects[:, 0] == 'No object')
one_object_carried = np.asarray(objects[~no_objects_carried, 1] == 'No object')

# Then, get the counts of trials with no objects and two objects, as well as the annotations for trials with one object:
count_trials_no_objects = no_objects_carried.nonzero()[0].shape[0]
count_trials_two_objects = objects[~no_objects_carried][~one_object_carried].shape[0]
objects_by_trial_one_object = objects[~no_objects_carried][one_object_carried, 0]

# Now we analyse objects at the session level by getting the counts of sessions with no objects and two objects, as well
# as the annotations for sessions with one object:
objects_by_session = np.split(objects, indices_where_session_changes)

count_sessions_no_objects = 0 # Number of cases of no objects brought to a session, initialised to zero.
count_sessions_two_objects = 0 # Number of cases of two objects brought to a session, initialised to zero.
objects_by_session_one_object = [] # Cases of one object brought to a session.

for objects_session in objects_by_session:

    count_trials = 0

    for objects_trial in objects_session:

        if np.all(objects_trial == 'No object'):

            count_trials += 1

            if count_trials == objects_session.shape[0]:

                count_sessions_no_objects += 1

        elif objects_trial[0] != 'No object' and objects_trial[1] == 'No object':

            objects_by_session_one_object.append(objects_trial[0])

            break

        else:

            count_sessions_two_objects += 1

            break

objects_by_session_one_object = np.array(objects_by_session_one_object)
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 4. ACROSS SESSIONS AND INDIVIDUAL TRIALS, GET THE NUMBER OF INSTANCES OF NO OBJECTS AND TWO OBJECTS, AS WELL
# # AS THE ANNOTATIONS OF INDIVIDUAL OBJECT CATEGORIES IN INSTANCES OF ONE OBJECT - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 5. GET THE COUNTS OF EACH ANNOTATION CATEGORY FOR CLOTHING, FOOTWEAR AND OBJECTS - START
# # -------------------------------------------------------------------------------------------------------------------
counts_clothing_top = np.array([np.asarray(clothing_not_nan_top == category).nonzero()[0].shape[0]
                                for category in categories_clothing_top])
counts_clothing_bot = np.array([np.asarray(clothing_not_nan_bot == category).nonzero()[0].shape[0]
                                for category in categories_clothing_bot])
counts_clothing_ttb = np.array([np.asarray(clothing_not_nan_ttb == category).nonzero()[0].shape[0]
                                for category in categories_clothing_ttb])
counts_footwear = np.array([np.asarray(footwear == category).nonzero()[0].shape[0]
                            for category in categories_footwear])

# For objects, we don't need to consider the 'No object' category (at index -1) as this category was already considered
# in Section 4:
counts_trials_one_object = []
counts_sessions_one_object = []

for category in categories_objects[:-1]:

    counts_trials_one_object.append(np.asarray(objects_by_trial_one_object == category).nonzero()[0].shape[0])
    counts_sessions_one_object.append(np.asarray(objects_by_session_one_object == category).nonzero()[0].shape[0])

# Append the counts for no objects and two objects that were obtained in Section 4:
counts_trials_objects = np.array(counts_trials_one_object +
                                 [count_trials_no_objects] +
                                 [count_trials_two_objects])
counts_sessions_objects = np.array(counts_sessions_one_object +
                                   [count_sessions_no_objects] +
                                   [count_sessions_two_objects])

# The categories_objects variable containing the original set of object categories does not account for cases of two
# objects. Define a new variable with an extra category for cases of two objects:
categories_objects_extra = np.concatenate((categories_objects, np.array(["Two objects"])))
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 5. GET THE COUNTS OF EACH ANNOTATION CATEGORY FOR CLOTHING, FOOTWEAR AND OBJECTS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 6. SORT THE COUNTS AND CATEGORIES - START
# # -------------------------------------------------------------------------------------------------------------------
# Get the indices that would sort the counts for each category in descending order:
indices_sort_counts_clothing_top = np.argsort(counts_clothing_top)[::-1]
indices_sort_counts_clothing_bot = np.argsort(counts_clothing_bot)[::-1]
indices_sort_counts_clothing_ttb = np.argsort(counts_clothing_ttb)[::-1]
indices_sort_counts_footwear = np.argsort(counts_footwear)[::-1]
indices_sort_counts_trials_objects = np.argsort(counts_trials_objects)[::-1]
indices_sort_counts_sessions_objects = np.argsort(counts_sessions_objects)[::-1]

# Use the indices to sort the counts for each category:
counts_clothing_top_sorted = counts_clothing_top[indices_sort_counts_clothing_top]
counts_clothing_bot_sorted = counts_clothing_bot[indices_sort_counts_clothing_bot]
counts_clothing_ttb_sorted = counts_clothing_ttb[indices_sort_counts_clothing_ttb]
counts_footwear_sorted = counts_footwear[indices_sort_counts_footwear]
counts_trials_objects_sorted = counts_trials_objects[indices_sort_counts_trials_objects]
counts_sessions_objects_sorted = counts_sessions_objects[indices_sort_counts_sessions_objects]

# Use the indices to sort the categories. For objects, we use the categories array with the 'Two objects' category:
categories_clothing_top_sorted = categories_clothing_top[indices_sort_counts_clothing_top]
categories_clothing_bot_sorted = categories_clothing_bot[indices_sort_counts_clothing_bot]
categories_clothing_ttb_sorted = categories_clothing_ttb[indices_sort_counts_clothing_ttb]
categories_footwear_sorted = categories_footwear[indices_sort_counts_footwear]
categories_trials_objects_sorted = categories_objects_extra[indices_sort_counts_trials_objects]
categories_sessions_objects_sorted = categories_objects_extra[indices_sort_counts_sessions_objects]
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 6. SORT THE COUNTS AND CATEGORIES - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 7. SELECT A SUBSET OF THE MOST COMMON CATEGORIES AND COUNTS TO INCLUDE IN EACH BAR CHART - START
# # -------------------------------------------------------------------------------------------------------------------
# Bar charts for clothing top, clothing bottom and objects will include the five most common categories along with an
# 'Others' category to represent key information in a concise and space efficient manner:
categories_clothing_top_plot = np.concatenate((categories_clothing_top_sorted[:5],
                                               np.array(["Others"])))
categories_clothing_bot_plot = np.concatenate((categories_clothing_bot_sorted[:5],
                                               np.array(["Others"])))
categories_trials_objects_plot = np.concatenate((categories_trials_objects_sorted[:5],
                                                 np.array(["Others"])))
categories_sessions_objects_plot = np.concatenate((categories_sessions_objects_sorted[:5],
                                                   np.array(["Others"])))

counts_clothing_top_others = np.sum(counts_clothing_top_sorted[5:])
counts_clothing_bot_others = np.sum(counts_clothing_bot_sorted[5:])
counts_trials_objects_others = np.sum(counts_trials_objects_sorted[5:])
counts_sessions_objects_others = np.sum(counts_sessions_objects_sorted[5:])

counts_clothing_top_plot = np.concatenate((counts_clothing_top_sorted[:5],
                                           np.array([counts_clothing_top_others])))
counts_clothing_bot_plot = np.concatenate((counts_clothing_bot_sorted[:5],
                                           np.array([counts_clothing_bot_others])))
counts_trials_objects_plot = np.concatenate((counts_trials_objects_sorted[:5],
                                             np.array([counts_trials_objects_others])))
counts_sessions_objects_plot = np.concatenate((counts_sessions_objects_sorted[:5],
                                               np.array([counts_sessions_objects_others])))

# For top-to-bottom clothing, there were only three categories with counts above zero:
categories_clothing_ttb_plot = np.concatenate((categories_clothing_ttb_sorted[:3], np.array(["Others"])))
counts_clothing_ttb_others = np.sum(counts_clothing_ttb_sorted[3:])
counts_clothing_ttb_plot = np.concatenate((counts_clothing_ttb_sorted[:3], np.array([counts_clothing_ttb_others])))

# For footwear, there was originally an 'Other' category which was the third most common (index 2 in the sorted
# categories). Subsume this category into the 'Others' category:
categories_footwear_plot = np.concatenate((categories_footwear_sorted[[0, 1, 3, 4, 5]],
                                           np.array(["Others"])))
counts_footwear_others = np.sum(np.concatenate((np.array([counts_footwear_sorted[2]]),
                                                counts_footwear_sorted[6:])))
counts_footwear_plot = np.concatenate((counts_footwear_sorted[[0, 1, 3, 4, 5]],
                                       np.array([counts_footwear_others])))

# Shorten certain category labels so that they fit in the figure:
categories_clothing_top_plot[1] = "Sweater" # Originally "Sweater/Sweatshirt/Jumper/Cardigan/Hoodie"
categories_clothing_bot_plot[3] = "Sweat pants"  # Originally "Tracksuit/sweat pants"
categories_clothing_bot_plot[4] = "Tights"  # Originally "Leggings/tights"
categories_clothing_ttb_plot[2] = "Overalls"  # Originally "Dungaree/overalls"
categories_footwear_plot[1] = "Flat canvas" # Originally "Flat canvas (slip-on/laced)"
categories_footwear_plot[2] = "Boot (M)" # Originally "Men's ankle boot (flat)"
categories_footwear_plot[4] = "Boot (F)" # Originally Women's ankle boot (flat)"
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 7. SELECT A SUBSET OF THE MOST COMMON CATEGORIES AND COUNTS TO INCLUDE IN EACH BAR CHART - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 8. DEFINE BINS FOR NUMBER OF TRIALS PER PARTICIPANT AND OBJECT MASS - START
# # -------------------------------------------------------------------------------------------------------------------
# Number of trials per participant:
count_trials_0_to_5 = 0
count_trials_5_to_10 = 0
count_trials_10_to_15 = 0
count_trials_15_to_20 = 0

for count_trials in counts_trials_by_id:

    if count_trials <= 5:

        count_trials_0_to_5 += 1

    elif 5 < count_trials <= 10:

        count_trials_5_to_10 += 1

    elif 10 < count_trials <= 15:

        count_trials_10_to_15 += 1

    elif 15 < count_trials <= 20:

        count_trials_15_to_20 += 1

# Object mass (kg):
count_masses_objects_0_to_5 = 0
count_masses_objects_5_to_10 = 0
count_masses_objects_10_to_15 = 0
count_masses_objects_15_to_20 = 0

# Object mass (percent body mass):
count_masses_objects_pc_bm_0_to_5 = 0
count_masses_objects_pc_bm_5_to_10 = 0
count_masses_objects_pc_bm_10_to_15 = 0
count_masses_objects_pc_bm_15_to_20 = 0

for i in range(masses_objects.shape[0]):

    if masses_objects[i] <= 5:

        count_masses_objects_0_to_5 += 1

    elif 5 < masses_objects[i] <= 10:

        count_masses_objects_5_to_10 += 1

    elif 10 < masses_objects[i] <= 15:

        count_masses_objects_10_to_15 += 1

    elif 15 < masses_objects[i] <= 20:

        count_masses_objects_15_to_20 += 1

    if masses_objects_pc_bm[i] <= 5:

        count_masses_objects_pc_bm_0_to_5 += 1

    elif 5 < masses_objects_pc_bm[i] <= 10:

        count_masses_objects_pc_bm_5_to_10 += 1

    elif 10 < masses_objects_pc_bm[i] <= 15:

        count_masses_objects_pc_bm_10_to_15 += 1

    elif 15 < masses_objects_pc_bm[i] < 21: # Highest = 20.36%.

        count_masses_objects_pc_bm_15_to_20 += 1

counts_trials = np.array([count_trials_0_to_5,
                          count_trials_5_to_10,
                          count_trials_10_to_15,
                          count_trials_15_to_20])
counts_masses_objects = np.array([count_masses_objects_0_to_5,
                                  count_masses_objects_5_to_10,
                                  count_masses_objects_10_to_15,
                                  count_masses_objects_15_to_20])
counts_masses_objects_pc_bm = np.array([count_masses_objects_pc_bm_0_to_5,
                                        count_masses_objects_pc_bm_5_to_10,
                                        count_masses_objects_pc_bm_10_to_15,
                                        count_masses_objects_pc_bm_15_to_20])
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 8. DEFINE BINS FOR NUMBER OF TRIALS PER PARTICIPANT AND OBJECT MASS - END
# # -------------------------------------------------------------------------------------------------------------------


# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 9. GENERATE THE FIGURE - START
# # -------------------------------------------------------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('tableau-colorblind10')
mpl.rcParams['mathtext.default'] = 'regular'

fig = plt.figure(constrained_layout=True, figsize=(7.0, 8.5))
gs = fig.add_gridspec(5, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])
ax7 = fig.add_subplot(gs[3, 0])
ax8 = fig.add_subplot(gs[3, 1])
ax9 = fig.add_subplot(gs[4, 0])
ax10 = fig.add_subplot(gs[4, 1])

# Create the bar chart for number of sessions per participant:
ax1_y_pos = np.arange(5)
ax1.barh(ax1_y_pos, np.unique(counts_sessions_by_id, return_counts=True)[1], height=0.8)
ax1.set_yticks(ax1_y_pos, labels=np.arange(1, 6).astype(str))
ax1.tick_params(labelsize=9)
ax1.invert_yaxis()
ax1.set_ylabel("Number of sessions per ID", size=10, labelpad=5)
ax1.set_xlim(0, 100)
ax1.set_xticks(np.arange(0, 101, 20))

# Create the bar chart for number of trials per participant:
ax2_y_pos = np.arange(4)
ax2.barh(ax2_y_pos, counts_trials, height=0.8)
ax2.set_yticks(ax2_y_pos, labels=["0-5", "5-10", "10-15", "15-20"])
ax2.tick_params(labelsize=9)
ax2.invert_yaxis()
ax2.set_ylabel("Number of trials per ID", size=10, labelpad=5)
ax2.set_xlim(0, 100)
ax2.set_xticks(np.arange(0, 101, 20))

# Create the bar chart for number of instances of different objects being brought to sessions:
ax3_y_pos = np.arange(categories_sessions_objects_plot.shape[0])
ax3.barh(ax3_y_pos, counts_sessions_objects_plot, height=0.8)
ax3.set_yticks(ax3_y_pos, labels=categories_sessions_objects_plot)
ax3.tick_params(labelsize=9)
ax3.invert_yaxis()
ax3.set_ylabel("Object(s) brought to session", size=10, labelpad=5)
ax3.set_xlim(0, 200)
ax3.set_xticks(np.arange(0, 201, 40))

# Create the bar chart for number of instances of different objects being carried in trials:
ax4_y_pos = np.arange(categories_trials_objects_plot.shape[0])
ax4.barh(ax4_y_pos, counts_trials_objects_plot, height=0.8)
ax4.set_yticks(ax4_y_pos, labels=categories_trials_objects_plot)
ax4.tick_params(labelsize=9)
ax4.invert_yaxis()
ax4.set_ylabel("Object(s) carried in trial", size=10, labelpad=5)
ax4.set_xlim(0, 1000)
ax4.set_xticks(np.arange(0, 1001, 200))

# Create the bar chart for mass of carried object(s) in kg:
ax5_y_pos = np.arange(4)
ax5.barh(ax5_y_pos, counts_masses_objects, height=0.8)
ax5.set_yticks(ax5_y_pos, labels=["0-5", "5-10", "10-15", "15-20"])
ax5.tick_params(labelsize=9)
ax5.invert_yaxis()
ax5.set_ylabel("Mass of carried\nobject(s) (kg)", size=10, labelpad=5)
ax5.set_xlim(0, 150)
ax5.set_xticks(np.arange(0, 151, 30))

# Create the bar chart for mass of carried object(s) relative to body mass as a percentage:
ax6_y_pos = np.arange(4)
ax6.barh(ax6_y_pos, counts_masses_objects_pc_bm, height=0.8)
ax6.set_yticks(ax6_y_pos, labels=["0-5", "5-10", "10-15", "15-20"])
ax6.tick_params(labelsize=9)
ax6.invert_yaxis()
ax6.set_ylabel("Mass of carried\nobject(s) (% BM)", size=10, labelpad=5)
ax6.set_xlim(0, 150)
ax6.set_xticks(np.arange(0, 151, 30))

# Create the bar chart for number of instances of different footwear being worn in trials:
ax7_y_pos = np.arange(categories_footwear_plot.shape[0])
ax7.barh(ax7_y_pos, counts_footwear_plot, height=0.8)
ax7.set_yticks(ax7_y_pos, labels=categories_footwear_plot)
ax7.tick_params(labelsize=9)
ax7.invert_yaxis()
ax7.set_ylabel('Footwear worn in trial', size=10)
ax7.set_xlim(0, 600)
ax7.set_xticks(np.arange(0, 601, 120))

# Create the bar chart for number of instances of different clothing tops being worn in trials:
ax8_y_pos = np.arange(categories_clothing_top_plot.shape[0])
ax8.barh(ax8_y_pos, counts_clothing_top_plot, height=0.8)
ax8.set_yticks(ax8_y_pos, labels=categories_clothing_top_plot)
ax8.tick_params(labelsize=9)
ax8.invert_yaxis()
ax8.set_ylabel("Clothing - top", size=10, labelpad=5)
ax8.set_xlim(0, 600)
ax8.set_xticks(np.arange(0, 601, 120))

# Create the bar chart for number of instances of different clothing bottom items being worn in trials:
ax9_y_pos = np.arange(categories_clothing_bot_plot.shape[0])
ax9.barh(ax9_y_pos, counts_clothing_bot_plot, height=0.8)
ax9.set_yticks(ax9_y_pos, labels=categories_clothing_bot_plot)
ax9.tick_params(labelsize=9)
ax9.invert_yaxis()
ax9.set_ylabel("Clothing - bottom", size=10, labelpad=5)
ax9.set_xlim(0, 600)
ax9.set_xticks(np.arange(0, 601, 120))

# Create the bar chart for number of instances of different clothing top-to-bottom items being worn in trials:
ax10_y_pos = np.arange(categories_clothing_ttb_plot.shape[0])
ax10.barh(ax10_y_pos, counts_clothing_ttb_plot, height=0.8)
ax10.set_yticks(ax10_y_pos, labels=categories_clothing_ttb_plot)
ax10.tick_params(labelsize=9)
ax10.invert_yaxis()
ax10.set_ylabel("Clothing - top-to-bottom", size=10, labelpad=5)
ax10.set_xlim(0, 10)
ax10.set_xticks(np.arange(0, 11, 2))

fig.supxlabel("Count", size=10)

# Save and/or show the figure:
plt.savefig('./Figures/fig 6 (bar chart).pdf', dpi=1200)
# plt.show()
# # -------------------------------------------------------------------------------------------------------------------
# # SECTION 9. GENERATE THE FIGURE - END
# # -------------------------------------------------------------------------------------------------------------------
