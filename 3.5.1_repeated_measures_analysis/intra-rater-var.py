import os, pdb
import pandas as pd
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from dotenv import load_dotenv
from itertools import combinations
from scipy import stats
import json
import pingouin as pg


load_dotenv()  # take environment variables from .env.
DATA_DIR = os.getenv("DATA_DIR")
OUT_DIR = os.getenv("OUT_DIR")
STATS_DIR = os.getenv("STATS_DIR")

df = pd.read_csv(os.path.join(STATS_DIR, "dice_scores.csv"))
df['Session_Group'] = df['Session'].apply(lambda x: 'ex1_ex2' if x in ['ex1', 'ex2'] else 'ex3_ex4_ex5')
df.head(2)
# ------------------------------------ fake data ------------------------------------ #

# pingouin data
# data = pg.read_dataset('icc')
# icc = pg.intraclass_corr(data=data, targets='Wine', raters='Judge', ratings='Scores').round(3)

# Example with more images to test:
# example_data = pd.DataFrame({
#     'Image': [1,1,2,2,3,3,4,4,5,5] * 2,  # 5 images, 2 sessions each
#     'Annotator': ['Annotator1', 'Annotator2'] * 10,
#     'Session': [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2],
#     'DICE': [0.8, 0.75, 0.82, 0.76, 0.85, 0.83, 0.87, 0.84, 0.81, 0.79,
#              0.81, 0.76, 0.83, 0.77, 0.84, 0.82, 0.86, 0.83, 0.82, 0.78]
# })
# simulate missing data
# example_data = pd.DataFrame({
#     'Image': [1,2,2,3,3,4,4,5,5] + [1,1,2,2,3,3,4,4,5,5],  # 5 images, 2 sessions each
#     'Annotator': ['Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2'],
#     'Session': [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2],
#     'DICE': [0.75, 0.82, 0.76, 0.85, 0.83, 0.87, 0.84, 0.81, 0.79,
#              0.81, 0.76, 0.83, 0.77, 0.84, 0.82, 0.86, 0.83, 0.82, 0.78]
# })
# example_data_ann_1 = example_data[example_data['Annotator'] == 'Annotator1']
# example_data_ann_2 = example_data[example_data['Annotator'] == 'Annotator2']

# ------------------------------------ fake data ------------------------------------ #

# Real data
## we use all data when the stats are intra
data_ann_andres_ex1_ex2 = df[(df['Annotator'] == "andres") & (df['Session_Group'] == "ex1_ex2")]
data_ann_bmarks_ex1_ex2 = df[(df['Annotator'] == "bmarks") &(df['Session_Group'] == "ex1_ex2")]
data_ann_dmilner_ex1_ex2 = df[(df['Annotator'] == "dmilner") &(df['Session_Group'] == "ex1_ex2")]
data_ann_iroseto_ex1_ex2 = df[(df['Annotator'] == "iroseto") &(df['Session_Group'] == "ex1_ex2")]
data_ann_kaberidgway_ex1_ex2 = df[(df['Annotator'] == "kaberidgway") &(df['Session_Group'] == "ex1_ex2")]
data_ann_larguinchona_ex1_ex2 = df[(df['Annotator'] == "larguinchona") &(df['Session_Group'] == "ex1_ex2")]
data_ann_lbarrientos_ex1_ex2 = df[(df['Annotator'] == "lbarrientos") &(df['Session_Group'] == "ex1_ex2")]
data_ann_mtukel_ex1_ex2 = df[(df['Annotator'] == "mtukel") &(df['Session_Group'] == "ex1_ex2")]
data_ann_rgnanaraj_ex1_ex2 = df[(df['Annotator'] == "rgnanaraj") &(df['Session_Group'] == "ex1_ex2")]
data_ann_zgill_ex1_ex2 = df[(df['Annotator'] == "zgill") &(df['Session_Group'] == "ex1_ex2")]

data_ann_andres_ex3_ex4_ex5 = df[(df['Annotator'] == "andres") &(df['Session_Group'] == "ex3_ex4_ex5")]
data_ann_bmarks_ex3_ex4_ex5 = df[(df['Annotator'] == "bmarks") &(df['Session_Group'] == "ex3_ex4_ex5")]
data_ann_dmilner_ex3_ex4_ex5 = df[(df['Annotator'] == "dmilner") &(df['Session_Group'] == "ex3_ex4_ex5")]
data_ann_iroseto_ex3_ex4_ex5 = df[(df['Annotator'] == "iroseto") &(df['Session_Group'] == "ex3_ex4_ex5")]
data_ann_kaberidgway_ex3_ex4_ex5 = df[(df['Annotator'] == "kaberidgway") &(df['Session_Group'] == "ex3_ex4_ex5")]
data_ann_larguinchona_ex3_ex4_ex5 = df[(df['Annotator'] == "larguinchona") &(df['Session_Group'] == "ex3_ex4_ex5")]
data_ann_lbarrientos_ex3_ex4_ex5 = df[(df['Annotator'] == "lbarrientos") &(df['Session_Group'] == "ex3_ex4_ex5")]
data_ann_mtukel_ex3_ex4_ex5 = df[(df['Annotator'] == "mtukel") &(df['Session_Group'] == "ex3_ex4_ex5")]
data_ann_rgnanaraj_ex3_ex4_ex5 = df[(df['Annotator'] == "rgnanaraj") &(df['Session_Group'] == "ex3_ex4_ex5")]
data_ann_zgill_ex3_ex4_ex5 = df[(df['Annotator'] == "zgill") &(df['Session_Group'] == "ex3_ex4_ex5")]


def analyze_intrarater_reliability(data_df):
    """
    Analyze intra-rater reliability using ICC and create visualization
    
    Parameters:
    data_df: DataFrame with columns 'Image', 'Session', 'Score'
    
    Returns:
    icc_results: ICC analysis results
    fig: Matplotlib figure with visualization

    Generally, for medical image analysis:

    ICC > 0.9: Excellent reliability
    ICC 0.75-0.9: Good reliability
    ICC 0.5-0.75: Moderate reliability
    ICC < 0.5: Poor reliability

    Key things:
      * raters="Sessions": makes it so that the algorithm
        thinks ex1-ex5 are "people". This works because dfs fed in
        are only one person at a time, so we want to compare this person
        to themselves for each image, or exercises. 
    """
    # Calculate ICC
    icc = pg.intraclass_corr(data=data_df, 
                            targets='image_id', 
                            raters='Session',
                            ratings='DICE')
    # Create visualization
    plt.figure(figsize=(30, 8))
    plt.tight_layout()
    sns.boxplot(data=data_df, x='image_id', y='DICE')
    plt.title('DICE Score Distribution Across Images')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)  # You can adjust this value as needed
    return icc, plt.gcf()


# icc_results, fig = analyze_intrarater_reliability(example_data)
# icc_results, fig = analyze_intrarater_reliability(example_data_ann_1)
# icc_results, fig = analyze_intrarater_reliability(example_data_ann_2)
# icc_results, fig = analyze_intrarater_reliability(example_data_ann_2)



icc_results_andres_ex1_ex2, fig_andres_ex1_ex2 = analyze_intrarater_reliability(data_ann_andres_ex1_ex2)
icc_results_bmarks_ex1_ex2, fig_bmarks_ex1_ex2 = analyze_intrarater_reliability(data_ann_bmarks_ex1_ex2)
icc_results_dmilner_ex1_ex2, fig_dmilner_ex1_ex2 = analyze_intrarater_reliability(data_ann_dmilner_ex1_ex2)
icc_results_iroseto_ex1_ex2, fig_iroseto_ex1_ex2 = analyze_intrarater_reliability(data_ann_iroseto_ex1_ex2)
icc_results_kaberidgway_ex1_ex2, fig_kaberidgway_ex1_ex2 = analyze_intrarater_reliability(data_ann_kaberidgway_ex1_ex2)
icc_results_larguinchona_ex1_ex2, fig_larguinchona_ex1_ex2 = analyze_intrarater_reliability(data_ann_larguinchona_ex1_ex2)
icc_results_lbarrientos_ex1_ex2, fig_lbarrientos_ex1_ex2 = analyze_intrarater_reliability(data_ann_lbarrientos_ex1_ex2)
icc_results_mtukel_ex1_ex2, fig_mtukel_ex1_ex2 = analyze_intrarater_reliability(data_ann_mtukel_ex1_ex2)
icc_results_rgnanaraj_ex1_ex2, fig_rgnanaraj_ex1_ex2 = analyze_intrarater_reliability(data_ann_rgnanaraj_ex1_ex2)
icc_results_zgill_ex1_ex2, fig_zgill_ex1_ex2 = analyze_intrarater_reliability(data_ann_zgill_ex1_ex2)


icc_results_andres_ex3_ex4_ex5, fig_andres_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_andres_ex3_ex4_ex5)
icc_results_bmarks_ex3_ex4_ex5, fig_bmarks_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_bmarks_ex3_ex4_ex5)
icc_results_dmilner_ex3_ex4_ex5, fig_dmilner_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_dmilner_ex3_ex4_ex5)
icc_results_iroseto_ex3_ex4_ex5, fig_iroseto_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_iroseto_ex3_ex4_ex5)
icc_results_kaberidgway_ex3_ex4_ex5, fig_kaberidgway_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_kaberidgway_ex3_ex4_ex5)
icc_results_larguinchona_ex3_ex4_ex5, fig_larguinchona_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_larguinchona_ex3_ex4_ex5)
icc_results_lbarrientos_ex3_ex4_ex5, fig_lbarrientos_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_lbarrientos_ex3_ex4_ex5)
icc_results_mtukel_ex3_ex4_ex5, fig_mtukel_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_mtukel_ex3_ex4_ex5)
icc_results_rgnanaraj_ex3_ex4_ex5, fig_rgnanaraj_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_rgnanaraj_ex3_ex4_ex5)
icc_results_zgill_ex3_ex4_ex5, fig_zgill_ex3_ex4_ex5 = analyze_intrarater_reliability(data_ann_zgill_ex3_ex4_ex5)


# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_andres.png"))
# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_bmarks.png"))
# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_dmilner.png"))
# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_iroseto.png"))
# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_kaberidgway.png"))
# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_larguinchona.png"))
# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_lbarrientos.png"))
# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_mtukel.png"))
# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_rgnanaraj.png"))
# plt.savefig(os.path.join(STATS_DIR, f"DICE-by-image_zgill.png"))

icc_results_andres_ex1_ex2['annotator'] = 'andres'
icc_results_bmarks_ex1_ex2['annotator'] = 'bmarks'
icc_results_dmilner_ex1_ex2['annotator'] = 'dmilner'
icc_results_iroseto_ex1_ex2['annotator'] = 'iroseto'
icc_results_kaberidgway_ex1_ex2['annotator'] = 'kaberidgway'
icc_results_larguinchona_ex1_ex2['annotator'] = 'larguinchona'
icc_results_lbarrientos_ex1_ex2['annotator'] = 'lbarrientos'
icc_results_mtukel_ex1_ex2['annotator'] = 'mtukel'
icc_results_rgnanaraj_ex1_ex2['annotator'] = 'rgnanaraj'
icc_results_zgill_ex1_ex2['annotator'] = 'zgill'

icc_results_ex1_ex2 = pd.concat([icc_results_andres_ex1_ex2,icc_results_bmarks_ex1_ex2,icc_results_dmilner_ex1_ex2,icc_results_iroseto_ex1_ex2,icc_results_kaberidgway_ex1_ex2,icc_results_larguinchona_ex1_ex2,icc_results_lbarrientos_ex1_ex2,icc_results_mtukel_ex1_ex2,icc_results_rgnanaraj_ex1_ex2,icc_results_zgill_ex1_ex2], axis=0)
icc_results_ex1_ex2['Session_Group'] = "ex1_ex2"

icc_results_andres_ex3_ex4_ex5['annotator'] = 'andres'
icc_results_bmarks_ex3_ex4_ex5['annotator'] = 'bmarks'
icc_results_dmilner_ex3_ex4_ex5['annotator'] = 'dmilner'
icc_results_iroseto_ex3_ex4_ex5['annotator'] = 'iroseto'
icc_results_kaberidgway_ex3_ex4_ex5['annotator'] = 'kaberidgway'
icc_results_larguinchona_ex3_ex4_ex5['annotator'] = 'larguinchona'
icc_results_lbarrientos_ex3_ex4_ex5['annotator'] = 'lbarrientos'
icc_results_mtukel_ex3_ex4_ex5['annotator'] = 'mtukel'
icc_results_rgnanaraj_ex3_ex4_ex5['annotator'] = 'rgnanaraj'
icc_results_zgill_ex3_ex4_ex5['annotator'] = 'zgill'

icc_results_ex3_ex4_ex5 = pd.concat([icc_results_andres_ex3_ex4_ex5,icc_results_bmarks_ex3_ex4_ex5,icc_results_dmilner_ex3_ex4_ex5,icc_results_iroseto_ex3_ex4_ex5,icc_results_kaberidgway_ex3_ex4_ex5,icc_results_larguinchona_ex3_ex4_ex5,icc_results_lbarrientos_ex3_ex4_ex5,icc_results_mtukel_ex3_ex4_ex5,icc_results_rgnanaraj_ex3_ex4_ex5,icc_results_zgill_ex3_ex4_ex5], axis=0)
icc_results_ex3_ex4_ex5['Session_Group'] = "ex3_ex4_ex5"

icc_results = pd.concat([
icc_results_ex1_ex2,
icc_results_ex3_ex4_ex5
    ], axis=0)
# Reset the index to ensure there are no duplicate index labels
icc_results = icc_results.reset_index(drop=True)




# Plotting
plt.figure(figsize=(10, 8))
icc_results_ex1_ex2 = icc_results_ex1_ex2[icc_results_ex1_ex2['annotator'] != 'rgnanaraj']
ICC_scatter_plot = sns.scatterplot(data=icc_results_ex1_ex2, x='Type', y='ICC', hue='annotator', s=150)

ICC_scatter_plot.set_xlabel('ICC Type')
ICC_scatter_plot.set_ylabel('Value')
ICC_scatter_plot.legend(title = 'Person')
# plt.legend(title="Person")
ICC_scatter_plot.set_title('Intra Rater ICC Values by Type for Each Person ex1_ex2')
plt.savefig(os.path.join(STATS_DIR, "intra-rater-by-ex1_ex2.png"))




# Plotting
plt.figure(figsize=(10, 8))
icc_results_ex3_ex4_ex5 = icc_results_ex3_ex4_ex5[icc_results_ex3_ex4_ex5['annotator'] != 'rgnanaraj']
ICC_scatter_plot = sns.scatterplot(data=icc_results_ex3_ex4_ex5, x='Type', y='ICC', hue='annotator', s=150)

ICC_scatter_plot.set_xlabel('ICC Type')
ICC_scatter_plot.set_ylabel('Value')
ICC_scatter_plot.legend(title = 'Person')
# plt.legend(title="Person")
ICC_scatter_plot.set_title('Intra Rater ICC Values by Type for Each Person ex3_ex4_ex5')
plt.savefig(os.path.join(STATS_DIR, "intra-rater-by-ex3_ex4_ex5.png"))

# ----------------


# Define the number of rows and columns for the subplot grid
rows, cols = 2, 5  # You can switch to (5, 2) if preferred

# Create the figure and subplots
fig, axes = plt.subplots(rows, cols, figsize=(20, 16))  # Adjust the figsize as needed
fig.suptitle("Intra Rater ICC Values by Type for Each Person", fontsize=16)

# Flatten the axes array for easy indexing
axes = axes.flatten()


# Loop through each annotator and plot on each subplot
for i, annotator in enumerate(icc_results['annotator'].unique()):
    # Filter the data for the current annotator
    annotator_data = icc_results[icc_results['annotator'] == annotator]
    # Plot on the current subplot
    sns.scatterplot(
        data=annotator_data,
        x='Type',
        y='ICC',
        hue='Session_Group',
        s=150,
        ax=axes[i]
    )
    # Set labels and title for each subplot
    axes[i].set_xlabel('ICC Type')
    axes[i].set_ylabel('Value')
    axes[i].set_title(f'{annotator}')
    axes[i].set_ylim(0, 1)  # Assuming ICC values are between 0 and 1
    axes[i].legend(title='Session Group', loc='upper right')
    

# Remove any extra empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])



# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to leave space for the suptitle
plt.savefig(os.path.join(STATS_DIR, "intra-rater-by-annotator-grid.png"))
# plt.show()
# ----------------