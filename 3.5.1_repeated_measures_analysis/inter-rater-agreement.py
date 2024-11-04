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

#  Calculate ICC (inter-rater agreement)
## we use test data when the stats are inter

# data_ann_andres = df[df['Annotator'] == "andres"]
# data_ann_bmarks = df[df['Annotator'] == "bmarks"]
# data_ann_dmilner = df[df['Annotator'] == "dmilner"]
# data_ann_iroseto = df[df['Annotator'] == "iroseto"]
# data_ann_kaberidgway = df[df['Annotator'] == "kaberidgway"]
# data_ann_larguinchona = df[df['Annotator'] == "larguinchona"]
# data_ann_lbarrientos = df[df['Annotator'] == "lbarrientos"]
# data_ann_mtukel = df[df['Annotator'] == "mtukel"]
# data_ann_rgnanaraj = df[df['Annotator'] == "rgnanaraj"]
# data_ann_zgill = df[df['Annotator'] == "zgill"]


# icc_results = pg.intraclass_corr(data=example_data,
#                                 targets='Image',
#                                 raters='Annotator',
#                                 ratings='DICE')


# Filter for the test set
test_set_df = df[df['image_id'].isin(['image_101','image_102','image_103','image_104','image_105','image_106','image_107','image_108','image_109','image_110'])]

def analyze_interrater_reliability(data_df):
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
      * raters="Annotator": makes it so that the algorithm
        uses annotators as raters so that all images for all exercises
        are used to create an ICC. We feed in two groups of exercizes to get
        different perspectives. 
    """
    
    icc_results = pg.intraclass_corr(data=data_df,
                                    targets='Image',
                                    raters='Annotator',
                                    ratings='DICE',
                                    nan_policy='omit')
    
    return icc_results

test_set_df.head(1)

# Ex1 and Ex2
test_set_ex1_ex2_df = test_set_df[test_set_df['Session'].isin(['ex1','ex2'])]
ex_1_ex2_icc_results = analyze_interrater_reliability(test_set_ex1_ex2_df)
ex_1_ex2_icc_results['exercises'] = 'ex_1 & ex2'

# Ex3, Ex4 and Ex5
test_set_ex3_ex4_ex5_df = test_set_df[test_set_df['Session'].isin(['ex3','ex4', 'ex5'])]
ex_3_ex4_ex5_icc_results = analyze_interrater_reliability(test_set_ex3_ex4_ex5_df)
ex_3_ex4_ex5_icc_results['exercises'] = 'ex_1 & ex_2 & ex_3'


icc_results = pd.concat([
    ex_1_ex2_icc_results,
    ex_3_ex4_ex5_icc_results,
    ], axis=0)

# Plotting
plt.figure(figsize=(10, 8))
ICC_scatter_plot = sns.scatterplot(data=icc_results, x='Type', y='ICC', hue='exercises', style='exercises', s=150)

ICC_scatter_plot.set_xlabel('ICC Type')
ICC_scatter_plot.set_ylabel('Value')
ICC_scatter_plot.legend(title = 'Exercises')
ICC_scatter_plot.set_title('Inter Rater ICC Values by Type for Manual and AI-assisted Annotations')
plt.savefig(os.path.join(STATS_DIR, "inter-rater-by-exercise-type.png"))





# Create visualization
plt.figure(figsize=(80, 40))

# Violin plot
plt.subplot(2, 1, 1)
sns.violinplot(data=df, x='Annotator', y='DICE', inner='box')
plt.title('Distribution of DICE Scores by Annotator')
plt.ylabel('DICE Score vs Ground Truth')

# Box plot by image
plt.subplot(2, 1, 2)
sns.boxplot(data=df, x='Image', y='DICE', hue='Annotator')
plt.title('DICE Scores by Image and Annotator')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(STATS_DIR, f"inter-rater.png"))

print("\nIntraclass (inter-rater) Correlation Results:")
print(icc_results)





# Create the first plot (Violin plot)
plt.figure(figsize=(80, 40))  # Adjust size as needed
sns.violinplot(data=df, x='Annotator', y='DICE', inner='box')
plt.title('Distribution of DICE Scores by Annotator', fontsize=30)
plt.ylabel('DICE Score vs Ground Truth', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(os.path.join(STATS_DIR, "inter_rater_violin.png"))  # Save the violin plot
plt.show()  # Show the first plot

# Create the second plot (Box plot by image)
plt.figure(figsize=(80, 40))  # Adjust size as needed
sns.boxplot(data=df, x='Image', y='DICE', hue='Annotator')
plt.title('DICE Scores by Image and Annotator', fontsize=30)
plt.xticks(rotation=45, fontsize=20)  # Rotate x-ticks for better readability
plt.ylabel('DICE Score vs Ground Truth', fontsize=24)
plt.yticks(fontsize=20)
plt.savefig(os.path.join(STATS_DIR, "inter_rater_box.png"))  # Save the box plot
plt.show()  # Show the second plot



print("\nIntraclass Correlation Results:")
print(icc_results)
