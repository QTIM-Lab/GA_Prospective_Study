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


load_dotenv()  # take environment variables from .env.
DATA_DIR = os.getenv("DATA_DIR")
OUT_DIR = os.getenv("OUT_DIR")

def get_image_array(image_path):
    sample_image = Image.open(image_path)
    img_array = np.array(sample_image)
    img_array_r = img_array[:,:,0]
    # img_array_g = img_array[:,:,1]
    # img_array_b = img_array[:,:,2]
    # img_array_a = img_array[:,:,3]
    # pdb.set_trace()
    if np.all(img_array_r == 0):
        img_array_normalized = img_array_r
    else:
        img_array_normalized = img_array_r / np.max(img_array_r)
    # Plot the image using Seaborn and Matplotlib
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(img_array[0], cbar=False, xticklabels=False, yticklabels=False)
    # plt.title("Image Preview")
    # plt.imsave(os.path.join(OUT_DIR, os.path.basename(image_path)), img_array_normalized)
    # Flatten the 2D arrays to 1D
    img_array_r_flat = img_array_normalized.flatten()
    return img_array_r_flat


# Annotators
annotators = {
    "andres":{f"ex{i}":[] for i in range(1,6)},
    "bmarks":{f"ex{i}":[] for i in range(1,6)},
    "dmilner":{f"ex{i}":[] for i in range(1,6)},
    "iroseto":{f"ex{i}":[] for i in range(1,6)},
    "kaberidgway":{f"ex{i}":[] for i in range(1,6)},
    "larguinchona":{f"ex{i}":[] for i in range(1,6)},
    "lbarrientos":{f"ex{i}":[] for i in range(1,6)},
    "mtukel":{f"ex{i}":[] for i in range(1,6)},
    "rgnanaraj":{f"ex{i}":[] for i in range(1,6)},
    "zgill":{f"ex{i}":[] for i in range(1,6)},
}

# Get image names and paths
for annotator in annotators.keys():
    # annotator = list(annotators.keys())[0] # loop 1
    image_names = pd.read_csv(os.path.join(DATA_DIR, "annotator_image_keys", f"{annotator}_image_key.csv"))['image_path_orig'].to_list()
    annotators[annotator]['images'] = [os.path.join(DATA_DIR, "Images", "Test_Train_Final", image) for image in image_names]
    for ex in ['ex1','ex2','ex3','ex4','ex5']:
        # ex = list(annotators[annotator].keys())[0] # loop 1
        files = os.listdir(os.path.join(DATA_DIR, "Annotations", f"{ex}", f"{annotator}_{ex}"))
        # pdb.set_trace()
        segmentations = [os.path.join(DATA_DIR, "Annotations", f"{ex}", f"{annotator}_{ex}", file) for file in files if file.find("png") != -1]
        annotators[annotator][f'{ex}'] = segmentations

# Need GT and we need all 5 image annotation sets.
## GT
nj_files = os.listdir(os.path.join(DATA_DIR, "Annotations","niranjan_110"))
nj_segmentations = [os.path.join(DATA_DIR, "Annotations", "niranjan_110", file) for file in nj_files if file.find("png") != -1]
# len(nj_segmentations) # 110 - QA
test_train_final_files = os.listdir(os.path.join(DATA_DIR, "Images", "Test_Train_Final"))
nj_images = [os.path.join(DATA_DIR, "Images", "Test_Train_Final", image) for image in test_train_final_files]

niranjan_110 = {
    "niranjan":{
        "ground_truth":nj_segmentations,
        "images":nj_images,
        },
}

## DICE
# niranjan_110['niranjan']['ground_truth']
# niranjan_110['niranjan']['images']

dice = {
    "andres":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "bmarks":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "dmilner":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "iroseto":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "kaberidgway":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "larguinchona":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "lbarrientos":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "mtukel":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "rgnanaraj":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "zgill":{f'{ex}':None for ex in ['ex1','ex2','ex3','ex4','ex5']},
}

for annotator in annotators.keys(): #in ['bmarks']:
    # annotator = list(annotators.keys())[0] # loop 1
    print(f"On annotator: {annotator}")
    for ex in ['ex1','ex2','ex3','ex4','ex5']:
        # ex = list(annotator.keys())[0] # loop 1
        # pdb.set_trace()
        print(f"On Ex: {ex}")
        count = 0
        if len(annotators[annotator][f'{ex}']) != 20:
            pass
        else:
            for image in annotators[annotator]['images']:
                count += 1
                print(f'On image: {image} | count: {count}')
                # if image == '/sddata/data/GA_Prospective_Study/Data/Images/Test_Train_Final/AxisUCH01_2962_84400_20150520095224346c03029965ed58812.png':
                #     pdb.set_trace()
                # image = annotators[annotator]['images'][0] # loop 1
                # pdb.set_trace()
                sample_image_file = image
                sample_file_name = os.path.basename(sample_image_file)
                sample_file = [image for image in annotators[annotator][ex] if image.find(sample_file_name) != -1][0]
                gt_file = [image for image in niranjan_110['niranjan']['ground_truth'] if image.find(sample_file_name) != -1][0]
                sample_image_np = get_image_array(sample_image_file)
                sample_np = get_image_array(sample_file)
                gt_np = get_image_array(gt_file)
                # np.unique(sample_np)
                # np.unique(gt_np)
                f1 = f1_score(sample_np, gt_np)
                dice[annotator][ex] = f1













import pandas as pd
import pingouin as pg

# data = pg.read_dataset('icc')
# icc = pg.intraclass_corr(data=data, targets='Wine', raters='Judge', ratings='Scores').round(3)


# Example with more images to test:
example_data = pd.DataFrame({
    'Image': [1,1,2,2,3,3,4,4,5,5] * 2,  # 5 images, 2 sessions each
    'Annotator': ['Annotator1', 'Annotator2'] * 10,
    'Session': [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2],
    'DICE': [0.8, 0.75, 0.82, 0.76, 0.85, 0.83, 0.87, 0.84, 0.81, 0.79,
             0.81, 0.76, 0.83, 0.77, 0.84, 0.82, 0.86, 0.83, 0.82, 0.78]
})
# simulate missing data
example_data = pd.DataFrame({
    'Image': [1,2,2,3,3,4,4,5,5] + [1,1,2,2,3,3,4,4,5,5],  # 5 images, 2 sessions each
    'Annotator': ['Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2', 'Annotator1', 'Annotator2'],
    'Session': [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2],
    'DICE': [0.75, 0.82, 0.76, 0.85, 0.83, 0.87, 0.84, 0.81, 0.79,
             0.81, 0.76, 0.83, 0.77, 0.84, 0.82, 0.86, 0.83, 0.82, 0.78]
})


example_data_ann_1 = example_data[example_data['Annotator'] == 'Annotator1']
example_data_ann_2 = example_data[example_data['Annotator'] == 'Annotator2']



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
    """
    # Calculate ICC
    icc = pg.intraclass_corr(data=data_df, 
                            targets='Image', 
                            raters='Session',
                            ratings='DICE')
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_df, x='Image', y='DICE')
    plt.title('Score Distribution Across Images')
    plt.xticks(rotation=45)
    
    return icc, plt.gcf()


icc_results, fig = analyze_intrarater_reliability(example_data)
icc_results, fig = analyze_intrarater_reliability(example_data_ann_1)
icc_results, fig = analyze_intrarater_reliability(example_data_ann_2)


plt.savefig(os.path.join(OUT_DIR, f"intra-rater.png"))
plt.savefig(os.path.join(OUT_DIR, f"intra-rater-ann1.png"))
plt.savefig(os.path.join(OUT_DIR, f"intra-rater-ann2.png"))







# Calculate ICC (inter-rater agreement)
icc_results = pg.intraclass_corr(data=example_data.
                                targets='Image',
                                raters='Annotator',
                                ratings='DICE')

# Create visualization
plt.figure(figsize=(12, 6))

# Violin plot
plt.subplot(1, 2, 1)
sns.violinplot(data=df, x='Annotator', y='DICE', inner='box')
plt.title('Distribution of DICE Scores by Annotator')
plt.ylabel('DICE Score vs Ground Truth')

# Box plot by image
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Image', y='DICE', hue='Annotator')
plt.title('DICE Scores by Image and Annotator')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"inter-rater.png"))

print("\nIntraclass Correlation Results:")
print(icc_results)
