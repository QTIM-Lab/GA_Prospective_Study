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
nj_files = os.listdir(os.path.join(DATA_DIR, "Annotations","niranjan_110", "binarized_red_new_PPA_files"))
nj_segmentations = [os.path.join(DATA_DIR, "Annotations", "niranjan_110", "binarized_red_new_PPA_files", file) for file in nj_files if file.find("png") != -1]
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
    "andres":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "bmarks":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "dmilner":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "iroseto":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "kaberidgway":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "larguinchona":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "lbarrientos":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "mtukel":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "rgnanaraj":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
    "zgill":{f'{ex}':{} for ex in ['ex1','ex2','ex3','ex4','ex5']},
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
                dice[annotator][ex][image] = f1

# write
with open(os.path.join(STATS_DIR, "dice_scores.json"), mode='w') as file:
    file.write(json.dumps(dice, indent=2))


# read
with open(os.path.join(STATS_DIR, "dice_scores.json"), mode='r') as file:
    dice = json.loads(file.read())


nj_images
image_key = pd.read_csv(os.path.join(DATA_DIR, "Images", "image_key.csv"))
image_key['file_name'] = [os.path.basename(file) for file in image_key['image_path_orig']]

# DICE json to dataframe
df = pd.DataFrame()
for annotator in annotators.keys():
    for ex in ['ex1','ex2','ex3','ex4','ex5']:
        if len(annotators[annotator][f'{ex}']) != 20:
            pass
        else:
            for image in annotators[annotator]['images']:
                # pdb.set_trace()
                image_id = f"image_{image_key[image_key['file_name'] == os.path.basename(image)]['order'].values[0]}"
                df_row = pd.DataFrame({
                    'Image':[image],
                    'Annotator':[annotator],
                    'Session':[ex],
                    'DICE':[dice[annotator][ex][image]],
                    'image_id': image_id
                    })
                # pdb.set_trace()
                df = pd.concat([df, df_row])

df.to_csv(os.path.join(STATS_DIR, "dice_scores.csv"), index=None)

