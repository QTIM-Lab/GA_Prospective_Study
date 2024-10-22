import os, pdb
import pandas as pd
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from dotenv import load_dotenv
from itertools import combinations

load_dotenv()  # take environment variables from .env.
DATA_DIR = os.getenv("DATA_DIR")
OUT_DIR = os.getenv("OUT_DIR")

test_images = {
"1": "AxisUCH01_2962_84400_20150520095224346c03029965ed58812.png",
"2": "AxisUCH01_2962_84400_20150520095231335c2e7b2fc593cef52.png",
"3": "AxisUCH01_3441_95115_201509160910553589f5842fc6fb483c1.png",
"4": "AxisUCH01_4978_85284_20150601131525411a52867ff949f49a.png",
"5": "AxisUCH01_4978_85284_20150601131532993f67246550f5a6edc.png",
"6": "AxisUCH01_8026_123640_20160615112455600c2ee8680d2934ffe.png",
"7": "AxisUCH01_8026_123640_201606151125002498e30a4b5009e04ce.png",
"8": "AxisUCH01_21901_107977_201601281143170606df48ffdca68fb06.png",
"9": "AxisUCH01_22462_111835_20160307154631576a577c3b669742e69.png",
"10": "AxisUCH01_23945_122523_201606061501205589063520431b3b84e.png"
}

def get_test_image_annotations(image_list):
    # pdb.set_trace()
    test_image_annotations = {}
    for image in image_list:
        for test_image_key in test_images.keys():
            if image.find(test_images[test_image_key]) != -1:
                test_image_annotations[test_image_key] = image
    return test_image_annotations

andres_dir = os.path.join(DATA_DIR, "andres_manual_1")
andres_images = [file for file in os.listdir(andres_dir) if file.find(".png") != -1]
andres_test_image_annotations = get_test_image_annotations(andres_images)

bmarks_dir = os.path.join(DATA_DIR, "bmarks_manual_1")
bmarks_images = [file for file in os.listdir(bmarks_dir) if file.find(".png") != -1]
bmarks_test_image_annotations = get_test_image_annotations(bmarks_images)

dmilner_dir = os.path.join(DATA_DIR, "dmilner_manual_1")
dmilner_images = [file for file in os.listdir(dmilner_dir) if file.find(".png") != -1]
dmilner_test_image_annotations = get_test_image_annotations(dmilner_images)

iroseto_dir = os.path.join(DATA_DIR, "iroseto_manual_1")
iroseto_images = [file for file in os.listdir(iroseto_dir) if file.find(".png") != -1]
iroseto_test_image_annotations = get_test_image_annotations(iroseto_images)

kaberidgway_dir = os.path.join(DATA_DIR, "kaberidgway_manual_1")
kaberidgway_images = [file for file in os.listdir(kaberidgway_dir) if file.find(".png") != -1]
kaberidgway_test_image_annotations = get_test_image_annotations(kaberidgway_images)

larguinchona_dir = os.path.join(DATA_DIR, "larguinchona_manual_1")
larguinchona_images = [file for file in os.listdir(larguinchona_dir) if file.find(".png") != -1]
larguinchona_test_image_annotations = get_test_image_annotations(larguinchona_images)

lbarrientos_dir = os.path.join(DATA_DIR, "lbarrientos_manual_1")
lbarrientos_images = [file for file in os.listdir(lbarrientos_dir) if file.find(".png") != -1]
lbarrientos_test_image_annotations = get_test_image_annotations(lbarrientos_images)

mtukel_dir = os.path.join(DATA_DIR, "mtukel_manual_1")
mtukel_images = [file for file in os.listdir(mtukel_dir) if file.find(".png") != -1]
mtukel_test_image_annotations = get_test_image_annotations(mtukel_images)

rgnanaraj_dir = os.path.join(DATA_DIR, "rgnanaraj_manual_1")
rgnanaraj_images = [file for file in os.listdir(rgnanaraj_dir) if file.find(".png") != -1]
rgnanaraj_test_image_annotations = get_test_image_annotations(rgnanaraj_images)

zgill_dir = os.path.join(DATA_DIR, "zgill_manual_1")
zgill_images = [file for file in os.listdir(zgill_dir) if file.find(".png") != -1]
zgill_test_image_annotations = get_test_image_annotations(zgill_images)


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


annotators = [
    "andres",
    "bmarks",
    "dmilner",
    "iroseto",
    "kaberidgway",
    "larguinchona",
    "lbarrientos",
    "mtukel",
    "rgnanaraj",
    "zgill",
]

combos_annotators = list(combinations(annotators, 2))

for image_key in test_images.keys():
    bmarks_image_key_flat = get_image_array(os.path.join(bmarks_dir, bmarks_test_image_annotations[image_key]))
    larguinchona_image_key_flat = get_image_array(os.path.join(larguinchona_dir, larguinchona_test_image_annotations[image_key]))
    iroseto_image_key_flat = get_image_array(os.path.join(iroseto_dir, iroseto_test_image_annotations[image_key]))
    lbarrientos_image_key_flat = get_image_array(os.path.join(lbarrientos_dir, lbarrientos_test_image_annotations[image_key]))
    mtukel_image_key_flat = get_image_array(os.path.join(mtukel_dir, mtukel_test_image_annotations[image_key]))
    rgnanaraj_image_key_flat = get_image_array(os.path.join(rgnanaraj_dir, rgnanaraj_test_image_annotations[image_key]))
    zgill_image_key_flat = get_image_array(os.path.join(zgill_dir, zgill_test_image_annotations[image_key]))
    
    # Flattened Arrayas
    flattened_arrays = [
        bmarks_image_key_flat,
        larguinchona_image_key_flat,
        iroseto_image_key_flat,
        lbarrientos_image_key_flat,
        mtukel_image_key_flat,
        rgnanaraj_image_key_flat,
        zgill_image_key_flat,
    ]
    combos_arrays = list(combinations(flattened_arrays, 2))
    f1_combo_scores = {}
    # Calculate F1 scores
    for combo_annotators, combo_arrays in zip(combos_annotators, combos_arrays):
        # pdb.set_trace()
        f1 = f1_score(combo_arrays[0], combo_arrays[1])
        f1_combo_scores[f"{combo_annotators[0]}_{combo_annotators[1]}"] = f1
    # Create a matrix for the heatmap
    annotators_count = len(annotators)
    f1_matrix = np.zeros((annotators_count, annotators_count))
    # Fill the matrix with F1 scores
    for (i, j), f1 in zip(combinations(range(annotators_count), 2), f1_combo_scores.values()):
        f1_matrix[i, j] = f1
        f1_matrix[j, i] = f1  # Because the matrix is symmetric
    f1_df = pd.DataFrame(f1_matrix, index=annotators, columns=annotators)
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(f1_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("F1 Score Heatmap between Annotators")
    plt.savefig(os.path.join(OUT_DIR, f"f1_score_heatmap_image_{image_key}.png"))
    plt.close()
    # pdb.set_trace()

def main():
    pass




if __name__ == "__main__":
    main()


