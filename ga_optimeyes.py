import os, pdb
import pandas as pd
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from dotenv import load_dotenv


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


bmarks_dir = os.path.join(DATA_DIR, "bmarks_manual_1")
bmarks_images = [file for file in os.listdir(bmarks_dir) if file.find(".png") != -1]
bmarks_test_image_annotations = get_test_image_annotations(bmarks_images)

larguinchona_dir = os.path.join(DATA_DIR, "larguinchona_manual_1")
larguinchona_images = [file for file in os.listdir(larguinchona_dir) if file.find(".png") != -1]
larguinchona_test_image_annotations = get_test_image_annotations(larguinchona_images)

def get_image_array(image_path):
    sample_image = Image.open(image_path)
    img_array = np.array(sample_image)
    img_array_r = img_array[:,:,0]
    # img_array_g = img_array[:,:,1]
    # img_array_b = img_array[:,:,2]
    # img_array_a = img_array[:,:,3]
    img_array_normalized = img_array_r / np.max(img_array_r)
    # Plot the image using Seaborn and Matplotlib
    plt.figure(figsize=(10, 10))
    sns.heatmap(img_array[0], cbar=False, xticklabels=False, yticklabels=False)
    plt.title("Image Preview")
    plt.imsave(os.path.join(OUT_DIR, os.path.basename(image_path)), img_array_normalized)
    # Flatten the 2D arrays to 1D
    img_array_r_flat = img_array_normalized.flatten()
    return img_array_r_flat


bmarks_image_1_flat = get_image_array(os.path.join(bmarks_dir, bmarks_test_image_annotations["1"]))
larguinchona_image_1_flat = get_image_array(os.path.join(larguinchona_dir, larguinchona_test_image_annotations["1"]))


# Calculate F1 score
f1 = f1_score(bmarks_image_1_flat, larguinchona_image_1_flat)
print(f"F1 Score: {f1}")
