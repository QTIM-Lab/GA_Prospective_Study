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
