import os, re, json, time, hashlib, unicodedata, warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.distance import Levenshtein
from scipy.stats import spearmanr
import easyocr
import ollama

warnings.filterwarnings("ignore")

STUDENT = "Капітанюк М.А."
VARIANT = 3
STUDENT_ID = 901
SEED = 1000 * VARIANT + STUDENT_ID
rng = np.random.default_rng(SEED)

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print(f"Студент: {STUDENT} | варіант {VARIANT} | seed = {SEED}")

def stamp(ax, title):
    ax.set_title(f"{title}\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)
