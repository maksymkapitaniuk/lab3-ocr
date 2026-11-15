import shutil
import os, re, json, time, hashlib, unicodedata, warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
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

_FD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for f in ("DejaVuSans.ttf", "DejaVuSerif.ttf"):
    if not os.path.exists(f"fonts/{f}"):
        shutil.copy(os.path.join(_FD, f), "fonts")

FONTS = [f"fonts/{n}" for n in ["DejaVuSans.ttf", "DejaVuSerif.ttf",
                                "LiberationSans-Regular.ttf", "LiberationSerif-Regular.ttf"]]
missing = [f for f in FONTS if not os.path.exists(f)]
assert not missing, f"Покладіть у папку fonts/ файли: {missing}"

os.makedirs("data", exist_ok=True)
os.makedirs("data/synthetic", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print(f"Студент: {STUDENT} | варіант {VARIANT} | seed = {SEED}")

def stamp(ax, title):
    ax.set_title(f"{title}\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)

with open("corpus.txt", encoding="utf-8") as f:
  corpus = [s.strip() for s in f if 20 <= len(s.strip()) <= 70]
lines = [str(s) for s in rng.choice(corpus, size=40, replace=False)]

def render_line(text, font_path, size=32, pad=12):
  font = ImageFont.truetype(font_path, size)
  l, t, r, b = font.getbbox(text)
  img = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 255)
  ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=0)
  return np.array(img)

def degrade(img, kind, level, rng):
  out = img.astype(np.float32)
  if kind == "noise" and level > 0:
    out = out + rng.normal(0, level, out.shape)
  elif kind == "blur" and level > 0:
    out = cv2.GaussianBlur(out, (0, 0), level)
  elif kind == "rotate" and level != 0:
    h, w = out.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), level, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(out, M, (nw, nh), borderValue=255)
  elif kind == "scale" and level > 1:
    h, w = out.shape
    small = cv2.resize(out, (int(w / level), int(h / level)),
                       interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
  elif kind == "jpeg" and level < 100:
    ok, enc = cv2.imencode(".jpg", out.astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, int(level)])
    out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)
  elif kind == "contrast":
    out = 255 - (255 - out) * (level / 100)
  return np.clip(out, 0, 255).astype(np.uint8)

FACTOR, LEVELS = "contrast", [100, 70, 50, 35, 25, 15]
records = []

for i, text in enumerate(lines):
  font = FONTS[i % len(FONTS)]
  clean = render_line(text, font, size=int(rng.integers(22, 36)))
  for lvl in LEVELS:
    path = f"data/synthetic/{i:03d}_{FACTOR}_{lvl}.png"
    cv2.imwrite(path, degrade(clean, FACTOR, lvl, rng))
    records.append({"id": i, "level": lvl, "path": path, "gt": text,
                    "font": os.path.basename(font)})
    
ds = pd.DataFrame(records)
ds.to_csv("data/ground_truth.csv", index=False, lineterminator="\n")
with open("data/ground_truth.csv", "rb") as f:
  sha = hashlib.sha256(f.read()).hexdigest()[:16]
print("Записів:", len(ds), "| SHA-256 еталону:", sha)

fig, axes = plt.subplots(len(LEVELS), 1, figsize=(7, 0.75 * len(LEVELS)))
for ax, lvl in zip(axes, LEVELS):
  img = cv2.imread(f"data/synthetic/000_{FACTOR}_{lvl}.png", cv2.IMREAD_GRAYSCALE)
  ax.imshow(img, cmap="gray", vmin=0, vmax=255)
  ax.set_title(f"контраст = {lvl}", fontsize=8, loc="left")
  ax.axis("off")
fig.savefig("figures/fig0_samples.png", dpi=150, bbox_inches="tight")
