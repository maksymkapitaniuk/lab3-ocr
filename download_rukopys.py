import json
import math
import os
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "UkrainianCatholicUniversity/rukopys"
STUDENT, VARIANT, STUDENT_ID = "Капітанюк М.А.", 3, 901
SEED = 1000 * VARIANT + STUDENT_ID
N_MAIN = 120
N_ILLEG = 40
MAX_PER_PAGE = 3
PAD = 8
SOURCES_MAIN = ["dictation", "school", "university"]
ANNOT_MAIN = {"annotator"}
BAD_MARKERS = ("~~", "[illegible]", "{", "}")

OUT_DIR = "data/rukopys"
FIG_DIR = "figures"
CSV_PATH = "data/rukopys_labels.csv"

def repo_path(file_name):
    return file_name if file_name.startswith("train/") else f"train/{file_name}"


def load_regions():
    meta = hf_hub_download(REPO, "train/metadata.jsonl", repo_type="dataset")
    with open(meta, encoding="utf-8") as f:
        pages = [json.loads(line) for line in f if line.strip()]
    rows = []
    for p in pages:
        for k, r in enumerate(p.get("regions") or []):
            x1, y1, x2, y2 = r["bbox"]
            rows.append({
                "page_file": p["file_name"],
                "source": p.get("source"),
                "source_detail": p.get("source_detail"),
                "annotation_source": p.get("annotation_source"),
                "img_w": p.get("image_width"), "img_h": p.get("image_height"),
                "region_idx": k,
                "type": r.get("type"), "language": r.get("language"),
                "legibility": r.get("legibility"),
                "gt": (r.get("text") or "").strip(),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    df = pd.DataFrame(rows)
    df["w"] = df.x2 - df.x1
    df["h"] = df.y2 - df.y1
    return pages, df


def balanced_pick(df, n, key="source"):
    groups = [g for _, g in df.groupby(key)]
    if not groups:
        return df.head(0)
    per = n // len(groups)
    chosen = pd.concat([g.head(per) for g in groups])
    if len(chosen) < n:
        chosen = pd.concat([chosen, df.drop(chosen.index).head(n - len(chosen))])
    return chosen.head(n)


def select_lines(allr):
    hw = allr[(allr.type == "handwritten") & (allr.w >= 100) & (allr.h >= 25)
              & (allr.w / allr.h >= 2.0)].copy()
    hw = hw.sample(frac=1, random_state=SEED)

    main = hw[(hw.legibility == "legible") & (hw.language == "uk")
              & hw.source.isin(SOURCES_MAIN) & hw.annotation_source.isin(ANNOT_MAIN)
              & hw["gt"].str.len().between(12, 90)
              & ~hw["gt"].apply(lambda t: any(m in t for m in BAD_MARKERS))]
    main = main.drop_duplicates("gt")
    main = main[main.groupby("page_file").cumcount() < MAX_PER_PAGE]
    main = balanced_pick(main, N_MAIN).assign(subset="main")

    ill = hw[hw.legibility == "illegible"]
    ill = ill[ill.groupby("page_file").cumcount() < MAX_PER_PAGE]
    ill = balanced_pick(ill, N_ILLEG).assign(subset="illeg")

    return pd.concat([main, ill]).reset_index(drop=True)


def crop_line(img, x1, y1, x2, y2):
    W, H = img.size
    box = (max(0, int(x1) - PAD), max(0, int(y1) - PAD),
           min(W, int(x2) + PAD), min(H, int(y2) + PAD))
    return img.crop(box)


def save_crops(sel):
    os.makedirs(OUT_DIR, exist_ok=True)
    page_paths = {}
    for pf in tqdm(sorted(sel.page_file.unique()), desc="Завантаження сторінок"):
        page_paths[pf] = hf_hub_download(REPO, repo_path(pf), repo_type="dataset")

    paths, cw, ch, mism = [], [], [], 0
    counters = {"main": 0, "illeg": 0}
    for r in sel.itertuples():
        img = Image.open(page_paths[r.page_file]).convert("RGB")
        if (r.img_w, r.img_h) != img.size:
            mism += 1
        crop = crop_line(img, r.x1, r.y1, r.x2, r.y2)
        name = f"{r.subset}_{counters[r.subset]:04d}.png"
        counters[r.subset] += 1
        p = os.path.join(OUT_DIR, name)
        crop.save(p)
        paths.append(p); cw.append(crop.width); ch.append(crop.height)
    sel["image_path"], sel["crop_w"], sel["crop_h"] = paths, cw, ch
    return sel, page_paths

def main():
    pages, allr = load_regions()
    print(f"Сторінок у train: {len(pages)}, регіонів: {len(allr)}\n")
    print("Регіони за типом і видимістю:")
    print(allr.groupby(["type", "legibility"]).size().to_string(), "\n")
    print("Рукописні рядки за джерелом / розміткою / видимістю:")
    print(allr[allr.type == "handwritten"]
          .groupby(["source", "annotation_source", "legibility"]).size().to_string(), "\n")

    sel = select_lines(allr)
    print("Відібрано:")
    print(sel.groupby(["subset", "source"]).size().to_string())
    print("сторінок використано:", sel.page_file.nunique(), "\n")

    sel, _ = save_crops(sel)
    cols = ["subset", "image_path", "gt", "legibility", "source", "source_detail",
            "annotation_source", "page_file", "region_idx", "x1", "y1", "x2", "y2",
            "crop_w", "crop_h"]
    sel.insert(0, "id", range(len(sel)))
    sel[["id"] + cols].to_csv(CSV_PATH, index=False, encoding="utf-8")
    sha = hashlib.sha256(open(CSV_PATH, "rb").read()).hexdigest()[:16]
    print(f"\nЗбережено {len(sel)} рядків -> {CSV_PATH} (SHA-256: {sha})")
    print("Розміри обрізків, пікселі:\n", sel[["crop_w", "crop_h"]].describe().round(0).to_string())

if __name__ == "__main__":
    main()
