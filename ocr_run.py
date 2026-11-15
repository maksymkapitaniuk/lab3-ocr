import argparse
import csv
import glob
import io
import json
import os
import platform
import sys
import time
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ExifTags
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 1000 * 3 + 901
VLM_MODEL = "qwen2.5vl:7b"
PROMPT = ("Перепиши дослівно весь текст із зображення, зберігаючи рядки. "
          "Нічого не виправляй, не перекладай і не додавай пояснень.")
VLM_NUM_PREDICT = {"line": 300, "page": 1500}
VLM_NUM_CTX = 8192
VLM_RETRIES = 3
PAGE_LONG_SIDE = {"E": 1600, "V": 1600}

OUT_CSV = "results/ocr_raw.csv"
RUN_INFO = "results/run_info.json"
FIELDS = ["group", "item_id", "level", "engine", "prep", "subset", "legibility",
          "gt", "pred", "conf", "sec", "done_reason", "taken", "camera"]

LEVELS_ALL = [100, 70, 50, 35, 25, 15]
ABL_LEVELS = [100, 25, 15]

PREPS = {
  "raw": (),
  "stretch": ("stretch",),
  "clahe": ("clahe",),
  "upscale": ("upscale",),
  "stretch+upscale": ("stretch", "upscale"),
  "stretch+otsu": ("stretch", "binarize"),
}
MAIN_PREPS = ["raw", "stretch"]
ABL_PREPS = ["clahe", "upscale", "stretch+upscale", "stretch+otsu"]

def preprocess(img, steps=(), page=False):
  out = img
  for st in steps:
    if st == "stretch":
      lo, hi = (np.percentile(out, (1, 99)) if page else (float(out.min()), float(out.max())))
      if hi - lo >= 1:
        out = np.clip((out.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    elif st == "clahe":
      out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)
    elif st == "upscale":
      out = cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    elif st == "binarize":
      out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif st == "adaptive":
      out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 15)
    else:
      raise ValueError(f"невідомий крок обробки: {st}")
  return out


def load_gray(path, page=False, long_side=2000):
  arr = np.array(ImageOps.exif_transpose(Image.open(path)).convert("L"))
  if page:
    k = long_side / max(arr.shape)
    arr = cv2.resize(arr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
  return arr

def exif_info(path):
  exif = Image.open(path).getexif()
  tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
  return str(tags.get("DateTime", "")), str(tags.get("Model", ""))

_reader = None
_client = None

def get_reader():
  global _reader
  if _reader is None:
    import easyocr, torch
    _reader = easyocr.Reader(["uk", "en"], gpu=torch.cuda.is_available(), verbose=False)
  return _reader

def get_client():
  global _client
  if _client is None:
    import ollama
    _client = ollama.Client(timeout=900)
  return _client

def group_into_lines(res):
  items = []
  for box, text, conf in res:
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    items.append({"x": min(xs), "yc": (min(ys) + max(ys)) / 2,
                  "h": max(ys) - min(ys), "text": text})
  if not items:
    return []
  med_h = float(np.median([i["h"] for i in items]))
  items.sort(key=lambda i: i["yc"])
  rows, cur = [], [items[0]]
  for it in items[1:]:
    if it["yc"] - np.mean([c["yc"] for c in cur]) < 0.6 * med_h:
      cur.append(it)
    else:
      rows.append(cur)
      cur = [it]
  rows.append(cur)
  return [" ".join(i["text"] for i in sorted(r, key=lambda i: i["x"])) for r in rows]

def ocr_easyocr(img, kind="line"):
  t0 = time.perf_counter()
  res = get_reader().readtext(img, detail=1, canvas_size=1600)
  text = "\n".join(group_into_lines(res))
  conf = float(np.mean([r[2] for r in res])) if res else 0.0
  return text, conf, time.perf_counter() - t0, ""

def ocr_vlm(img, kind="line"):
  buf = io.BytesIO()
  Image.fromarray(img).save(buf, format="PNG")
  data = buf.getvalue()
  options = {"temperature": 0, "seed": SEED, "num_predict": VLM_NUM_PREDICT[kind]}
  if VLM_NUM_CTX:
    options["num_ctx"] = VLM_NUM_CTX
  for attempt in range(1, VLM_RETRIES + 1):
    t0 = time.perf_counter()
    try:
      resp = get_client().chat(
        model=VLM_MODEL, keep_alive="30m", options=options,
        messages=[{"role": "user", "content": PROMPT, "images": [data]}])
    except Exception as e:
      tqdm.write(f"  помилка Ollama (спроба {attempt}/{VLM_RETRIES}): {e}")
      time.sleep(10)
      continue
    sec = time.perf_counter() - t0
    text = resp["message"]["content"].strip()
    reason = str(getattr(resp, "done_reason", "") or "")
    if text or reason == "stop":
      return text, "", sec, f"{reason}|attempt={attempt}"
    tqdm.write(f"  порожня відповідь, done_reason={reason!r}, {sec:.1f} с, спроба {attempt}/{VLM_RETRIES}")
    time.sleep(10)
  raise RuntimeError("VLM повертає порожню відповідь без done_reason=stop")

ENGINES = {"E": ocr_easyocr, "V": ocr_vlm}

def preflight_vlm():
  try:
    names = [m.model for m in get_client().list().models]
  except Exception as e:
    sys.exit(f"Ollama недоступна ({e}). Запустіть застосунок Ollama і повторіть.")
  if VLM_MODEL not in names:
    sys.exit(f"Модель {VLM_MODEL} не знайдено. Виконайте: ollama pull {VLM_MODEL}")

def warm_up(engine):
  blank = np.full((64, 256), 255, np.uint8)
  try:
    ENGINES[engine](blank)
  except Exception as e:
    print(f"(прогрів {engine}: {e})")

def build_tasks(groups, engines, limit, vlm_prep):
  tasks = []

  def add(item, engine, preps):
    for p in preps:
      tasks.append({**item, "engine": engine, "prep": p})

  if "own" in groups:
    photos = sorted({os.path.normpath(p) for ext in ("jpg", "jpeg", "JPG")
                     for p in glob.glob(f"my_photos/*.{ext}")})
    for p in photos:
      txt = os.path.splitext(p)[0] + ".txt"
      if not os.path.exists(txt):
        sys.exit(f"Немає транскрипції {txt}")
      taken, camera = exif_info(p)
      item = {"group": "own", "item_id": os.path.splitext(os.path.basename(p))[0],
              "level": "", "path": p, "gt": open(txt, encoding="utf-8").read().strip(),
              "subset": "", "legibility": "", "page": True, "taken": taken, "camera": camera}
      add(item, "E", ["raw", "stretch"])
      add(item, "V", ["raw", "stretch"])

  if "rukopys" in groups:
    df = pd.read_csv("data/rukopys_labels.csv", keep_default_na=False)
    if limit:
      df = df.groupby("subset").head(limit)
    for r in df.itertuples():
      item = {"group": "rukopys", "item_id": os.path.splitext(os.path.basename(r.image_path))[0],
              "level": "", "path": r.image_path, "gt": r.gt, "subset": r.subset,
              "legibility": r.legibility, "page": False, "taken": "", "camera": ""}
      add(item, "E", ["raw"])
      add(item, "V", ["raw"])

  if "syn" in groups:
    df = pd.read_csv("data/ground_truth.csv")
    if limit:
      df = df[df.id < limit]
    for r in df.itertuples():
      item = {"group": "syn", "item_id": f"{r.id:03d}", "level": str(int(r.level)),
              "path": r.path, "gt": r.gt, "subset": "", "legibility": "", "page": False,
              "taken": "", "camera": ""}
      add(item, "E", MAIN_PREPS + (ABL_PREPS if int(r.level) in ABL_LEVELS else []))
      add(item, "V", ["raw"] + (["stretch"] if vlm_prep else []))

  order_e = {"E": 0, "V": 1}
  order_g = {"own": 0, "rukopys": 1, "syn": 2}
  tasks = [t for t in tasks if t["engine"] in engines]
  tasks.sort(key=lambda t: (order_e[t["engine"]], order_g[t["group"]], t["item_id"],
                            int(t["level"] or 0), t["prep"]))
  return tasks

def task_key(t):
  return (t["group"], str(t["item_id"]), str(t["level"]), t["engine"], t["prep"])

def load_done():
  if not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0:
    return set()
  d = pd.read_csv(OUT_CSV, dtype=str, keep_default_na=False)
  return set(zip(d.group, d.item_id, d.level, d.engine, d.prep))

def write_run_info(engines):
  def ver(p):
    try:
      return version(p)
    except PackageNotFoundError:
      return None
  info = {"started": datetime.now().isoformat(timespec="seconds"),
          "python": sys.version.split()[0], "platform": platform.platform(),
          "seed": SEED, "engines": engines, "prompt": PROMPT,
          "vlm_model": VLM_MODEL, "vlm_num_predict": VLM_NUM_PREDICT,
          "vlm_num_ctx": VLM_NUM_CTX,
          "preps": {k: list(v) for k, v in PREPS.items()},
          "packages": {p: ver(p) for p in ["easyocr", "torch", "ollama", "opencv-python",
                                           "pillow", "numpy", "pandas"]}}
  try:
    import torch
    info["cuda"] = torch.cuda.is_available()
  except Exception:
    pass
  if "V" in engines:
    try:
      for m in get_client().list().models:
        if m.model == VLM_MODEL:
          info["vlm_digest"] = m.digest
    except Exception:
      pass
  runs = []
  if os.path.exists(RUN_INFO):
    runs = json.load(open(RUN_INFO, encoding="utf-8")).get("runs", [])
  runs.append(info)
  json.dump({"runs": runs}, open(RUN_INFO, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--group", default="all", choices=["all", "syn", "rukopys", "own"])
  ap.add_argument("--engine", default="all", choices=["all", "E", "V"])
  ap.add_argument("--limit", type=int, default=0, help="димовий тест: перші N рядків (на групу/підмножину)")
  ap.add_argument("--vlm-prep", action="store_true", help="додатково V + stretch на синтетиці (удвічі довше)")
  a = ap.parse_args()

  groups = ["own", "rukopys", "syn"] if a.group == "all" else [a.group]
  engines = ["E", "V"] if a.engine == "all" else [a.engine]
  os.makedirs("results", exist_ok=True)

  tasks = build_tasks(groups, engines, a.limit, a.vlm_prep)
  done = load_done()
  todo = [t for t in tasks if task_key(t) not in done]
  print(f"Усього завдань: {len(tasks)} | уже в кеші: {len(tasks) - len(todo)} | до виконання: {len(todo)}")
  for e in engines:
    print(f"   {e}: {sum(t['engine'] == e for t in todo)}")
  if not todo:
    return

  used = sorted({t["engine"] for t in todo})
  if "V" in used:
    preflight_vlm()
  write_run_info(used)
  for e in used:
    print(f"Прогрів {e}...")
    warm_up(e)

  new_file = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
  failed = 0
  with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, FIELDS)
    if new_file:
      w.writeheader()
    bar = tqdm(todo, unit="зобр")
    for t in bar:
      try:
        img = preprocess(load_gray(t["path"], t["page"], PAGE_LONG_SIDE[t["engine"]]), PREPS[t["prep"]], page=t["page"])
        text, conf, sec, reason = ENGINES[t["engine"]](img, "page" if t["page"] else "line")
      except Exception as e:
        failed += 1
        tqdm.write(f"ПОМИЛКА {task_key(t)}: {e}")
        continue
      w.writerow({"group": t["group"], "item_id": t["item_id"], "level": t["level"],
                  "engine": t["engine"], "prep": t["prep"], "subset": t["subset"],
                  "legibility": t["legibility"], "gt": t["gt"], "pred": text,
                  "conf": conf if conf == "" else round(conf, 4), "sec": round(sec, 3),
                  "done_reason": reason, "taken": t["taken"], "camera": t["camera"]})
      f.flush()
      bar.set_postfix(engine=t["engine"], sec=f"{sec:.1f}")
  print(f"Готово. Помилок: {failed}. Результати: {OUT_CSV}")

if __name__ == "__main__":
    main()
