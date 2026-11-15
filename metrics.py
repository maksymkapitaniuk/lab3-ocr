import os
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

os.chdir(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

RAW, OUT, SUM = "results/ocr_raw.csv", "results/metrics.csv", "results/summary.csv"
APOSTROPHES = "\u2019\u02bc\u2018\u0060\u00b4"
HALLUC_RATIO = 1.3
HALLUC_CER = 0.5

def normalize(s):
  s = unicodedata.normalize("NFC", str(s))
  for a in APOSTROPHES:
    s = s.replace(a, "'")
  return re.sub(r"\s+", " ", s).strip()

def check_completeness(d):
  print("=== Перевірка повноти ===")
  key = ["group", "item_id", "level", "engine", "prep"]
  print("дублікатів ключа:", int(d.duplicated(key).sum()))
  exp = {}
  if os.path.exists("data/ground_truth.csv"):
    g = pd.read_csv("data/ground_truth.csv")
    n, lv = g.id.nunique(), set(g.level.unique())
    n_abl = len(lv & {100, 25, 15})
    exp[("syn", "E")] = n * (2 * len(lv) + 4 * n_abl)
    exp[("syn", "V")] = n * len(lv)
  if os.path.exists("data/rukopys_labels.csv"):
    r = pd.read_csv("data/rukopys_labels.csv", keep_default_na=False)
    exp[("rukopys", "E")] = exp[("rukopys", "V")] = len(r)
  photos = {os.path.splitext(f)[0] for f in os.listdir("my_photos")
            if f.lower().endswith((".jpg", ".jpeg"))} if os.path.isdir("my_photos") else set()
  if photos:
    exp[("own", "E")] = exp[("own", "V")] = 2 * len(photos)
  rows = []
  for (grp, eng), e in sorted(exp.items()):
    a = int(((d.group == grp) & (d.engine == eng)).sum())
    rows.append({"group": grp, "engine": eng, "очікується": e, "є": a,
                 "статус": "OK" if a == e else "ПЕРЕВІРТЕ"})
  print(pd.DataFrame(rows).to_string(index=False))
  print()

def main():
  d = pd.read_csv(RAW, dtype=str, keep_default_na=False)
  check_completeness(d)

  d["gt_n"] = d["gt"].map(normalize)
  d["pred_n"] = d.pred.map(normalize)
  d["gt_len"] = d.gt_n.str.len()
  d["pred_len"] = d.pred_n.str.len()
  d["gt_words"] = d.gt_n.str.split().str.len()
  d["char_edits"] = [Levenshtein.distance(g, p) for g, p in zip(d.gt_n, d.pred_n)]
  d["word_edits"] = [Levenshtein.distance(g.split(), p.split()) for g, p in zip(d.gt_n, d.pred_n)]

  has_gt = d.gt_len > 0
  d["cer"] = np.where(has_gt, d.char_edits / d.gt_len.clip(lower=1), np.nan)
  d["wer"] = np.where(has_gt, d.word_edits / d.gt_words.clip(lower=1), np.nan)
  d["len_ratio"] = np.where(has_gt, d.pred_len / d.gt_len.clip(lower=1), np.nan)
  d["empty_pred"] = d.pred_len == 0

  d["level"] = pd.to_numeric(d.level, errors="coerce")
  d["conf"] = pd.to_numeric(d.conf, errors="coerce")
  d["sec"] = pd.to_numeric(d.sec, errors="coerce")
  d["reason"] = d.done_reason.str.split("|").str[0]
  d["attempts"] = pd.to_numeric(d.done_reason.str.extract(r"attempt=(\d+)")[0], errors="coerce")

  d["halluc_len"] = (has_gt & (d.len_ratio > HALLUC_RATIO)) | (~has_gt & ~d.empty_pred)
  d["halluc_cer"] = (has_gt & (d.cer > HALLUC_CER)) | (~has_gt & ~d.empty_pred)

  d.drop(columns=["gt_n", "pred_n"]).to_csv(OUT, index=False, encoding="utf-8")

  d["_e"] = d.char_edits.where(has_gt)
  d["_l"] = d.gt_len.where(has_gt)
  keys = ["group", "engine", "prep", "subset", "level"]
  s = d.groupby(keys, dropna=False).agg(
    n=("cer", "size"), cer_mean=("cer", "mean"), cer_median=("cer", "median"),
    _e=("_e", "sum"), _l=("_l", "sum"), wer_mean=("wer", "mean"),
    empty_share=("empty_pred", "mean"), sec_mean=("sec", "mean")).reset_index()
  s["cer_micro"] = s._e / s._l.clip(lower=1)
  s = s.drop(columns=["_e", "_l"])
  s.to_csv(SUM, index=False, encoding="utf-8")

  pd.options.display.float_format = "{:.3f}".format
  syn = s[s.group == "syn"]
  if len(syn):
    print("=== Синтетика: середній CER (макро) за рівнем контрасту, % ===")
    print((syn.pivot_table(index=["engine", "prep"], columns="level", values="cer_mean") * 100)
          .round(1).to_string(), "\n")
  ruk = s[s.group == "rukopys"]
  if len(ruk):
    print("=== RUKOPYS ===")
    print(ruk[["engine", "subset", "n", "cer_mean", "cer_median", "wer_mean",
               "empty_share", "sec_mean"]].to_string(index=False), "\n")
  own = d[d.group == "own"]
  if len(own):
    print("=== Власні фото: CER ===")
    print(own.pivot_table(index="item_id", columns=["engine", "prep"], values="cer").to_string(), "\n")

  v = d[d.engine == "V"]
  if len(v) and v.attempts.notna().any():
    print(f"VLM: запусків із повторами (attempt>1): {int((v.attempts > 1).sum())} з {len(v)}; "
          f"відповідей без done_reason=stop: {int((v.reason != 'stop').sum())}")
  il = d[(d.group == "rukopys") & (d.subset == "illeg")]
  if len(il):
    print("\n=== Особливість варіанта: нерозбірливі рядки ===")
    print(il.groupby("engine").agg(n=("cer", "size"), порожніх_gt=("gt_len", lambda x: int((x == 0).sum())),
                                   порожніх_відповідей=("empty_pred", "sum"),
                                   halluc_len=("halluc_len", "sum"),
                                   halluc_cer=("halluc_cer", "sum")).to_string())
  print(f"\nЗбережено: {OUT}, {SUM}")

if __name__ == "__main__":
  main()
