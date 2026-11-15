import itertools
import os
import re
import unicodedata
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

RAW, OUT, SUM = "results/ocr_raw.csv", "results/metrics.csv", "results/summary.csv"
APOSTROPHES = "’ʼ‘`´"
HALLUC_RATIO = 1.3
HALLUC_CER = 0.5

VARIANT = 3
STUDENT_ID = 901
SEED = int(os.environ.get("SEED", 1000 * VARIANT + STUDENT_ID))

N_BOOT = 2000
CER_LIMIT = 0.10
RISK_LIMIT = 0.02
FAIL_CER = 1.0
HARDER_IS_LOWER = True
TOP_CONF = 15

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

def boot_means(x, rng):
  """Середні N_BOOT бутстреп-вибірок (з поверненням, того ж розміру)."""
  x = np.asarray(x, dtype=float)
  x = x[~np.isnan(x)]
  if len(x) == 0:
    return np.full(1, np.nan)
  return rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)

def ci95(x, rng):
  m = boot_means(x, rng)
  return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))

def summarize(g, rng):
  ok = g.gt_len > 0
  n_gt = int(ok.sum())
  c_lo, c_hi = ci95(g.cer, rng)
  w_lo, w_hi = ci95(g.wer, rng)
  sec = g.sec.mean()
  return {
    "n": len(g), "n_gt": n_gt,
    "cer_mean": g.cer.mean(), "cer_ci_lo": c_lo, "cer_ci_hi": c_hi, "cer_median": g.cer.median(),
    "cer_micro": g.char_edits[ok].sum() / max(1, g.gt_len[ok].sum()),
    "wer_mean": g.wer.mean(), "wer_ci_lo": w_lo, "wer_ci_hi": w_hi, "wer_median": g.wer.median(),
    "wer_micro": g.word_edits[ok].sum() / max(1, g.gt_words[ok].sum()),
    "fail_share": float((g.cer[ok] >= FAIL_CER).mean()) if n_gt else np.nan,
    "empty_share": g.empty_pred.mean(),
    "halluc_len_share": g.halluc_len.mean(), "halluc_cer_share": g.halluc_cer.mean(),
    "sec_mean": sec, "sec_median": g.sec.median(),
    "lines_per_min": 60 / sec if sec > 0 else np.nan}

def ci_table(t, metric):
  t = t.copy()
  t["v"] = [f"{m * 100:.1f} [{lo * 100:.1f}; {hi * 100:.1f}]" if pd.notna(m) else "—"
            for m, lo, hi in zip(t[f"{metric}_mean"], t[f"{metric}_ci_lo"], t[f"{metric}_ci_hi"])]
  p = t.pivot_table(index=["engine", "prep"], columns="level", values="v", aggfunc="first")
  p.columns = [f"{c:g}" for c in p.columns]
  return p

def level_pivot(t, value, scale=100):
  p = t.pivot_table(index=["engine", "prep"], columns="level", values=value) * scale
  p.columns = [f"{c:g}" for c in p.columns]
  return p.round(1)

def base_prep(d):
  return d[d.prep == "raw"] if (d.prep == "raw").any() else d

def robustness(syn):
  lv = sorted(syn.level.dropna().unique(), reverse=HARDER_IS_LOWER)
  m = syn.pivot_table(index=["engine", "prep"], columns="level", values="cer_mean")[lv]
  full = m[m.notna().all(axis=1)]
  rob = []
  for (eng, prep), r in full.iterrows():
    over = [l for l in lv if r[l] > CER_LIMIT]
    rob.append({"engine": eng, "prep": prep, "cer_легкий_рівень": r[lv[0]],
                "рівень_зламу": f"{over[0]:g}" if over else "—", "cer_найважчий_рівень": r[lv[-1]]})
  cross = []
  for prep, sub in full.groupby(level="prep"):
    engines = list(sub.index.get_level_values("engine"))
    for a, b in itertools.combinations(engines, 2):
      sign = np.sign(full.loc[(a, prep)].values - full.loc[(b, prep)].values)
      for i in range(len(lv) - 1):
        if sign[i] * sign[i + 1] < 0:
          cross.append({"prep": prep, "пара": f"{a} vs {b}", "між_рівнями": f"{lv[i]:g} і {lv[i + 1]:g}",
                        "кращий_до": a if sign[i] < 0 else b, "кращий_після": a if sign[i + 1] < 0 else b})
  return pd.DataFrame(rob), pd.DataFrame(cross)

def ablation(syn):
  for eng, g in syn.groupby("engine"):
    cnt = g.groupby("level").prep.nunique()
    if cnt.max() <= 2:
      continue
    t = g[g.level.isin(cnt[cnt == cnt.max()].index)]
    for metric in ("cer", "wer"):
      print(f"=== Абляція {eng}: середній {metric.upper()} (макро), % ===")
      p = t.pivot_table(index="prep", columns="level", values=f"{metric}_mean") * 100
      p.columns = [f"{c:g}" for c in p.columns]
      print(p.round(1).to_string(), "\n")

def paired_bootstrap(d, rng):
  x = d[d.gt_len > 0].copy()
  x["level"] = x.level.fillna(-1)
  idx = ["group", "subset", "prep", "level", "item_id"]
  p = (x.drop_duplicates(idx + ["engine"]).set_index(idx + ["engine"]).cer
       .unstack("engine").reset_index())
  engines = [c for c in p.columns if c not in idx]
  if len(engines) < 2:
    return pd.DataFrame()
  rows = []
  def add(scope, level, sub):
    for a, b in itertools.combinations(engines, 2):
      diff = (sub[a] - sub[b]).dropna()
      if len(diff) < 2:
        continue
      m = boot_means(diff, rng)
      lo, hi = np.quantile(m, [0.025, 0.975])
      rows.append({**scope, "level": level, "a": a, "b": b, "n": len(diff),
                   "diff_mean": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                   "zero_in_ci": bool(lo <= 0 <= hi), "share_a_not_worse": float((m <= 0).mean())})
  for (grp, prep, sub_name, lvl), sub in p.groupby(["group", "prep", "subset", "level"]):
    add({"group": grp, "prep": prep, "subset": sub_name}, "—" if lvl == -1 else lvl, sub)
  for (grp, prep), sub in p.groupby(["group", "prep"]):
    add({"group": grp, "prep": prep, "subset": "ALL"}, "ALL", sub)
  return pd.DataFrame(rows)

def print_paired(pb):
  t = pb.copy()
  t["mark"] = ""
  syn_lv = t[(t.group == "syn") & (t.level != "ALL")]
  for _, g in syn_lv.groupby(["prep", "a", "b"]):
    t.loc[g.diff_mean.abs().idxmax(), "mark"] = "◄ макс. різниця"
  t[["diff_mean", "ci_lo", "ci_hi"]] *= 100
  t = t.rename(columns={"diff_mean": "різн_%(a−b)", "ci_lo": "ДІ_низ_%", "ci_hi": "ДІ_верх_%",
                        "zero_in_ci": "ДІ_містить_0"})
  print("=== Парна бутстреп-різниця CER (a − b), 95% ДІ, % ===")
  print(t.drop(columns="share_a_not_worse").to_string(index=False), "\n")

def confidence(d):
  x = d[(d.gt_len > 0) & d.conf.notna()]
  rows = []
  for (grp, eng, prep), g in x.groupby(["group", "engine", "prep"]):
    if len(g) < 5:
      continue
    rho, pval = spearmanr(g.conf, g.cer)
    g = g.sort_values("conf", ascending=False, kind="mergesort")
    c, e = g.conf.values, g.cer.values
    k = np.arange(1, len(g) + 1)
    risk = np.cumsum(e) / k
    tie_end = np.append(c[1:] != c[:-1], True)
    ok = tie_end & (risk <= RISK_LIMIT)
    if ok.any():
      i = np.flatnonzero(ok)[-1]
      thr, cov, rsk = c[i], k[i] / len(g), risk[i]
    else:
      thr, cov, rsk = np.nan, 0.0, np.nan
    rows.append({"group": grp, "engine": eng, "prep": prep, "n": len(g), "spearman_rho": rho,
                 "p_value": pval, "cer_all": e.mean(), f"поріг_conf(CER≤{RISK_LIMIT:.0%})": thr,
                 "частка_авто": cov, "cer_прийнятих": rsk})
  return pd.DataFrame(rows)

def pareto(d):
  rows = []
  x = d[d.gt_len > 0]
  for grp, g in x.groupby("group"):
    if grp == "syn":
      nl = g.groupby(["engine", "prep"])["level"].nunique()
      keep = set(nl[nl == nl.max()].index)
      g = g[[(e, p) in keep for e, p in zip(g.engine, g.prep)]]
    a = g.groupby(["engine", "prep"]).agg(cer_mean=("cer", "mean"), wer_mean=("wer", "mean"),
                                          sec_mean=("sec", "mean")).reset_index()
    a["lines_per_min"] = 60 / a.sec_mean
    dom = []
    for i, r in a.iterrows():
      o = a.drop(i)
      dom.append(bool(((o.cer_mean <= r.cer_mean) & (o.sec_mean <= r.sec_mean)
                       & ((o.cer_mean < r.cer_mean) | (o.sec_mean < r.sec_mean))).any()))
    a["статус"] = np.where(dom, "домінується", "фронт Парето")
    a.insert(0, "group", grp)
    rows.append(a)
  return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def sym(c):
  return "␣" if c == " " else c

def confusions(gt, pred):
  pairs = []
  for op in Levenshtein.editops(gt, pred):
    if op.tag == "replace":
      pairs.append(f"{sym(gt[op.src_pos])} → {sym(pred[op.dest_pos])}")
    elif op.tag == "delete":
      pairs.append(f"{sym(gt[op.src_pos])} → ∅")
    else:
      pairs.append(f"∅ → {sym(pred[op.dest_pos])}")
  return pairs

def confusion_table(d):
  x = base_prep(d)
  x = x[(x.gt_len > 0) & (x.cer < 0.5)]
  rows = []
  for (grp, eng), g in x.groupby(["group", "engine"]):
    cnt = Counter()
    for gt, pred in zip(g.gt_n, g.pred_n):
      cnt.update(confusions(gt, pred))
    for pair, k in cnt.most_common(TOP_CONF):
      rows.append({"group": grp, "engine": eng, "заміна": pair, "кількість": k})
  return pd.DataFrame(rows)

def worst5(d):
  x = base_prep(d)
  x = x[x.gt_len > 0]
  cols = ["group", "engine", "prep", "subset", "item_id", "level", "cer", "wer", "len_ratio", "gt", "pred"]
  parts = [g.sort_values(["cer", "gt_len"], ascending=False).head(5)[cols]
           for _, g in x.groupby(["group", "engine"])]
  return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def group_compare(d):
  x = base_prep(d)
  x = x[x.gt_len > 0]
  syn = x[x.group == "syn"]
  clean = (syn.level.max() if HARDER_IS_LOWER else syn.level.min()) if len(syn) else np.nan
  rows = []
  for eng, g in x.groupby("engine"):
    for metric in ("cer", "wer"):
      def m(sel):
        return g[sel][metric].mean() if sel.any() else np.nan
      r = {"engine": eng, "metric": metric.upper(),
           "syn_чисті": m((g.group == "syn") & (g.level == clean)), "syn_усі_рівні": m(g.group == "syn"),
           "rukopys": m(g.group == "rukopys"), "own": m(g.group == "own")}
      r["own/syn_чисті"] = r["own"] / r["syn_чисті"] if r["syn_чисті"] > 0 else np.nan
      r["own/rukopys"] = r["own"] / r["rukopys"] if r["rukopys"] > 0 else np.nan
      rows.append(r)
  return pd.DataFrame(rows)

def main():
  rng = np.random.default_rng(SEED)
  print(f"VARIANT={VARIANT}, SEED={SEED}, N_BOOT={N_BOOT}  (SEED має збігатися з підписом на графіках)")
  if STUDENT_ID == 0 and "SEED" not in os.environ:
    print("УВАГА: STUDENT_ID = 0 -- задайте свої останні 3 цифри залікової книжки\n")
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

  keys = ["group", "engine", "prep", "subset", "level"]
  s = pd.DataFrame([{**dict(zip(keys, key)), **summarize(g, rng)}
                    for key, g in d.groupby(keys, dropna=False)])
  s.to_csv(SUM, index=False, encoding="utf-8")
  saved = [OUT, SUM]

  pd.options.display.float_format = "{:.3f}".format
  print("Усереднення: «макро» = середнє CER/WER окремих рядків, «мікро» = сума помилок / сума довжин еталонів.")
  print("Нормалізація: NFC, апострофи -> ', послідовності пробілів -> один; регістр і пунктуація зберігаються.\n")

  syn = s[s.group == "syn"]
  if len(syn):
    print("=== Синтетика: середній CER (макро) за рівнем контрасту, % ===")
    print(level_pivot(syn, "cer_mean").to_string(), "\n")
    print("=== Синтетика: CER, % -- середнє [95% бутстреп-ДІ] ===")
    print(ci_table(syn, "cer").to_string(), "\n")
    print("=== Синтетика: середній WER (макро) за рівнем контрасту, % ===")
    print(level_pivot(syn, "wer_mean").to_string(), "\n")
    print("=== Синтетика: WER, % -- середнє [95% бутстреп-ДІ] ===")
    print(ci_table(syn, "wer").to_string(), "\n")
    print("=== Синтетика: CER (мікро), % ===")
    print(level_pivot(syn, "cer_micro").to_string(), "\n")
    print("=== Синтетика: медіана CER, % та частка повних провалів (CER ≥ 1), % ===")
    print(level_pivot(syn, "cer_median").to_string(), "\n")
    print(level_pivot(syn, "fail_share").to_string(), "\n")
    rob, cross = robustness(syn)
    print(f"=== Крива стійкості: плато, точка зламу (CER > {CER_LIMIT:.0%}) ===")
    print(rob.to_string(index=False), "\n")
    print("=== Перетини кривих (зміна ранжування інструментів) ===")
    print(cross.to_string(index=False) if len(cross) else "перетинів немає", "\n")
    rob.to_csv("results/robustness.csv", index=False, encoding="utf-8")
    cross.to_csv("results/curve_crossings.csv", index=False, encoding="utf-8")
    saved += ["results/robustness.csv", "results/curve_crossings.csv"]
    ablation(syn)

  ruk = s[s.group == "rukopys"]
  if len(ruk):
    print("=== RUKOPYS (CER/WER у частках; ДІ -- 95% бутстреп) ===")
    print(ruk[["engine", "prep", "subset", "n", "cer_mean", "cer_ci_lo", "cer_ci_hi", "cer_median", "cer_micro",
               "wer_mean", "wer_ci_lo", "wer_ci_hi", "wer_median", "wer_micro", "fail_share", "empty_share",
               "sec_mean", "lines_per_min"]].to_string(index=False), "\n")
  own = d[d.group == "own"]
  if len(own):
    print("=== Власні фото: CER ===")
    print(own.pivot_table(index="item_id", columns=["engine", "prep"], values="cer").to_string(), "\n")
    print("=== Власні фото: WER ===")
    print(own.pivot_table(index="item_id", columns=["engine", "prep"], values="wer").to_string(), "\n")
    ob = base_prep(own)
    worst = ob.loc[ob.groupby("engine").cer.idxmax(), ["engine", "item_id", "cer", "wer"]]
    print("=== Власні фото: найгірше зображення для кожного інструмента ===")
    print(worst.to_string(index=False), "\n")

  pb = paired_bootstrap(d, rng)
  if len(pb):
    print_paired(pb)
    pb.to_csv("results/paired_bootstrap.csv", index=False, encoding="utf-8")
    saved.append("results/paired_bootstrap.csv")

  cf = confidence(d)
  if len(cf):
    print(f"=== Впевненість: Спірмен ρ(conf, CER) і поріг для CER прийнятих ≤ {RISK_LIMIT:.0%} ===")
    print(cf.to_string(index=False), "\n")
    cf.to_csv("results/confidence.csv", index=False, encoding="utf-8")
    saved.append("results/confidence.csv")
  else:
    print("Впевненість: жоден інструмент не повертає conf (ρ і ризик-покриття не будуються).\n")

  pt = pareto(d)
  if len(pt):
    print("=== Компроміс «час-CER» (середнє за всіма рівнями/зразками групи) ===")
    print(pt.to_string(index=False), "\n")
    pt.to_csv("results/pareto.csv", index=False, encoding="utf-8")
    saved.append("results/pareto.csv")

  gc = group_compare(d)
  if len(gc):
    print("=== Порівняння груп даних (конфігурація raw) ===")
    print(gc.to_string(index=False), "\n")
    gc.to_csv("results/group_compare.csv", index=False, encoding="utf-8")
    saved.append("results/group_compare.csv")

  ct = confusion_table(d)
  if len(ct):
    print(f"=== Найчастіші заміни символів (топ-{TOP_CONF} збережено; тут топ-10) ===")
    print(ct.groupby(["group", "engine"]).head(10).to_string(index=False), "\n")
    ct.to_csv("results/confusions.csv", index=False, encoding="utf-8")
    saved.append("results/confusions.csv")

  w5 = worst5(d)
  if len(w5):
    shown = w5.copy()
    shown["gt"], shown["pred"] = shown["gt"].str[:40], shown["pred"].str[:40]
    print("=== П'ять найгірших зразків (group × engine) ===")
    print(shown.to_string(index=False), "\n")
    w5.to_csv("results/worst5.csv", index=False, encoding="utf-8")
    saved.append("results/worst5.csv")

  v = d[d.engine == "V"]
  if len(v) and v.attempts.notna().any():
    print(f"VLM: запусків із повторами (attempt>1): {int((v.attempts > 1).sum())} з {len(v)}; "
          f"відповідей без done_reason=stop: {int((v.reason != 'stop').sum())}")
  if len(v):
    print("VLM: відношення довжини відповіді до еталону (len_ratio), медіана / частка > "
          f"{HALLUC_RATIO}:")
    print(v[v.gt_len > 0].groupby("group").agg(медіана=("len_ratio", "median"),
                                                частка_галюцинацій=("halluc_len", "mean")).to_string())
  il = d[(d.group == "rukopys") & (d.subset == "illeg")]
  if len(il):
    print("\n=== Особливість варіанта: нерозбірливі рядки ===")
    print(il.groupby("engine").agg(n=("cer", "size"), порожніх_gt=("gt_len", lambda x: int((x == 0).sum())),
                                   порожніх_відповідей=("empty_pred", "sum"),
                                   halluc_len=("halluc_len", "sum"),
                                   halluc_cer=("halluc_cer", "sum")).to_string())
  print("\nЗбережено: " + ", ".join(saved))

if __name__ == "__main__":
  main()
