import itertools
import os
import re
import textwrap
import traceback
import unicodedata
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein
from scipy.stats import spearmanr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STUDENT, VARIANT, STUDENT_ID = "Капітанюк М.А.", 3, 901
SEED = 1000 * VARIANT + STUDENT_ID
rng = np.random.default_rng(SEED)

MET, FIG = "results/metrics.csv", "figures"
THRESH_ROBUST = 0.10          # допустимий CER для кривої стійкості
RISK_CER = 0.02               # допустимий CER для кривої ризик-покриття
ABL_LEVELS = [100, 25, 15]
N_BOOT = 2000
ENG = {"E": "EasyOCR", "V": "Qwen2.5-VL"}
APOSTROPHES = "\u2019\u02bc\u2018\u0060\u00b4"
_HOMO_SRC = "аa еe іi оo рp сc уy хx кk АA ВB ЕE ІI КK МM НH ОO РP СC ТT ХX"
HOMO = {p for w in _HOMO_SRC.split() for p in ((w[0], w[1]), (w[1], w[0]))}

LOG = []
d = syn = ruk = own = None
L_MAX = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def stamp(ax, title):
    ax.set_title(f"{title}\n{STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)


def lab(eng, prep):
    return f"{ENG[eng]} / {prep}"


def normalize(s):
    s = unicodedata.normalize("NFC", str(s))
    for a in APOSTROPHES:
        s = s.replace(a, "'")
    return re.sub(r"\s+", " ", s).strip()


def boot_ci(x, n_boot=N_BOOT, alpha=0.05):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def ratio_txt(a, b):
    return f"×{b/a:.1f}" if a > 0 else "не визначено (у розбірливих 0 %)"


def describe(name, x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        log(f"{name}: немає даних")
        return
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    log(f"{name}: n={len(x)}, середнє {x.mean()*100:.1f} %, медіана {med*100:.1f} %, "
        f"Q1-Q3 {q1*100:.1f}-{q3*100:.1f} %, макс {x.max()*100:.1f} %, частка CER>=1: {np.mean(x >= 1)*100:.1f} %")


def image_path(r):
    if r.group == "syn":
        return f"data/synthetic/{r.item_id}_contrast_{int(r.level)}.png"
    if r.group == "rukopys":
        return f"data/rukopys/{r.item_id}.png"
    return f"my_photos/{r.item_id}.jpg"


def load():
    x = pd.read_csv(MET, dtype={"item_id": str})
    for c in ["gt", "pred", "subset", "legibility", "done_reason", "reason", "taken", "camera"]:
        if c in x:
            x[c] = x[c].fillna("")
    x["level"] = pd.to_numeric(x["level"], errors="coerce")
    return x


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Рис. 1
def fig1():
    log("\n=== Рис. 1. Крива стійкості (синтетика) ===")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    series = {}
    for eng, prep in [("E", "raw"), ("E", "stretch"), ("V", "raw")]:
        g = syn[(syn.engine == eng) & (syn.prep == prep)]
        rows = []
        for l in sorted(g.level.dropna().unique(), reverse=True):
            x = g[g.level == l].cer.values
            lo, hi = boot_ci(x)
            rows.append((l, np.nanmean(x), lo, hi))
        t = pd.DataFrame(rows, columns=["level", "mean", "lo", "hi"])
        series[(eng, prep)] = t
        ax.plot(t.level, t["mean"], marker="o", label=lab(eng, prep))
        ax.fill_between(t.level, t.lo, t.hi, alpha=0.15)
        over = t[t["mean"] > THRESH_ROBUST]
        msg = (f"перевищує {THRESH_ROBUST*100:.0f} % уперше на рівні {over.level.iloc[0]:.0f} %"
               if len(over) else f"поріг {THRESH_ROBUST*100:.0f} % не перевищено")
        log(f"{lab(eng, prep)}: CER {t['mean'].min()*100:.1f}-{t['mean'].max()*100:.1f} %; {msg}")
        log("   за рівнями (100->15): " + ", ".join(f"{m*100:.1f}" for m in t["mean"]))
        w = t.hi - t.lo
        j = w.idxmax()
        log(f"   найширший 95% ДІ: {w.max()*100:.2f} п.п. на рівні {t.level[j]:.0f} % "
            f"[{t.lo[j]*100:.1f}; {t.hi[j]*100:.1f}] %; найвужчий: {w.min()*100:.2f} п.п.")
    for a, b in itertools.combinations(series, 2):
        m = series[a].merge(series[b], on="level", suffixes=("_a", "_b"))
        s = np.sign(m.mean_a - m.mean_b).values
        cross = [(int(m.level[i]), int(m.level[i + 1])) for i in range(len(s) - 1) if s[i] * s[i + 1] < 0]
        log(f"перетин {lab(*a)} і {lab(*b)}: " + (f"між рівнями {cross}" if cross else "немає"))
    ax.axhline(THRESH_ROBUST, ls="--", c="grey", lw=1)
    ax.text(0.99, THRESH_ROBUST * 1.02, f"поріг CER = {THRESH_ROBUST*100:.0f} %", color="grey",
            fontsize=8, ha="right", transform=ax.get_yaxis_transform())
    ax.invert_xaxis()
    ax.set_xticks(sorted(syn.level.dropna().unique(), reverse=True))
    ax.set_xlabel("контраст тексту і тла, % (праворуч гірше)")
    ax.set_ylabel("CER (середнє, 95% ДІ)")
    ax.set_ylim(bottom=0)
    ax.legend()
    stamp(ax, "Крива стійкості до втрати контрасту")
    save(fig, "fig1_robustness.png")


# ---------------------------------------------------------------- Рис. 2
def box_panel(ax, groups, labels):
    keep = [(g[~np.isnan(g)], l) for g, l in zip(groups, labels) if len(g[~np.isnan(g)])]
    ax.boxplot([g for g, _ in keep], showmeans=True, widths=0.5)
    ax.set_xticks(range(1, len(keep) + 1))
    ax.set_xticklabels([l for _, l in keep], fontsize=8)
    for k, (g, _) in enumerate(keep, 1):
        ax.scatter(np.full(len(g), k) + rng.uniform(-0.12, 0.12, len(g)), g, s=8, alpha=0.4)
    return keep


def fig2():
    log(f"\n=== Рис. 2. Діаграма розмаху (синтетика на рівні {L_MAX:.0f} %, RUKOPYS) ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    cfg = [("E", "raw"), ("E", "stretch"), ("V", "raw")]
    groups = [syn[(syn.engine == e) & (syn.prep == p) & (syn.level == L_MAX)].cer.values for e, p in cfg]
    labels = [lab(e, p) for e, p in cfg]
    box_panel(axes[0], groups, labels)
    for g, l in zip(groups, labels):
        describe(f"синтетика {L_MAX:.0f} %, {l}", g)
    axes[0].set_ylabel("CER окремого рядка")
    stamp(axes[0], f"Синтетика, контраст {L_MAX:.0f} %")

    cfg2 = [("E", "main"), ("V", "main"), ("E", "illeg"), ("V", "illeg")]
    groups2 = [ruk[(ruk.engine == e) & (ruk.subset == s)].cer.values for e, s in cfg2]
    labels2 = [f"{ENG[e]}\n{'розбірливі' if s == 'main' else 'нерозбірливі'}" for e, s in cfg2]
    box_panel(axes[1], groups2, labels2)
    allv = np.concatenate([g[~np.isnan(g)] for g in groups2])
    top = max(1.2, np.percentile(allv, 99) * 1.1)
    axes[1].set_ylim(0, top)
    log(f"(на RUKOPYS вісь обрізано на {top:.2f}; точок вище межі осі: {int((allv > top).sum())})")
    for g, (e, s) in zip(groups2, cfg2):
        describe(f"RUKOPYS {s}, {ENG[e]}", g)
    axes[1].set_ylabel("CER окремого рядка")
    stamp(axes[1], "RUKOPYS (без деградації)")
    save(fig, "fig2_boxplot.png")


# ---------------------------------------------------------------- Рис. 3
def fig3():
    log("\n=== Рис. 3. Впевненість і ризик-покриття (лише EasyOCR; у VLM немає впевненості) ===")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    sets = {"синтетика (raw)": syn[(syn.engine == "E") & (syn.prep == "raw")],
            "RUKOPYS розбірливі": ruk[(ruk.engine == "E") & (ruk.subset == "main")]}
    for name, g in sets.items():
        g = g.dropna(subset=["conf", "cer"])
        rho, p = spearmanr(g.conf, g.cer)
        axes[0].scatter(g.conf, g.cer, s=10, alpha=0.5, label=f"{name}: ρ = {rho:.2f}")
        g = g.sort_values("conf", ascending=False)
        cov = np.arange(1, len(g) + 1) / len(g)
        risk = g.cer.cumsum().values / np.arange(1, len(g) + 1)
        axes[1].plot(cov, risk, label=name)
        log(f"{name}: n={len(g)}, Спірмен ρ = {rho:.2f} (p = {p:.3g}); середня впевненість {g.conf.mean():.2f}")
        ok = np.where(risk <= RISK_CER)[0]
        if len(ok):
            i = ok[-1]
            log(f"   при середньому CER прийнятих <= {RISK_CER*100:.0f} %: покриття до {cov[i]*100:.0f} % рядків, "
                f"поріг впевненості ≈ {g.conf.iloc[i]:.2f}")
        else:
            log(f"   жоден поріг впевненості не дає середній CER <= {RISK_CER*100:.0f} % "
                f"(мінімум на кривій {risk.min()*100:.1f} %)")
    for ax in axes:
        ax.set_yscale("symlog", linthresh=0.05)
        ax.legend(fontsize=8)
    axes[0].set_xlabel("впевненість EasyOCR")
    axes[0].set_ylabel("CER (symlog)")
    stamp(axes[0], "Впевненість проти помилки")
    axes[1].axhline(RISK_CER, ls="--", c="grey", lw=1)
    axes[1].set_xlabel("частка автоматично прийнятих рядків (coverage)")
    axes[1].set_ylabel("середній CER прийнятих (symlog)")
    stamp(axes[1], "Крива ризик-покриття")
    save(fig, "fig3_confidence.png")


# ---------------------------------------------------------------- Рис. 4
def fig4():
    log(f"\n=== Рис. 4. Час-CER (синтетика, рівні {ABL_LEVELS}, де є всі конфігурації) ===")
    sub = syn[syn.level.isin(ABL_LEVELS)]
    agg = (sub.groupby(["engine", "prep"])
           .agg(cer=("cer", "mean"), sec=("sec", "mean"), sec_std=("sec", "std")).reset_index())
    pts = list(zip(agg.sec, agg.cer))
    agg["front"] = [not any((t2 <= t and c2 <= c) and (t2 < t or c2 < c)
                            for j, (t2, c2) in enumerate(pts) if j != i) for i, (t, c) in enumerate(pts)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for r in agg.itertuples():
        ax.scatter(r.sec * 1000, r.cer, s=60, c="tab:green" if r.front else "tab:grey")
        ax.annotate(lab(r.engine, r.prep), (r.sec * 1000, r.cer), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
        log(f"{lab(r.engine, r.prep)}: CER {r.cer*100:.2f} %, час {r.sec*1000:.0f} ± {r.sec_std*1000:.0f} мс; "
            + ("на фронті Парето" if r.front else "домінується"))
    ax.set_xscale("log")
    ax.set_xlabel("середній час на рядок, мс (log)")
    ax.set_ylabel("середній CER")
    stamp(ax, "Компроміс якість-швидкість (зелені: фронт Парето)")
    save(fig, "fig4_pareto.png")


# ---------------------------------------------------------------- Рис. 5
def fig5():
    log(f"\n=== Рис. 5. Абляція попередньої обробки (EasyOCR, синтетика) ===")
    order = ["raw", "stretch", "clahe", "upscale", "stretch+upscale", "stretch+otsu"]
    g = syn[(syn.engine == "E") & syn.level.isin(ABL_LEVELS)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.26
    tab = {}
    for k, l in enumerate(ABL_LEVELS):
        ms, lo, hi = [], [], []
        for p in order:
            x = g[(g.prep == p) & (g.level == l)].cer.values
            m = np.nanmean(x) if len(x) else np.nan
            a, b = boot_ci(x)
            ms.append(m); lo.append(a); hi.append(b)
        ms, lo, hi = map(np.array, (ms, lo, hi))
        ax.bar(np.arange(len(order)) + (k - 1) * w, ms, w, label=f"контраст {int(l)} %",
               yerr=[ms - lo, hi - ms], capsize=2)
        tab[int(l)] = ms * 100
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel("середній CER (EasyOCR), 95% ДІ")
    ax.legend()
    stamp(ax, "Внесок кроків попередньої обробки")
    save(fig, "fig5_ablation.png")
    t = pd.DataFrame(tab, index=order).round(2)
    log("середній CER, %:\n" + t.to_string())
    for l in t.columns:
        dlt = (t[l] - t.loc["raw", l]).drop("raw")
        log(f"рівень {l} %: найкраще {dlt.idxmin()} ({dlt.min():+.2f} п.п. до raw), "
            f"найслабший ефект {dlt.idxmax()} ({dlt.max():+.2f} п.п.)")


# ---------------------------------------------------------------- Рис. 6
def confusions(gt, pred):
    gt, pred = normalize(gt), normalize(pred)
    ops = []
    for op in Levenshtein.editops(gt, pred):
        if op.tag == "replace":
            ops.append((gt[op.src_pos], pred[op.dest_pos]))
        elif op.tag == "delete":
            ops.append((gt[op.src_pos], "∅"))
        else:
            ops.append(("∅", pred[op.dest_pos]))
    return ops


def sym(c):
    return "␣" if c == " " else c


def fig6():
    log("\n=== Рис. 6. Заміни символів (рядки з CER < 0.5) ===")
    panels = [("EasyOCR, синтетика (raw)", syn[(syn.engine == "E") & (syn.prep == "raw")]),
              ("Qwen2.5-VL, синтетика (raw)", syn[(syn.engine == "V") & (syn.prep == "raw")]),
              ("Qwen2.5-VL, RUKOPYS розбірливі", ruk[(ruk.engine == "V") & (ruk.subset == "main")])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, g) in zip(axes, panels):
        g = g[(g.cer < 0.5) & (g.gt_len > 0)]
        cnt = Counter()
        for r in g.itertuples():
            cnt.update(confusions(r.gt, r.pred))
        top = cnt.most_common(12)
        total = sum(cnt.values())
        homo = sum(v for (a, b), v in cnt.items() if (a, b) in HOMO)
        log(f"{name}: рядків {len(g)}, операцій редагування {total}, з них гомогліфи (кирилиця↔латиниця): {homo}")
        log("   топ-10: " + "; ".join(f"{sym(a)}→{sym(b)} ×{v}" for (a, b), v in top[:10]))
        if top:
            ax.barh([f"{sym(a)} → {sym(b)}" for (a, b), _ in top][::-1], [v for _, v in top][::-1])
        ax.set_xlabel("кількість")
        stamp(ax, name)
    save(fig, "fig6_confusions.png")


# ---------------------------------------------------------------- Рис. 7
def fig7():
    log("\n=== Рис. 7. CER на трьох групах даних (raw) ===")
    groups = [("синтетика\n(контраст 100 %)", syn[syn.level == 100]),
              ("RUKOPYS\n(розбірливі)", ruk[ruk.subset == "main"]),
              ("власні фото\n(сторінки)", own)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    w = 0.36
    means = {}
    for k, eng in enumerate(["E", "V"]):
        for i, (name, g) in enumerate(groups):
            x = g[(g.engine == eng) & (g.prep == "raw")].cer.dropna().values
            m = x.mean()
            means[(eng, i)] = m
            pos = i + (k - 0.5) * w
            ax.bar(pos, max(m, 1e-3), w, color=f"C{k}", label=ENG[eng] if i == 0 else None)
            if len(x) >= 10:
                lo, hi = boot_ci(x)
                ax.errorbar(pos, max(m, 1e-3), yerr=[[max(m - lo, 0)], [hi - m]], c="k", capsize=3)
            else:
                ax.scatter(np.full(len(x), pos), np.maximum(x, 1e-3), c="k", s=18, zorder=3)
            ax.text(pos, max(m, 1e-3) * 1.15, f"{m*100:.1f}", ha="center", fontsize=8)
            log(f"{ENG[eng]}, {name.replace(chr(10), ' ')}: середній CER {m*100:.1f} % (n={len(x)})")
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([n for n, _ in groups])
    ax.set_ylabel("середній CER (log), % біля стовпців")
    ax.legend()
    stamp(ax, "Порівняння груп даних")
    save(fig, "fig7_groups.png")
    for eng in ["E", "V"]:
        s, r, o = means[(eng, 0)], means[(eng, 1)], means[(eng, 2)]
        log(f"{ENG[eng]}: власні/синтетика = ×{o/max(s,1e-9):.1f}; RUKOPYS/синтетика = ×{r/max(s,1e-9):.1f}; "
            f"власні/RUKOPYS = ×{o/max(r,1e-9):.2f}")
    log("CER кожної сторінки (власні фото):\n" +
        own.pivot_table(index="item_id", columns=["engine", "prep"], values="cer").round(3).to_string())


# ---------------------------------------------------------------- Рис. 8
def fig8():
    log("\n=== Рис. 8. Особливість варіанта: нерозбірливі рядки RUKOPYS ===")
    r = ruk.copy()
    r["cat"] = np.where(r.subset == "main", "розбірливі",
                        np.where(r.gt_len > 0, "нерозбірливі,\ngt непорожній", "нерозбірливі,\ngt порожній"))
    cats = ["розбірливі", "нерозбірливі,\ngt непорожній", "нерозбірливі,\ngt порожній"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, eng in zip(axes[:2], ["V", "E"]):
        g = r[r.engine == eng]
        x = np.arange(len(cats))
        s1 = [g[g.cat == c].halluc_len.mean() if (g.cat == c).any() else np.nan for c in cats]
        s2 = [g[g.cat == c].halluc_cer.mean() if (g.cat == c).any() else np.nan for c in cats]
        ax.bar(x - 0.2, s1, 0.4, label="довжина: len(pred)/len(gt) > 1.3")
        ax.bar(x + 0.2, s2, 0.4, label="зміст: CER > 0.5")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={int((g.cat == c).sum())})" for c in cats], fontsize=8)
        ax.set_ylim(0, 1.3)
        ax.set_ylabel("частка кандидатів у галюцинацію")
        ax.legend(fontsize=7, loc="upper left")
        stamp(ax, f"{ENG[eng]}")
        for c, a, b in zip(cats, s1, s2):
            log(f"{ENG[eng]}, {c.replace(chr(10), ' ')}: n={int((g.cat == c).sum())}, "
                f"галюцинація за довжиною {a*100:.0f} %, за змістом {b*100:.0f} %")
        m, il = g[g.subset == "main"], g[g.subset == "illeg"]
        a, b = m.halluc_len.mean(), il.halluc_len.mean()
        log(f"{ENG[eng]}: H10 (за довжиною): розбірливі {a*100:.1f} %, нерозбірливі {b*100:.1f} % -> {ratio_txt(a, b)}")
        a2, b2 = m.halluc_cer.mean(), il.halluc_cer.mean()
        log(f"{ENG[eng]}: розширений критерій (CER>0.5): розбірливі {a2*100:.1f} %, нерозбірливі {b2*100:.1f} % -> {ratio_txt(a2, b2)}")
    e = r[r.cat == cats[2]]
    ax = axes[2]
    vals = [e[e.engine == k].pred_len.values for k in ["V", "E"]]
    ax.boxplot(vals, showmeans=True, widths=0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([ENG["V"], ENG["E"]])
    for k, v in enumerate(vals, 1):
        ax.scatter(np.full(len(v), k) + rng.uniform(-0.12, 0.12, len(v)), v, s=10, alpha=0.5)
        log(f"{ENG['V' if k == 1 else 'E']}: для {len(v)} рядків із порожнім еталоном видано "
            f"{int((v > 0).sum())} непорожніх відповідей; довжина відповіді в середньому {v.mean():.0f} симв., макс {v.max():.0f}")
    ax.set_ylabel("символів у відповіді при порожньому еталоні")
    stamp(ax, "Що пишуть, коли людина нічого не прочитала")
    save(fig, "fig8_legibility.png")

    v = r[(r.engine == "V") & (r.halluc_len | r.halluc_cer)].copy()
    v["image_path"] = [image_path(x) for x in v.itertuples()]
    v = v.sort_values(["subset", "cer"], ascending=[True, False])
    cols = ["item_id", "subset", "legibility", "gt", "pred", "gt_len", "pred_len", "len_ratio",
            "cer", "halluc_len", "halluc_cer", "image_path"]
    v[cols].to_csv("results/halluc_cases_V.csv", index=False, encoding="utf-8")
    log(f"список випадків (V) збережено: results/halluc_cases_V.csv ({len(v)} рядків)")


# ---------------------------------------------------------------- Рис. 9
def fig9():
    log("\n=== Рис. 9. П'ять найгірших зразків (raw, усі групи) ===")
    out = []
    for eng in ["E", "V"]:
        g = d[(d.engine == eng) & (d.prep == "raw")].dropna(subset=["cer"]).nlargest(5, "cer")
        fig, axes = plt.subplots(5, 1, figsize=(11, 12))
        for ax, r in zip(axes, g.itertuples()):
            p = image_path(r)
            ax.axis("off")
            if os.path.exists(p):
                im = ImageOps.exif_transpose(Image.open(p)).convert("L")
                im.thumbnail((1000, 260))
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            gt, pr = textwrap.shorten(normalize(r.gt), 120), textwrap.shorten(normalize(r.pred), 120)
            ax.set_title(f"{r.group}/{r.item_id}" + (f"/{int(r.level)} %" if r.group == "syn" else "") +
                         f"  CER={r.cer:.2f}  відношення довжин={r.len_ratio:.2f}\n"
                         f"еталон: {gt}\nрезультат: {pr}", fontsize=7, loc="left")
            log(f"{ENG[eng]} {r.group}/{r.item_id}: CER {r.cer:.3f}, довжини gt/pred = {r.gt_len}/{r.pred_len}")
            out.append({"engine": eng, "group": r.group, "item_id": r.item_id, "level": r.level,
                        "cer": r.cer, "wer": r.wer, "len_ratio": r.len_ratio, "gt": r.gt,
                        "pred": r.pred, "image_path": p})
        fig.suptitle(f"П'ять найгірших зразків, {ENG[eng]}. {STUDENT}, варіант {VARIANT}, seed={SEED}", fontsize=10)
        fig.tight_layout()
        save(fig, f"fig9_worst5_{eng}.png")
    pd.DataFrame(out).to_csv("results/worst5.csv", index=False, encoding="utf-8")


# ---------------------------------------------------------------- Рис. 10
def fig10():
    log(f"\n=== Рис. 10. Бутстреп парної різниці E - V (синтетика, рівень {L_MAX:.0f} %, raw) ===")
    a = syn[(syn.engine == "E") & (syn.prep == "raw") & (syn.level == L_MAX)].set_index("item_id").cer
    b = syn[(syn.engine == "V") & (syn.prep == "raw") & (syn.level == L_MAX)].set_index("item_id").cer
    m = pd.concat([a, b], axis=1, keys=["E", "V"]).dropna()
    n = len(m)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    mE, mV = m.E.values[idx].mean(1), m.V.values[idx].mean(1)
    diff = mE - mV
    lo, hi = np.quantile(diff, [0.025, 0.975])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(mE, bins=30, alpha=0.6, label=f"{ENG['E']}: [{np.quantile(mE, .025):.3f}; {np.quantile(mE, .975):.3f}]")
    axes[0].hist(mV, bins=30, alpha=0.6, label=f"{ENG['V']}: [{np.quantile(mV, .025):.3f}; {np.quantile(mV, .975):.3f}]")
    axes[0].set_xlabel("середній CER у бутстреп-вибірці")
    axes[0].set_ylabel("кількість вибірок")
    axes[0].legend(fontsize=8)
    stamp(axes[0], "Окремі інтервали")
    axes[1].hist(diff, bins=30, color="C4", alpha=0.8)
    axes[1].axvline(0, c="k")
    for v in (lo, hi):
        axes[1].axvline(v, c="C4", ls="--")
    axes[1].set_xlabel("різниця середніх CER (EasyOCR − Qwen2.5-VL)")
    stamp(axes[1], f"Парна різниця: 95% ДІ [{lo:.3f}; {hi:.3f}]")
    save(fig, "fig10_bootstrap_diff.png")
    log(f"n={n} рядків; середня різниця {m.E.mean()-m.V.mean():.4f}; 95% ДІ [{lo:.4f}; {hi:.4f}]; "
        f"{'містить' if lo <= 0 <= hi else 'не містить'} нуль; частка вибірок, де EasyOCR не гірший: {np.mean(diff <= 0)*100:.1f} %")
    log("Для питання 2: ширина ДІ зменшується приблизно як 1/√n, тож для вдвічі вужчого інтервалу "
        f"потрібно приблизно в 4 рази більше рядків (замість {n}: близько {4*n}).")


def run(fn):
    try:
        fn()
    except Exception:
        log(f"!!! {fn.__name__} не вдалося:\n{traceback.format_exc()}")


def main():
    global d, syn, ruk, own, L_MAX
    d = load()
    syn, ruk, own = d[d.group == "syn"], d[d.group == "rukopys"], d[d.group == "own"]
    piv = syn[syn.prep == "raw"].groupby(["level", "engine"]).cer.mean().unstack()
    L_MAX = float((piv["E"] - piv["V"]).abs().idxmax())
    log(f"Рівень контрасту з найбільшою різницею E і V (raw): {L_MAX:.0f} %")
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10):
        run(fn)
    with open("results/plots_numbers.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    print(f"\nГотово. Рисунки: {FIG}/, числа для інтерпретації: results/plots_numbers.txt")


if __name__ == "__main__":
    main()