"""Generate every figure, table and headline number of the paper from results/analysis/summary.json.

Nothing in main.tex is typed by hand: numbers arrive through ``numbers.tex`` macros, tables through
``tables/*.tex`` and figures through ``figures/*.pdf``. Re-run after any change in the results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
S = json.loads((ROOT / "results" / "analysis" / "summary.json").read_text())
MAN = json.loads((ROOT / "app" / "public" / "data" / "manifest.json").read_text())
T = S["titles"]
DS = {"nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15", "cicids2017": "CIC-IDS2017"}
ENS = {"ens_all": "Ensemble (all)", "ens_diverse": "Ensemble (diverse 5)"}
COL = {"ae": "#5b4bd6", "vae": "#9a8cf0", "knn": "#0e9fb8", "iforest": "#1f9d6b", "gmm": "#d99a1e",
       "pca": "#3b82f6", "ecod": "#e8742a", "deep_svdd": "#d6409f", "flow": "#e0455a", "neutral": "#7aa31c"}  # fmt: skip

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight"})  # fmt: skip


def title(d: str) -> str:
    if d in ENS:
        return ENS[d]
    base, _, rest = d.partition("+")
    return T.get(base, base) + (f" +{rest}" if rest else "")


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("β", r"$\beta$")


macros: dict[str, str] = {}


def macro(name: str, value: str) -> None:
    macros[name] = value


# ------------------------------------------------------------------------------ main table
main = S.get("main")
if main:
    rows = [r for r in main["table"] if not r["detector"].startswith("ens_") or True]
    dets = sorted({r["detector"] for r in rows}, key=lambda d: (d.startswith("ens_"), -np.mean([r["roc"] for r in rows if r["detector"] == d and r["regime"] == "clean"] or [0])))  # fmt: skip
    cols = [(d, g) for g in ("clean", "natural") for d in DS if any(r["dataset"] == d and r["regime"] == g for r in rows)]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{1}}{{c}}{{{DS[d][:7]}-{g[:1].upper()}}}" for d, g in cols) + r" \\", r"\midrule"]  # fmt: skip
    best = {c: max((r["roc"] for r in rows if (r["dataset"], r["regime"]) == c and not r["detector"].startswith("ens_")), default=0) for c in cols}  # fmt: skip
    for d in dets:
        cells = []
        for c in cols:
            r = next((r for r in rows if r["detector"] == d and (r["dataset"], r["regime"]) == c), None)
            if r is None:
                cells.append("--")
                continue
            v = f"{r['roc']:.3f}"
            cells.append(rf"\textbf{{{v}}}" if abs(r["roc"] - best[c]) < 1e-9 else v)
        lines.append(tex(title(d)) + " & " + " & ".join(cells) + r" \\")
        if d == dets[-3] if len(dets) > 2 else False:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "main.tex").write_text("\n".join(lines))

    st = main["stats"].get("all_seeds")
    if st:
        macro("friedmanChi", f"{st['statistic']:.1f}")
        macro("friedmanBlocks", str(st["n_blocks"]))
        macro("cdValue", f"{st['cd']:.2f}")
        macro("bestDetector", tex(title(main["stats"]["best_by_rank"])))
        # CD diagram
        ranks = sorted(st["mean_ranks"].items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(ranks) / 2 + 1.1))
        lo, hi = np.floor(ranks[0][1]), np.ceil(ranks[-1][1])
        ax.set_xlim(lo - 0.3, hi + 0.3)
        ax.set_ylim(-len(ranks) / 2 - 2.2, 1.6)
        ax.axis("off")
        ax.plot([lo, hi], [0, 0], "k", lw=0.8)
        for t in range(int(lo), int(hi) + 1):
            ax.plot([t, t], [0, 0.12], "k", lw=0.8)
            ax.text(t, 0.3, str(t), ha="center", fontsize=7)
        ax.plot([lo, lo + st["cd"]], [1.1, 1.1], "#d99a1e", lw=1.6)
        ax.text(lo + st["cd"] / 2, 1.25, f"CD = {st['cd']:.2f}", ha="center", fontsize=7, color="#a56f00")
        half = (len(ranks) + 1) // 2
        for i, (d, r) in enumerate(ranks):
            left = i < half
            y = -1.2 - (i if left else len(ranks) - 1 - i) * 0.9 / 1.0 * 0.5
            x_text = lo - 0.2 if left else hi + 0.2
            ax.plot([r, r, x_text], [0, y, y], color="0.5", lw=0.5)
            ax.text(x_text + (-0.05 if left else 0.05), y, f"{title(d)} ({r:.1f})", ha="right" if left else "left", va="center", fontsize=7)  # fmt: skip
        for gi, g in enumerate(st["cd_groups"]):
            rs = [st["mean_ranks"][m] for m in g]
            ax.plot([min(rs) - 0.03, max(rs) + 0.03], [-len(ranks) / 2 - 1.2 - gi * 0.18] * 2, color="#5b4bd6", lw=2.2, solid_capstyle="round")  # fmt: skip
        fig.savefig(OUT / "figures" / "cd.pdf")
        plt.close(fig)
    singles = [r for r in main["table"] if r["regime"] == "clean" and not r["detector"].startswith("ens_")]
    if singles:
        b = max(singles, key=lambda r: r["roc"])
        macro("bestCleanAuc", f"{b['roc']:.3f}")
    ens = [r for r in main["table"] if r["detector"] == "ens_all" and r["regime"] == "clean"]
    if ens:
        macro("ensFprOne", f"{100 * np.mean([r['a1_fpr'] for r in ens]):.2f}")
    sing_fpr = [r["a1_fpr"] for r in singles]
    if sing_fpr:
        macro("singleFprOne", f"{100 * np.mean(sing_fpr):.2f}")

# ------------------------------------------------------------------------- contamination
c = S.get("contamination")
if c:
    dsets = [d for d in DS if any(r["dataset"] == d for r in c["table"])]
    fig, axes = plt.subplots(1, len(dsets), figsize=(2.4 * len(dsets) + 0.6, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dsets, strict=True):
        for det in ("ae", "knn", "iforest", "flow"):
            for suffix, ls in (("", "-"), ("+ctr0.1", "--")):
                rs = sorted((r for r in c["table"] if r["dataset"] == d and r["detector"] == det + suffix), key=lambda r: r["rho"])  # fmt: skip
                if rs:
                    ax.plot([100 * r["rho"] for r in rs], [r["roc"] for r in rs], ls, color=COL[det], lw=1.4,
                            label=title(det) + (" + CTR" if suffix else "") if d == dsets[0] else None)  # fmt: skip
        ax.set_title(DS[d])
        ax.set_xlabel(r"contamination $\rho$ (\%)")
        ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    axes[0].set_ylabel("ROC-AUC")
    fig.legend(loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(OUT / "figures" / "contamination.pdf")
    plt.close(fig)

    gains = [g for g in c["ctr_gain"] if abs(g["rho"] - 0.05) < 1e-9 and g["detector"].endswith("+ctr0.1")]
    if gains:
        macro("ctrGainFive", f"{np.mean([g['gain'] for g in gains]):+.3f}")
        macro("ctrGainFivePts", f"{100 * np.mean([g['gain'] for g in gains]):+.1f}")
        pos = [g for g in gains if g["gain"] > 0]
        macro("ctrWinFraction", f"{len(pos)}/{len(gains)}")
    deg = c["degradation"]
    if deg:
        macro("meanDropFivePts", f"{100 * np.mean([d['drop'] for d in deg]):.1f}")
        worst = max(deg, key=lambda d: d["drop"])
        macro("worstDropPts", f"{100 * worst['drop']:.0f}")
        macro("worstDropName", tex(title(worst["detector"])))
    def roc_at(ds: str, det: str, rho: float) -> float | None:
        return next((r["roc"] for r in c["table"] if r["dataset"] == ds and r["detector"] == det and abs(r["rho"] - rho) < 1e-9), None)

    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Dataset & Detector & $\rho{=}0$ & $\rho{=}5\%$ & $\rho{=}5\%$ + CTR & $\Delta$ \\", r"\midrule"]  # fmt: skip
    for d in dsets:
        for det in ("ae", "knn", "flow", "gmm"):
            a, b, cc = roc_at(d, det, 0.0), roc_at(d, det, 0.05), roc_at(d, det + "+ctr0.1", 0.05)
            if None in (a, b, cc):
                continue
            lines.append(rf"{DS[d]} & {tex(title(det))} & {a:.3f} & {b:.3f} & {cc:.3f} & {cc - b:+.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "contamination.tex").write_text("\n".join(lines))
    abl_ids = ["ae", "ae+ctr0.05", "ae+ctr0.1", "ae+ctr0.2", "ae+ctr0.1[iforest]", "ae+ctr0.1[ecod]", "ae+ctr0.1[pca]", "ae+ctr@rho"]
    names = {"ae": "none", "ae+ctr0.05": "CTR, trim 5\\%", "ae+ctr0.1": "CTR, trim 10\\% (committee)", "ae+ctr0.2": "CTR, trim 20\\%", "ae+ctr0.1[iforest]": "screener: IForest only", "ae+ctr0.1[ecod]": "screener: ECOD only", "ae+ctr0.1[pca]": "screener: PCA only", "ae+ctr@rho": "oracle trim $=\\rho$"}  # fmt: skip
    lines = [r"\begin{tabular}{l" + "r" * len(dsets) + "}", r"\toprule", "Purification of the AE & " + " & ".join(DS[d] for d in dsets) + r" \\", r"\midrule"]  # fmt: skip
    for i in abl_ids:
        vals = [next((r["roc"] for r in c["table"] if r["dataset"] == d and r["detector"] == i and abs(r["rho"] - 0.05) < 1e-9), None) for d in dsets]  # fmt: skip
        if any(v is not None for v in vals):
            lines.append(names[i] + " & " + " & ".join("--" if v is None else f"{v:.3f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "ablation.tex").write_text("\n".join(lines))

# ------------------------------------------------------------------------------------ drift
d = S.get("drift")
if d:
    days = ["Tue", "Wed", "Thu", "Fri"]
    dets = sorted({r["detector"] for r in d["table"] if not r["detector"].startswith("ens_")}, key=lambda x: np.mean([r["fpr_a1"] for r in d["table"] if r["detector"] == x]))  # fmt: skip
    M = np.array([[next((r["fpr_a1"] for r in d["table"] if r["detector"] == x and r["day"] == day), np.nan) for day in days] for x in dets])  # fmt: skip
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1, 1]})
    im = a1.imshow(np.log10(np.clip(M, 2e-3, 1)), cmap="magma_r", aspect="auto", vmin=-2.7, vmax=0)
    a1.set_xticks(range(4), days)
    a1.set_yticks(range(len(dets)), [title(x) for x in dets], fontsize=6.5)
    a1.grid(False)
    for i in range(M.shape[0]):
        for j in range(4):
            a1.text(j, i, f"{100 * M[i, j]:.1f}", ha="center", va="center", fontsize=5.8, color="white" if M[i, j] > 0.05 else "black")  # fmt: skip
    a1.set_title(r"realized FPR (\%), nominal 1\%", fontsize=8)
    pts = [(r["psi_benign"], r["fpr_a1"], r["detector"]) for r in d["table"] if not r["detector"].startswith("ens_") and r["fpr_a1"] > 0]  # fmt: skip
    a2.scatter([max(p[0], 1e-3) for p in pts], [p[1] for p in pts], s=9, c=[COL.get(p[2], "0.4") for p in pts], alpha=0.8)
    a2.set(xscale="log", yscale="log", xlabel="PSI of benign scores (no labels)", ylabel="realized FPR")
    a2.axvline(0.25, color="0.4", ls="--", lw=0.8)
    a2.axhline(0.03, color="0.4", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "drift.pdf")
    plt.close(fig)
    if d["psi_fpr_spearman"]:
        macro("psiSpearman", f"{d['psi_fpr_spearman']['rho']:.2f}")
    al = d["drift_alarm"]
    macro("alarmRecall", "n/a" if al["recall"] is None else f"{100 * al['recall']:.0f}")
    macro("alarmPrecision", "n/a" if al["precision"] is None else f"{100 * al['precision']:.0f}")
    fri = [r["fpr_a1"] for r in d["table"] if r["day"] == "Fri" and not r["detector"].startswith("ens_")]
    tue = [r["fpr_a1"] for r in d["table"] if r["day"] == "Tue" and not r["detector"].startswith("ens_")]
    macro("driftFprTue", f"{100 * np.mean(tue):.1f}")
    macro("driftFprFri", f"{100 * np.mean(fri):.1f}")

# ---------------------------------------------------------------------------------- transfer
t = S.get("transfer")
if t:
    cfg = [("unsw_nb15", "unsw_nb15", "source", "U$\\to$U"), ("unsw_nb15", "cicids2017", "source", "U$\\to$C"), ("unsw_nb15", "cicids2017", "target", "U$\\to$C dom."),
           ("cicids2017", "cicids2017", "source", "C$\\to$C"), ("cicids2017", "unsw_nb15", "source", "C$\\to$U"), ("cicids2017", "unsw_nb15", "target", "C$\\to$U dom.")]  # fmt: skip
    dets = sorted({r["detector"] for r in t["table"] if not r["detector"].startswith("ens_")}, key=lambda x: -np.mean([r["roc"] for r in t["table"] if r["detector"] == x]))  # fmt: skip
    M = np.array([[next((r["roc"] for r in t["table"] if r["detector"] == x and (r["source"], r["target"], r["scaler"]) == c[:3]), np.nan) for c in cfg] for x in dets])  # fmt: skip
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.imshow(M, cmap="viridis", vmin=0.3, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(cfg)), [c[3] for c in cfg], fontsize=7)
    ax.set_yticks(range(len(dets)), [title(x) for x in dets], fontsize=6.5)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] < 0.65 else "black")
    fig.savefig(OUT / "figures" / "transfer.pdf")
    plt.close(fig)
    for key, (s, tg, sc, _) in zip(("UU", "UC", "UCd", "CC", "CU", "CUd"), cfg, strict=True):
        vals = [r["roc"] for r in t["table"] if (r["source"], r["target"], r["scaler"]) == (s, tg, sc) and not r["detector"].startswith("ens_")]  # fmt: skip
        fpr = [r["a1_fpr"] for r in t["table"] if (r["source"], r["target"], r["scaler"]) == (s, tg, sc) and not r["detector"].startswith("ens_")]  # fmt: skip
        if vals:
            macro(f"transfer{key}", f"{np.mean(vals):.3f}")
            macro(f"transferFpr{key}", f"{100 * np.mean(fpr):.0f}")

# ------------------------------------------------------------------------------- reliability
if main and (main.get("coverage") or d):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), sharey=True)
    for ax, det in zip(axes, ("ae", "iforest", "ens_all"), strict=True):
        ds = next((k for k in DS if any(r["dataset"] == k and r["detector"] == det for r in main["coverage"])), None)
        if ds:
            rs = sorted((r for r in main["coverage"] if r["dataset"] == ds and r["regime"] == "clean" and r["detector"] == det), key=lambda r: r["alpha"])  # fmt: skip
            ax.plot([r["alpha"] for r in rs], [r["fpr"] for r in rs], color="#1f9d6b", label=f"in-dataset ({DS[ds]})")
        if d:
            for day, colr in (("Tue", "#d99a1e"), ("Fri", "#e0455a")):
                rs = sorted((r for r in d["coverage"] if r["detector"] == det and r["day"] == day), key=lambda r: r["alpha"])
                if rs:
                    ax.plot([r["alpha"] for r in rs], [r["fpr"] for r in rs], color=colr, label=f"CIC {day}")
        ax.plot([0, 0.2], [0, 0.2], "k--", lw=0.7)
        ax.set_title(title(det), fontsize=8)
        ax.set_xlabel(r"nominal $\alpha$")
    axes[0].set_ylabel("realized FPR")
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "reliability.pdf")
    plt.close(fig)

# ------------------------------------------------------------------------------- explainability
fig, axes = plt.subplots(1, len(MAN["datasets"]), figsize=(2.3 * len(MAN["datasets"]), 2.2), sharey=True)
axes = np.atleast_1d(axes)
for ax, m in zip(axes, MAN["datasets"], strict=True):
    dl = m["explain"]["deletion"]
    ax.plot(dl["k"], dl["attributed"], "-o", ms=3, color="#5b4bd6", label="top-$k$ attributed")
    ax.plot(dl["k"], dl["random"], "-o", ms=3, color="0.55", label="random $k$")
    ax.set_title(m["title"], fontsize=8)
    ax.set_xlabel("features deleted $k$")
    nmi = m["explain"]["attribution_cluster_nmi"]
    tag = {"nsl_kdd": "NSL", "unsw_nb15": "UNSW", "cicids2017": "CIC"}[m["key"]]  # macro names: letters only
    macro(f"nmi{tag}", "n/a" if nmi is None else f"{nmi:.2f}")
    macro(f"delThree{tag}", f"{100 * dl['attributed'][2]:.0f}")
    macro(f"delThreeRand{tag}", f"{100 * dl['random'][2]:.0f}")
axes[0].set_ylabel("score removed")
axes[0].legend(fontsize=6, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "figures" / "explain.pdf")
plt.close(fig)

macro("nDetectors", "15")
macro("nFlows", f"{sum(m['n_test'] for m in MAN['datasets']):,}".replace(",", r"\,"))
(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print("wrote", len(macros), "macros")
