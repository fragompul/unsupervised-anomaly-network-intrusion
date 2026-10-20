"""Generate every figure, table and headline number of the paper from results/analysis/summary.json.

Nothing in main.tex is typed by hand: numbers arrive through ``numbers.tex`` macros, tables through
``tables/*.tex`` and figures through ``figures/*.pdf``. Re-run after any change in the results.

Figures are drawn at the physical width of an IEEE two-column page (3.5 in column, 7.16 in page) so
that text inside them is rendered at its final size (7 pt and up), with an Okabe-Ito palette that stays
legible for colour-blind readers and in greyscale.
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
(OUT / "figures").mkdir(exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
S = json.loads((ROOT / "results" / "analysis" / "summary.json").read_text())
MAN = json.loads((ROOT / "app" / "public" / "data" / "manifest.json").read_text())
T = S["titles"]
DS = {"nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15", "cicids2017": "CIC-IDS2017"}
DS_SHORT = {"nsl_kdd": "NSL", "unsw_nb15": "UNSW", "cicids2017": "CIC"}
ENS = {"ens_all": "Ensemble (all 15)", "ens_diverse": "Ensemble (diverse 5)"}
COLW, PAGEW = 3.5, 7.16  # IEEE column and page widths in inches

# Okabe-Ito palette
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#6b7280",
)  # fmt: skip
COL = {"ae": BLUE, "knn": RED, "iforest": GREEN, "flow": PURPLE, "gmm": ORANGE, "vae": SKY}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "lines.linewidth": 1.3,
})  # fmt: skip


def title(d: str) -> str:
    if d in ENS:
        return ENS[d]
    base, _, rest = d.partition("+")
    return T.get(base, base) + (f" +{rest}" if rest else "")


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("β", r"$\beta$")


def panel(ax, letter: str) -> None:
    ax.text(-0.02, 1.07, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


macros: dict[str, str] = {}


def macro(name: str, value: str) -> None:
    macros[name] = value


# ------------------------------------------------------------------------------ main table
main = S.get("main")
if main:
    rows = main["table"]
    st = main["stats"].get("all_seeds")
    ranks = st["mean_ranks"] if st else {}
    dets = sorted(
        {r["detector"] for r in rows},
        key=lambda d: (d.startswith("ens_"), -np.mean([r["roc"] for r in rows if r["detector"] == d and r["regime"] == "clean"] or [0])),
    )  # fmt: skip
    cols = [(d, g) for g in ("clean", "natural") for d in DS if any(r["dataset"] == d and r["regime"] == g for r in rows)]

    def val(d: str, c: tuple[str, str]) -> float | None:
        r = next((r for r in rows if r["detector"] == d and (r["dataset"], r["regime"]) == c), None)
        return None if r is None else r["roc"]

    singles_ids = [d for d in dets if not d.startswith("ens_")]
    best, second = {}, {}
    for c in cols:
        v = sorted((val(d, c) for d in singles_ids if val(d, c) is not None), reverse=True)
        best[c], second[c] = v[0], v[1]
    n_clean = sum(g == "clean" for _, g in cols)
    lines = [
        r"\begin{tabular}{l" + "c" * len(cols) + "c}", r"\toprule",
        rf" & \multicolumn{{{n_clean}}}{{c}}{{Clean training}} & \multicolumn{{{len(cols) - n_clean}}}{{c}}{{Natural training}} & \\",
        rf"\cmidrule(lr){{2-{1 + n_clean}}}\cmidrule(lr){{{2 + n_clean}-{1 + len(cols)}}}",
        "Detector & " + " & ".join(DS_SHORT[d] for d, _ in cols) + r" & Rank \\", r"\midrule",
    ]  # fmt: skip
    for d in dets:
        if d == "ens_all":
            lines.append(r"\midrule")
        cells = []
        for c in cols:
            v = val(d, c)
            if v is None:
                cells.append("--")
            elif d.startswith("ens_"):
                cells.append(f"{v:.3f}")
            elif abs(v - best[c]) < 1e-9:
                cells.append(rf"\textbf{{{v:.3f}}}")
            elif abs(v - second[c]) < 1e-9:
                cells.append(rf"\underline{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        rk = f"{ranks[d]:.1f}" if d in ranks else "--"
        lines.append(tex(title(d)) + " & " + " & ".join(cells) + f" & {rk} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "main.tex").write_text("\n".join(lines))

    # detector summary table (settings are those of ids_anomaly.detectors; cost from E1, clean regime)
    spec = {
        "iforest": ("Tree", "200 trees, 256 samples/tree"),
        "ocsvm": ("Kernel", "RBF, $\\nu{=}0.05$, $\\le$8k rows"),
        "pca": ("Reconstruction", "95\\% variance retained"),
        "gmm": ("Density", "8 full-covariance components"),
        "knn": ("Distance", "mean distance to $k{=}10$ neighbours"),
        "lof": ("Distance", "$k{=}20$, novelty mode"),
        "ecod": ("Density", "parameter-free ECDF tails"),
        "copod": ("Density", "parameter-free copula tails"),
        "hbos": ("Density", "20 histogram bins"),
        "ae": ("Reconstruction", "MLP 128-64-$\\lfloor d/8\\rfloor$, MSE"),
        "deep_svdd": ("Deep one-class", "bias-free MLP, 10+30 epochs"),
        "vae": ("Reconstruction", "$\\beta{=}0.5$, negative ELBO"),
        "flow": ("Density", "RealNVP, 6 couplings, 128 hidden"),
        "neutral": ("Self-supervised", "$K{=}6$ learned transformations"),
        "mtt": ("Self-supervised", "2-layer, 4-head, 30\\% masking"),
    }  # fmt: skip
    cl = [r for r in rows if r["regime"] == "clean"]
    lines = [r"\begin{tabular}{llllrr}", r"\toprule", r"Detector & Family & Fixed configuration & Ref. & Fit (s) & Score (ms/1k) \\", r"\midrule"]  # fmt: skip
    refs = {"iforest": "liu2008", "ocsvm": "scholkopf2001", "pca": "", "gmm": "", "knn": "", "lof": "breunig2000",
            "ecod": "li2022", "copod": "li2020", "hbos": "goldstein2012", "ae": "", "deep_svdd": "ruff2018",
            "vae": "kingma2014,higgins2017", "flow": "dinh2017", "neutral": "qiu2021", "mtt": "gorishniy2021"}  # fmt: skip
    for d in singles_ids:
        fam, cfg = spec[d]
        fit = np.mean([r["fit_s"] for r in cl if r["detector"] == d])
        sc = np.mean([r["score_ms_per_1k"] for r in cl if r["detector"] == d])
        ref = rf"\cite{{{refs[d]}}}" if refs[d] else "--"
        lines.append(rf"{tex(title(d))} & {fam} & {cfg} & {ref} & {fit:.1f} & {sc:.1f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "detectors.tex").write_text("\n".join(lines))

    if st:
        macro("friedmanChi", f"{st['statistic']:.1f}")
        macro("friedmanBlocks", str(st["n_blocks"]))
        macro("cdValue", f"{st['cd']:.2f}")
        macro("bestDetector", tex(title(main["stats"]["best_by_rank"])))
        macro("bestRank", f"{ranks[main['stats']['best_by_rank']]:.1f}")
        macro("nTied", str(len(st["cd_groups"][0]) - 1))
        dr = main["stats"]["dataset_regime"]
        macro("drBlocks", str(dr["n_blocks"]))
        macro("drChi", f"{dr['statistic']:.1f}")
        macro("drP", f"{dr['p']:.2f}")
        macro("drCd", f"{dr['cd']:.2f}")

        # ----------------------------------------------------------- critical-difference diagram
        order = sorted(ranks.items(), key=lambda kv: kv[1])
        lo, hi = float(np.floor(order[0][1])), float(np.ceil(order[-1][1]))
        half = (len(order) + 1) // 2
        step = 0.62
        fig, ax = plt.subplots(figsize=(PAGEW, 2.55))
        ax.set_xlim(lo - 3.6, hi + 3.6)
        ax.set_ylim(-(half * step) - 1.55, 1.9)
        ax.axis("off")
        ax.plot([lo, hi], [0, 0], color="k", lw=0.8)
        for t in np.arange(lo, hi + 0.01, 0.5):
            major = abs(t - round(t)) < 1e-9
            ax.plot([t, t], [0, 0.16 if major else 0.08], color="k", lw=0.7)
            if major:
                ax.text(t, 0.32, f"{int(t)}", ha="center", va="bottom", fontsize=7.5)
        ax.text((lo + hi) / 2, 0.95, "Mean rank (lower is better)", ha="center", va="bottom", fontsize=7.5)
        cdx = lo + 0.1
        ax.plot([cdx, cdx + st["cd"]], [1.55, 1.55], color="k", lw=1.2)
        for x in (cdx, cdx + st["cd"]):
            ax.plot([x, x], [1.45, 1.65], color="k", lw=1.0)
        ax.text(cdx + st["cd"] / 2, 1.72, f"CD = {st['cd']:.2f}", ha="center", va="bottom", fontsize=7)
        for i, (d, r) in enumerate(order):
            left = i < half
            row = i if left else len(order) - 1 - i
            y = -0.55 - row * step
            x_text = lo - 0.45 if left else hi + 0.45
            ax.plot([r, r, x_text], [0, y, y], color="0.35", lw=0.5)
            ax.text(x_text + (-0.12 if left else 0.12), y, f"{title(d)} ({r:.1f})", ha="right" if left else "left", va="center", fontsize=7.2)  # fmt: skip
        base_y = -(half * step) - 0.55
        for gi, g in enumerate(st["cd_groups"]):
            rs = [st["mean_ranks"][m] for m in g]
            y = base_y - gi * 0.2
            ax.plot([min(rs) - 0.04, max(rs) + 0.04], [y, y], color=BLUE if gi == 0 else "k", lw=2.2 if gi == 0 else 1.6, solid_capstyle="round")  # fmt: skip
        fig.savefig(OUT / "figures" / "cd.pdf")
        plt.close(fig)

    singles = [r for r in rows if r["regime"] == "clean" and not r["detector"].startswith("ens_")]
    if singles:
        b = max(singles, key=lambda r: r["roc"])
        macro("bestCleanAuc", f"{b['roc']:.3f}")
        macro("bestCleanName", tex(title(b["detector"])))
    ens = [r for r in rows if r["detector"] == "ens_all" and r["regime"] == "clean"]
    if ens:
        macro("ensFprOne", f"{100 * np.mean([r['a1_fpr'] for r in ens]):.2f}")
    for k, tg in DS_SHORT.items():
        v = [r["a1_fpr"] for r in singles if r["dataset"] == k]
        if v:
            macro(f"singleFpr{tg}", f"{100 * np.mean(v):.1f}")
    div = [r for r in rows if r["detector"] == "ens_diverse" and r["regime"] == "clean"]
    if div:
        macro("divFprMin", f"{100 * min(r['a1_fpr'] for r in div):.1f}")
        macro("divFprMax", f"{100 * max(r['a1_fpr'] for r in div):.1f}")
        macro("divTprMin", f"{100 * min(r['a5_tpr'] for r in div):.0f}")
        macro("divTprMax", f"{100 * max(r['a5_tpr'] for r in div):.0f}")
    allu = [r for r in rows if r["detector"] == "ens_all" and r["regime"] == "clean" and r["dataset"] == "unsw_nb15"]
    if allu:
        macro("ensAllUnswTpr", f"{100 * allu[0]['a5_tpr']:.1f}")
    sing_fpr = [r["a1_fpr"] for r in singles]
    if sing_fpr:
        macro("singleFprOne", f"{100 * np.mean(sing_fpr):.2f}")
    nat = [r for r in rows if r["regime"] == "natural" and r["dataset"] == "unsw_nb15" and not r["detector"].startswith("ens_")]
    if nat:
        macro("unswNaturalBelow", str(sum(r["roc"] < 0.5 for r in nat)))
        macro("unswNaturalN", str(len(nat)))

# ------------------------------------------------------------------------- contamination
c = S.get("contamination")
if c:
    dsets = [d for d in DS if any(r["dataset"] == d for r in c["table"])]
    fig, axes = plt.subplots(1, len(dsets) + 1, figsize=(PAGEW, 2.05), gridspec_kw={"width_ratios": [1] * len(dsets) + [1.05]})
    for k, (ax, d) in enumerate(zip(axes[:-1], dsets, strict=True)):
        for det in ("ae", "knn", "iforest", "flow"):
            for suffix, ls in (("", "-"), ("+ctr0.1", "--")):
                rs = sorted((r for r in c["table"] if r["dataset"] == d and r["detector"] == det + suffix), key=lambda r: r["rho"])
                if rs:
                    lab = None
                    if k == 0:
                        lab = title(det) if not suffix else None
                    ax.plot([100 * r["rho"] for r in rs], [r["roc"] for r in rs], ls, color=COL[det], marker="o" if not suffix else None, ms=2.4, label=lab)  # fmt: skip
        ax.set_title(DS[d])
        ax.set_xlabel(r"contamination $\rho$ (%)")
        ax.axhline(0.5, color="0.5", lw=0.6, ls=":")
        ax.set_xticks([0, 5, 10, 20])
        ax.set_ylim(0.38, 1.0)
        panel(ax, "abc"[k])
    axes[0].set_ylabel("ROC-AUC")
    h, lab = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="k", ls="-"), plt.Line2D([], [], color="k", ls="--")]
    lab += ["plain", "+ CTR (10% trim)"]
    fig_legend = (h, lab)

    gains = [g for g in c["ctr_gain"] if g["detector"].endswith("+ctr0.1")]
    rhos = sorted({g["rho"] for g in gains})
    ax = axes[-1]
    rng = np.random.default_rng(0)
    for i, rho in enumerate(rhos):
        gg = [100 * g["gain"] for g in gains if abs(g["rho"] - rho) < 1e-9]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(gg)), gg, s=5, color=GREY, alpha=0.55, lw=0, zorder=2)
    means = [100 * np.mean([g["gain"] for g in gains if abs(g["rho"] - rho) < 1e-9]) for rho in rhos]
    ax.plot(range(len(rhos)), means, "-o", color=BLUE, ms=3, zorder=3, label="mean gain")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(rhos)), [f"{100 * r:g}" for r in rhos])
    ax.set_xlabel(r"contamination $\rho$ (%)")
    ax.set_ylabel("CTR gain (AUC points)")
    ax.set_title("CTR gain, all pairs")
    ax.legend(frameon=False, loc="upper right")
    panel(ax, "d")
    fig.tight_layout(w_pad=0.8)
    fig.legend(*fig_legend, loc="upper center", ncol=6, frameon=False, handlelength=1.8, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT / "figures" / "contamination.pdf")
    plt.close(fig)

    g5 = [g for g in c["ctr_gain"] if abs(g["rho"] - 0.05) < 1e-9 and g["detector"].endswith("+ctr0.1")]
    if g5:
        macro("ctrGainFive", f"{np.mean([g['gain'] for g in g5]):+.3f}")
        macro("ctrGainFivePts", f"{100 * np.mean([g['gain'] for g in g5]):+.1f}")
        macro("ctrWinFraction", f"{sum(g['gain'] > 0 for g in g5)}/{len(g5)}")
        macro("ctrPairs", str(len(g5)))
    for name, rho in (("Zero", 0.0), ("One", 0.01), ("Two", 0.02), ("Ten", 0.10), ("Twenty", 0.20)):
        gg = [g for g in c["ctr_gain"] if abs(g["rho"] - rho) < 1e-9 and g["detector"].endswith("+ctr0.1")]
        if gg:
            macro(f"ctrGain{name}Pts", f"{100 * np.mean([g['gain'] for g in gg]):+.1f}".replace("-", "$-$"))
            macro(f"ctrWin{name}", f"{sum(g['gain'] > 0 for g in gg)}/{len(gg)}")
    z = [g for g in c["ctr_gain"] if abs(g["rho"]) < 1e-9 and g["detector"].endswith("+ctr0.1")]
    if z:
        macro("ctrCostZero", f"{abs(100 * np.mean([g['gain'] for g in z])):.1f}")
    deg = c["degradation"]
    if deg:
        macro("meanDropFivePts", f"{100 * np.mean([d['drop'] for d in deg]):.1f}")
        worst = max(deg, key=lambda d: d["drop"])
        macro("worstDropPts", f"{100 * worst['drop']:.0f}")
        macro("worstDropName", tex(title(worst["detector"])))
        macro("worstDropDs", tex(DS[worst["dataset"]]))

    def roc_at(ds: str, det: str, rho: float) -> float | None:
        return next((r["roc"] for r in c["table"] if r["dataset"] == ds and r["detector"] == det and abs(r["rho"] - rho) < 1e-9), None)

    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Dataset & Detector & $\rho{=}0$ & $\rho{=}5\%$ & + CTR & $\Delta$ \\", r"\midrule"]  # fmt: skip
    for k, d in enumerate(dsets):
        for det in ("ae", "knn", "flow", "gmm"):
            a, b, cc = roc_at(d, det, 0.0), roc_at(d, det, 0.05), roc_at(d, det + "+ctr0.1", 0.05)
            if None in (a, b, cc):
                continue
            lines.append(rf"{DS[d]} & {tex(title(det))} & {a:.3f} & {b:.3f} & {cc:.3f} & {cc - b:+.3f} \\")
        if k < len(dsets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "contamination.tex").write_text("\n".join(lines))
    abl_ids = ["ae", "ae+ctr0.05", "ae+ctr0.1", "ae+ctr0.2", "ae+ctr0.1[iforest]", "ae+ctr0.1[ecod]", "ae+ctr0.1[pca]", "ae+ctr@rho"]
    names = {"ae": "No purification", "ae+ctr0.05": "Committee, 5\\%", "ae+ctr0.1": "Committee, 10\\% (default)", "ae+ctr0.2": "Committee, 20\\%", "ae+ctr0.1[iforest]": "IForest only, 10\\%", "ae+ctr0.1[ecod]": "ECOD only, 10\\%", "ae+ctr0.1[pca]": "PCA only, 10\\%", "ae+ctr@rho": "Oracle trim (5\\%)"}  # fmt: skip
    lines = [r"\begin{tabular}{l" + "r" * len(dsets) + "}", r"\toprule", "Training-set purification & " + " & ".join(DS[d] for d in dsets) + r" \\", r"\midrule"]  # fmt: skip
    for i in abl_ids:
        vals = [next((r["roc"] for r in c["table"] if r["dataset"] == d and r["detector"] == i and abs(r["rho"] - 0.05) < 1e-9), None) for d in dsets]  # fmt: skip
        if any(v is not None for v in vals):
            if i in ("ae+ctr0.05", "ae+ctr0.1[iforest]"):
                lines.append(r"\midrule")
            lines.append(names[i] + " & " + " & ".join("--" if v is None else f"{v:.3f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables" / "ablation.tex").write_text("\n".join(lines))

# ------------------------------------------------------------------------------------ drift
d = S.get("drift")
if d:
    days = ["Tue", "Wed", "Thu", "Fri"]
    dcol = {"Tue": BLUE, "Wed": GREEN, "Thu": RED, "Fri": ORANGE}
    dets = sorted({r["detector"] for r in d["table"] if not r["detector"].startswith("ens_")}, key=lambda x: np.mean([r["fpr_a1"] for r in d["table"] if r["detector"] == x]))  # fmt: skip
    M = np.array([[next((r["fpr_a1"] for r in d["table"] if r["detector"] == x and r["day"] == day), np.nan) for day in days] for x in dets])  # fmt: skip
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGEW, 2.9), gridspec_kw={"width_ratios": [1, 1.1]})
    im = a1.imshow(np.log10(np.clip(M, 2e-3, 1)), cmap="YlOrRd", aspect="auto", vmin=-2.7, vmax=-0.2)
    a1.set_xticks(range(4), ["Tuesday", "Wednesday", "Thursday", "Friday"])
    a1.set_yticks(range(len(dets)), [title(x) for x in dets])
    a1.grid(False)
    a1.spines[:].set_visible(False)
    a1.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(4):
            a1.text(j, i, f"{100 * M[i, j]:.1f}", ha="center", va="center", fontsize=6.3, color="white" if M[i, j] > 0.06 else "black")  # fmt: skip
    a1.set_title(r"Realized FPR (%) at nominal $\alpha{=}1\%$")
    panel(a1, "a")
    pts = [(r["psi_benign"], r["fpr_a1"], r["day"]) for r in d["table"] if not r["detector"].startswith("ens_") and r["fpr_a1"] > 0]  # fmt: skip
    for day in days:
        p = [q for q in pts if q[2] == day]
        a2.scatter([max(q[0], 1e-3) for q in p], [q[1] * 100 for q in p], s=11, color=dcol[day], alpha=0.85, lw=0.3, edgecolor="white", label=day)  # fmt: skip
    a2.set(xscale="log", yscale="log", xlabel="PSI of benign scores (label-free)", ylabel="Realized FPR (%)")
    a2.axvline(0.10, color="0.35", ls="--", lw=0.7)
    a2.axvline(0.25, color="0.35", ls="--", lw=0.7)
    a2.axhline(3.0, color="0.35", ls=":", lw=0.8)
    a2.text(0.105, 0.6, "PSI 0.10", rotation=90, fontsize=6.3, color="0.3", va="bottom")
    a2.text(0.26, 0.6, "PSI 0.25", rotation=90, fontsize=6.3, color="0.3", va="bottom")
    a2.annotate(r"$3\times$ nominal", xy=(0.02, 3.2), xycoords=("axes fraction", "data"), ha="left", va="bottom", fontsize=6.5, color="0.3")
    a2.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=1.0, handletextpad=0.1)
    rs_txt = rf" ($r_s={d['psi_fpr_spearman']['rho']:.2f}$)" if d["psi_fpr_spearman"] else ""
    a2.set_title("PSI of benign scores against realized FPR" + rs_txt)
    panel(a2, "b")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "figures" / "drift.pdf")
    plt.close(fig)
    if d["psi_fpr_spearman"]:
        macro("psiSpearman", f"{d['psi_fpr_spearman']['rho']:.2f}")
    al = next(a for a in d["drift_alarms"] if abs(a["threshold_psi"] - 0.10) < 1e-9)
    macro("alarmThr", "0.10")
    macro("alarmRecall", "n/a" if al["recall"] is None else f"{100 * al['recall']:.0f}")
    macro("alarmPrecision", "n/a" if al["precision"] is None else f"{100 * al['precision']:.0f}")
    lo_al = next(a for a in d["drift_alarms"] if abs(a["threshold_psi"] - 0.05) < 1e-9)
    macro("alarmLoRecall", f"{100 * lo_al['recall']:.0f}")
    macro("alarmLoPrecision", f"{100 * lo_al['precision']:.0f}")
    hi = d["drift_alarm"]
    macro("alarmHiPrecision", "n/a" if hi["precision"] is None else f"{100 * hi['precision']:.0f}")
    macro("alarmHiRecall", "n/a" if hi["recall"] is None else f"{100 * hi['recall']:.0f}")
    macro("alarmCases", str(al["tp"] + al["fn"]))
    macro("alarmPairs", str(sum(1 for r in d["table"] if not r["detector"].startswith("ens_") and r["fpr_a1"] is not None)))
    per_day = {day: float(np.mean([r["fpr_a1"] for r in d["table"] if r["day"] == day and not r["detector"].startswith("ens_")])) for day in days}  # fmt: skip
    worst, best = max(per_day, key=per_day.get), min(per_day, key=per_day.get)
    full = {"Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"}
    macro("driftWorstDay", full[worst])
    macro("driftBestDay", full[best])
    macro("driftWorstFpr", f"{100 * per_day[worst]:.1f}")
    macro("driftBestFpr", f"{100 * per_day[best]:.1f}")
    for day in days:
        macro(f"driftFpr{full[day]}", f"{100 * per_day[day]:.1f}")
    robust = sorted(dets, key=lambda x: np.mean([r["fpr_a1"] for r in d["table"] if r["detector"] == x]))
    macro("driftRobustA", tex(title(robust[0])))
    macro("driftRobustB", tex(title(robust[1])))
    macro("driftRobustC", tex(title(robust[2])))
    macro("driftFragileA", tex(title(robust[-1])))
    macro("driftFragileB", tex(title(robust[-2])))

# ---------------------------------------------------------------------------------- transfer
t = S.get("transfer")
if t:
    cfg = [("unsw_nb15", "unsw_nb15", "source", "U$\\to$U"), ("unsw_nb15", "cicids2017", "source", "U$\\to$C"), ("unsw_nb15", "cicids2017", "target", "U$\\to$C$^{\\dagger}$"),
           ("cicids2017", "cicids2017", "source", "C$\\to$C"), ("cicids2017", "unsw_nb15", "source", "C$\\to$U"), ("cicids2017", "unsw_nb15", "target", "C$\\to$U$^{\\dagger}$")]  # fmt: skip
    dets = sorted({r["detector"] for r in t["table"] if not r["detector"].startswith("ens_")}, key=lambda x: -np.mean([r["roc"] for r in t["table"] if r["detector"] == x]))  # fmt: skip
    M = np.array([[next((r["roc"] for r in t["table"] if r["detector"] == x and (r["source"], r["target"], r["scaler"]) == c[:3]), np.nan) for c in cfg] for x in dets])  # fmt: skip
    fig, ax = plt.subplots(figsize=(COLW, 3.9))
    im = ax.imshow(M, cmap="cividis", vmin=0.3, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(cfg)), [c[3] for c in cfg])
    ax.set_yticks(range(len(dets)), [title(x) for x in dets])
    ax.xaxis.tick_top()
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    for sep in (2.5,):
        ax.axvline(sep, color="white", lw=2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color="white" if M[i, j] < 0.62 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=30)
    cb.set_label("ROC-AUC", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)
    fig.savefig(OUT / "figures" / "transfer.pdf")
    plt.close(fig)
    for key, (s, tg, sc, _) in zip(("UU", "UC", "UCd", "CC", "CU", "CUd"), cfg, strict=True):
        vals = [r["roc"] for r in t["table"] if (r["source"], r["target"], r["scaler"]) == (s, tg, sc) and not r["detector"].startswith("ens_")]  # fmt: skip
        fpr = [r["a1_fpr"] for r in t["table"] if (r["source"], r["target"], r["scaler"]) == (s, tg, sc) and not r["detector"].startswith("ens_")]  # fmt: skip
        if vals:
            macro(f"transfer{key}", f"{np.mean(vals):.3f}")
            macro(f"transferFpr{key}", f"{100 * np.mean(fpr):.0f}")
    ret = [r["retention"] for r in t["retention"] if r["scaler"] == "source" and r["retention"] is not None]
    if ret:
        macro("transferRetention", f"{100 * np.mean(ret):.0f}")

# ------------------------------------------------------------------------------- reliability
if main and main.get("coverage"):
    fig, axes = plt.subplots(1, 3, figsize=(PAGEW, 2.15), sharey=True)
    indcol = {"nsl_kdd": SKY, "unsw_nb15": GREEN, "cicids2017": BLUE}
    for k, (ax, det) in enumerate(zip(axes, ("ae", "iforest", "ens_all"), strict=True)):
        for ds in DS:
            rs = sorted((r for r in main["coverage"] if r["dataset"] == ds and r["regime"] == "clean" and r["detector"] == det), key=lambda r: r["alpha"])  # fmt: skip
            if rs:
                ax.plot([100 * r["alpha"] for r in rs], [100 * r["fpr"] for r in rs], color=indcol[ds], lw=1.1, label=f"{DS[ds]} (in-dataset)")  # fmt: skip
        if d:
            for day, colr in (("Tue", ORANGE), ("Thu", RED), ("Fri", PURPLE)):
                rs = sorted((r for r in d["coverage"] if r["detector"] == det and r["day"] == day), key=lambda r: r["alpha"])
                if rs:
                    ax.plot([100 * r["alpha"] for r in rs], [100 * r["fpr"] for r in rs], color=colr, ls="--", lw=1.1, label=f"CIC {day} (drift)")
        ax.plot([0, 20], [0, 20], color="k", lw=0.7, ls=":")
        ax.set_title(title(det))
        ax.set_xlabel(r"nominal $\alpha$ (%)")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 40)
        panel(ax, "abc"[k])
    axes[0].set_ylabel("Realized FPR (%)")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="k", ls=":", lw=0.8))
    lab.append("valid (diagonal)")
    fig.tight_layout(w_pad=0.8)
    fig.legend(h, lab, loc="upper center", ncol=6, frameon=False, handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT / "figures" / "reliability.pdf")
    plt.close(fig)

# ------------------------------------------------------------------------------- explainability
fig, ax = plt.subplots(figsize=(COLW, 2.25))
dcols = {"nsl_kdd": SKY, "unsw_nb15": GREEN, "cicids2017": BLUE}
for m in MAN["datasets"]:
    dl = m["explain"]["deletion"]
    ax.plot(dl["k"], [100 * v for v in dl["attributed"]], "-o", ms=2.6, color=dcols[m["key"]], label=f"{m['title']}, top-$k$")
    ax.plot(dl["k"], [100 * v for v in dl["random"]], "--", color=dcols[m["key"]], alpha=0.6, lw=1.0)
    nmi = m["explain"]["attribution_cluster_nmi"]
    tag = DS_SHORT[m["key"]]
    macro(f"nmi{tag}", "n/a" if nmi is None else f"{nmi:.2f}")
    macro(f"delThree{tag}", f"{100 * dl['attributed'][2]:.0f}")
    macro(f"delThreeRand{tag}", f"{100 * dl['random'][2]:.0f}")
ax.plot([], [], "--", color="0.4", label="random $k$ (each dataset)")
ax.set_xlabel("Features deleted $k$")
ax.set_ylabel("Anomaly score removed (%)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.5, labelspacing=0.25, columnspacing=1.0)
fig.savefig(OUT / "figures" / "explain.pdf")
plt.close(fig)

# ------------------------------------------------------------------------------------ datasets
attack_train = {"nsl_kdd": "47", "unsw_nb15": "68", "cicids2017": r"$\approx$1.4"}
meta = {
    "nsl_kdd": ("2009", "125,973", "22,544", "41", "official"),
    "unsw_nb15": ("2015", "175,341", "82,332", "34", "official"),
    "cicids2017": ("2017", r"$\approx$150k", None, "78", "temporal"),
}  # fmt: skip
lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r"Dataset & Year & Train & Test & Raw feat. & Enc. feat. & Atk. test (\%) & Atk. train (\%) \\", r"\midrule"]  # fmt: skip
for m in MAN["datasets"]:
    yr, ntr, nte, nraw, _ = meta[m["key"]]
    nte = f"{m['n_test']:,}" if nte is None else nte
    lines.append(rf"{m['title']} & {yr} & {ntr} & {nte} & {nraw} & {m['n_features']} & {100 * m['attack_fraction']:.1f} & {attack_train[m['key']]} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tables" / "datasets.tex").write_text("\n".join(lines))

macro("nDetectors", "15")
macro("nFlows", f"{sum(m['n_test'] for m in MAN['datasets']):,}".replace(",", r"\,"))
(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print("wrote", len(macros), "macros")
