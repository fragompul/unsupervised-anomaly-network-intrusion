"""Bokeh server dashboard: interactive 3D embedding explorer + anomaly drill-down table.

Run with:
    bokeh serve --show dashboard/app.py

Architecture: rotation (azimuth/elevation sliders) is pure client-side CustomJS for a responsive
drag; embedding choice, coloring, score selection and the flagging threshold trigger Python
server callbacks that recompute the projected columns and the table's row filter. Scatter and
table share one ColumnDataSource, so box/lasso-selecting points highlights the matching table
rows and vice versa for free. Visual design (dark theme, cards, categorical palette) lives in
``theme.py``; the custom page chrome (fonts, background) lives in ``templates/index.html``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    CDSView,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    DataTable,
    Div,
    HoverTool,
    IndexFilter,
    InlineStyleSheet,
    LinearColorMapper,
    Select,
    Slider,
    Span,
    TableColumn,
)
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
from jinja2 import Environment, FileSystemLoader

from data import EMBEDDING_KEYS, SCORE_KEYS, Artifacts
from projection import ROTATE_PROJECT_JS, normalize_embedding, rotate_project
from theme import (
    ACCENT,
    CARD,
    CARD_BORDER,
    CARD_STYLE,
    CATEGORY_COLORS,
    DANGER,
    DASHBOARD_THEME,
    FONT_MONO,
    FONT_SANS,
    IS_ATTACK_COLORS,
    SELECT_CSS,
    SLIDER_CSS,
    TABLE_CSS,
    TEXT,
    TEXT_DIM,
    hsl_palette,
)

curdoc().theme = DASHBOARD_THEME
# Bokeh only auto-discovers templates/index.html for directory-style apps (main.py); a single-file
# `bokeh serve app.py` needs the Jinja2 template loaded and assigned explicitly.
_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"))
curdoc().template = _env.get_template("index.html")

artifacts = Artifacts()
NORM_EMBEDDINGS = {name: normalize_embedding(artifacts.embeddings[name]) for name in EMBEDDING_KEYS}

CATEGORICAL_FIELDS = ["attack_category", "label", "is_attack"]
COLOR_FIELDS = CATEGORICAL_FIELDS + list(SCORE_KEYS.keys())
CATEGORY_ORDER = ["normal", "dos", "probe", "r2l", "u2r", "unknown_attack"]

UNIQUE_LABELS = sorted(set(artifacts.label.tolist()))
LABEL_COLORS = dict(zip(UNIQUE_LABELS, hsl_palette(len(UNIQUE_LABELS)), strict=True))

state = {"embedding": "umap", "color_by": "attack_category", "score_field": "if_score",
         "azimuth": 0.6, "elevation": 0.35, "quantile": 0.20}


def _categorical_colors(field: str) -> np.ndarray:
    if field == "attack_category":
        return np.array([CATEGORY_COLORS.get(c, "#5a6484") for c in artifacts.attack_category])
    if field == "label":
        return np.array([LABEL_COLORS[c] for c in artifacts.label])
    return np.array([IS_ATTACK_COLORS[str(b)] for b in artifacts.is_attack])


def _build_columns(embedding: str, color_by: str, azimuth: float, elevation: float) -> dict:
    x3, y3, z3 = (NORM_EMBEDDINGS[embedding][:, i] for i in range(3))
    x, y, size, alpha = rotate_project(x3, y3, z3, azimuth, elevation)
    cols = dict(artifacts.base_columns())
    is_categorical = color_by in CATEGORICAL_FIELDS
    cols.update(
        x3=x3, y3=y3, z3=z3, x=x, y=y, size=size, alpha=alpha,
        cat_color=_categorical_colors(color_by) if is_categorical else np.full(artifacts.n_test, ACCENT),
        color_value=np.zeros(artifacts.n_test) if is_categorical else artifacts.scores[color_by],
    )
    return cols


source = ColumnDataSource(data=_build_columns(state["embedding"], state["color_by"], state["azimuth"], state["elevation"]))

score_values = artifacts.scores[state["score_field"]]
color_mapper = LinearColorMapper(palette=Turbo256, low=float(np.min(score_values)), high=float(np.max(score_values)))

scatter_fig = figure(
    title="3D EMBEDDING EXPLORER  ·  drag sliders to rotate",
    height=560, sizing_mode="stretch_width",
    tools="pan,box_select,lasso_select,tap,wheel_zoom,reset", toolbar_location="above",
)
scatter_fig.xaxis.visible = False
scatter_fig.yaxis.visible = False
scatter_fig.grid.visible = False

scatter_renderer = scatter_fig.scatter(
    "x", "y", source=source, size="size", alpha="alpha",
    fill_color="cat_color", line_color=None,
)

color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), width=10, visible=False)
scatter_fig.add_layout(color_bar, "right")

_TOOLTIP_ROW = (
    "<div style='display:flex;justify-content:space-between;gap:14px;"
    "font-family:{font_mono};font-size:11px;padding:1px 0;'>"
    "<span style='color:{text_dim};'>{label}</span>"
    "<span style='color:{text};font-weight:600;'>{value}</span></div>"
).format
hover = HoverTool(renderers=[scatter_renderer], tooltips=f"""
<div style="padding:8px 10px;min-width:170px;font-family:{FONT_SANS};">
  <div style="color:{TEXT};font-weight:700;font-size:12px;margin-bottom:4px;">
    #@index &middot; @label</div>
  <div style="color:{ACCENT};font-size:10px;text-transform:uppercase;letter-spacing:.04em;
              margin-bottom:6px;">@attack_category &middot; attack=@is_attack</div>
  {_TOOLTIP_ROW(font_mono=FONT_MONO, text_dim=TEXT_DIM, text=TEXT, label="IF", value="@if_score{0.000}")}
  {_TOOLTIP_ROW(font_mono=FONT_MONO, text_dim=TEXT_DIM, text=TEXT, label="OCSVM", value="@ocsvm_score{0.000}")}
  {_TOOLTIP_ROW(font_mono=FONT_MONO, text_dim=TEXT_DIM, text=TEXT, label="SVDD", value="@svdd_score{0.000}")}
  {_TOOLTIP_ROW(font_mono=FONT_MONO, text_dim=TEXT_DIM, text=TEXT, label="AE err", value="@ae_recon_error{0.000}")}
</div>
""")
scatter_fig.add_tools(hover)


def _swatch_legend(field: str) -> str:
    if field == "attack_category":
        items = CATEGORY_ORDER
        colors = CATEGORY_COLORS
    elif field == "is_attack":
        items, colors = ["False", "True"], IS_ATTACK_COLORS
    else:
        items, colors = None, None
    if items is None:
        return f"""<div style="color:{TEXT_DIM};font-family:{FONT_SANS};font-size:12px;">
        {len(UNIQUE_LABELS)} fine-grained labels, hover a point to see which.</div>"""
    pills = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin:2px 10px 2px 0;">'
        f'<span style="width:10px;height:10px;border-radius:3px;background:{colors.get(c, "#5a6484")};'
        f'display:inline-block;box-shadow:0 0 6px {colors.get(c, "#5a6484")}55;"></span>'
        f'<span style="color:{TEXT_DIM};font-family:{FONT_MONO};font-size:11px;text-transform:uppercase;'
        f'letter-spacing:.03em;">{c}</span></span>'
        for c in items
    )
    return f'<div style="display:flex;flex-wrap:wrap;align-items:center;">{pills}</div>'


legend_div = Div(text="", sizing_mode="stretch_width", styles={"margin-top": "10px"})


def _update_legend(color_by: str) -> None:
    if color_by in CATEGORICAL_FIELDS:
        legend_div.text = _swatch_legend(color_by)
    else:
        legend_div.text = (
            f'<div style="color:{TEXT_DIM};font-family:{FONT_SANS};font-size:12px;">'
            f"Colored by <b style='color:{TEXT}'>{SCORE_KEYS[color_by]}</b> &middot; "
            f"brighter = higher score</div>"
        )


_update_legend(state["color_by"])

# --- Score histogram with the current flagging threshold ---------------------------------------
hist_fig = figure(title="ANOMALY SCORE DISTRIBUTION", height=230, sizing_mode="stretch_width", tools="")
hist_source = ColumnDataSource(data=dict(top=[], left=[], right=[]))
hist_fig.quad(top="top", bottom=0, left="left", right="right", source=hist_source,
               fill_color=ACCENT, fill_alpha=0.75, line_color=CARD, line_width=1)
threshold_span = Span(location=0, dimension="height", line_color=DANGER, line_width=2, line_dash="dashed")
hist_fig.add_layout(threshold_span)


def _update_histogram(score_field: str, quantile: float) -> None:
    values = artifacts.scores[score_field]
    hist, edges = np.histogram(values, bins=40)
    hist_source.data = dict(top=hist, left=edges[:-1], right=edges[1:])
    threshold_span.location = float(np.quantile(values, 1 - quantile))
    hist_fig.title.text = f"{SCORE_KEYS[score_field].upper()} DISTRIBUTION  ·  RED = TOP {quantile:.0%} THRESHOLD"


_update_histogram(state["score_field"], state["quantile"])

# --- Anomaly drill-down table, filtered to the top-quantile flow of the selected score ---------
table_columns = [
    TableColumn(field="index", title="#"),
    TableColumn(field="label", title="Label"),
    TableColumn(field="attack_category", title="Category"),
    TableColumn(field="is_attack", title="Attack"),
    TableColumn(field="if_score", title="IF"),
    TableColumn(field="ocsvm_score", title="OCSVM"),
    TableColumn(field="svdd_score", title="SVDD"),
    TableColumn(field="ae_recon_error", title="AE err"),
]


def _top_indices(score_field: str, quantile: float) -> list[int]:
    values = artifacts.scores[score_field]
    threshold = np.quantile(values, 1 - quantile)
    return np.where(values >= threshold)[0].tolist()


table_view = CDSView(filter=IndexFilter(indices=_top_indices(state["score_field"], state["quantile"])))
data_table = DataTable(
    source=source, view=table_view, columns=table_columns, sizing_mode="stretch_width", height=280,
    stylesheets=[InlineStyleSheet(css=TABLE_CSS)], row_height=28,
)

# --- Widgets -------------------------------------------------------------------------------------
_select_style = [InlineStyleSheet(css=SELECT_CSS)]
_slider_style = [InlineStyleSheet(css=SLIDER_CSS)]

embedding_select = Select(title="Embedding", value=state["embedding"],
                           options=[("pca", "PCA"), ("umap", "UMAP"), ("autoencoder", "Autoencoder"), ("tsne", "t-SNE (subsample)")],
                           stylesheets=_select_style)
color_select = Select(title="Color by", value=state["color_by"],
                       options=[(f, SCORE_KEYS.get(f, f)) for f in COLOR_FIELDS], stylesheets=_select_style)
score_select = Select(title="Anomaly score", value=state["score_field"],
                       options=[(k, v) for k, v in SCORE_KEYS.items()], stylesheets=_select_style)
quantile_slider = Slider(title="Flag top X% as anomalous", start=0.01, end=0.50, step=0.01,
                          value=state["quantile"], stylesheets=_slider_style)
azimuth_slider = Slider(title="Rotate: azimuth", start=-3.14, end=3.14, step=0.02,
                         value=state["azimuth"], stylesheets=_slider_style)
elevation_slider = Slider(title="Rotate: elevation", start=-1.5, end=1.5, step=0.02,
                           value=state["elevation"], stylesheets=_slider_style)


def _on_embedding_or_color_change(attr, old, new) -> None:
    state["embedding"] = embedding_select.value
    state["color_by"] = color_select.value
    is_categorical = state["color_by"] in CATEGORICAL_FIELDS
    new_cols = _build_columns(state["embedding"], state["color_by"], azimuth_slider.value, elevation_slider.value)
    source.data = new_cols

    if is_categorical:
        scatter_renderer.glyph.fill_color = "cat_color"
        color_bar.visible = False
    else:
        values = artifacts.scores[state["color_by"]]
        color_mapper.low = float(np.min(values))
        color_mapper.high = float(np.max(values))
        new_cols["color_value"] = values
        source.data = new_cols
        scatter_renderer.glyph.fill_color = {"field": "color_value", "transform": color_mapper}
        color_bar.visible = True
    _update_legend(state["color_by"])


def _on_score_or_threshold_change(attr, old, new) -> None:
    state["score_field"] = score_select.value
    state["quantile"] = quantile_slider.value
    _update_histogram(state["score_field"], state["quantile"])
    table_view.filter = IndexFilter(indices=_top_indices(state["score_field"], state["quantile"]))


embedding_select.on_change("value", _on_embedding_or_color_change)
color_select.on_change("value", _on_embedding_or_color_change)
score_select.on_change("value", _on_score_or_threshold_change)
quantile_slider.on_change("value", _on_score_or_threshold_change)

rotate_callback_args = dict(source=source, az_slider=azimuth_slider, el_slider=elevation_slider)
rotate_callback = CustomJS(args=rotate_callback_args, code=ROTATE_PROJECT_JS)
azimuth_slider.js_on_change("value", rotate_callback)
elevation_slider.js_on_change("value", rotate_callback)

# --- KPI strip, computed from the real pipeline results -----------------------------------------
_m = artifacts.metrics
_best_method, _best_row = max(_m["anomaly_detection"].items(), key=lambda kv: kv[1]["roc_auc"])
_best_label = SCORE_KEYS.get(
    {"isolation_forest": "if_score", "one_class_svm": "ocsvm_score", "deep_svdd": "svdd_score",
     "autoencoder_reconstruction": "ae_recon_error"}[_best_method], _best_method,
)


def _kpi(label: str, value: str, accent: str = ACCENT) -> Div:
    html = f"""
    <div style="display:flex;flex-direction:column;gap:4px;">
      <div style="color:{TEXT_DIM};font-family:{FONT_SANS};font-size:11px;font-weight:600;
                  text-transform:uppercase;letter-spacing:.06em;">{label}</div>
      <div style="color:{accent};font-family:{FONT_MONO};font-size:22px;font-weight:700;">{value}</div>
    </div>
    """
    return Div(text=html, styles={**CARD_STYLE, "padding": "14px 18px", "min-width": "180px", "flex": "1"})


kpi_row = row(
    _kpi("Test flows", f"{_m['dataset']['n_test']:,}"),
    _kpi("Attack rate", f"{_m['dataset']['test_attack_fraction']:.1%}", DANGER),
    _kpi("Best detector", _best_label.split(" ")[0], ACCENT),
    _kpi("Best ROC-AUC", f"{_best_row['roc_auc']:.3f}", "#34d399"),
    _kpi("Methods compared", "9", "#a78bfa"),
    sizing_mode="stretch_width", spacing=14,
)

header = Div(text=f"""
<div>
  <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">
    <h1 style="margin:0;font-family:{FONT_SANS};font-size:26px;font-weight:800;color:{TEXT};
               letter-spacing:-0.02em;">Unsupervised Network Intrusion Detection</h1>
    <span style="color:{ACCENT};font-family:{FONT_MONO};font-size:12px;border:1px solid {CARD_BORDER};
                 border-radius:999px;padding:3px 10px;">LIVE EXPLORER</span>
  </div>
  <p style="color:{TEXT_DIM};font-family:{FONT_SANS};font-size:13.5px;margin:8px 0 0;max-width:900px;
            line-height:1.5;">
    NSL-KDD test split · 41-feature flows reduced to 3D via PCA, UMAP, Autoencoder and t-SNE ·
    scored by four independent anomaly detectors. Rotate the embedding, recolor by ground-truth
    category or by any detector's score, and drill into the flows each method flags as most
    anomalous.
  </p>
</div>
""", sizing_mode="stretch_width")

controls_card = column(
    Div(text=f'<div style="color:{TEXT_DIM};font-family:{FONT_SANS};font-size:11px;'
             f'font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">'
             f"Controls</div>"),
    embedding_select, color_select, score_select, quantile_slider,
    Div(text=f'<div style="height:1px;background:{CARD_BORDER};margin:6px 0 10px;"></div>'),
    azimuth_slider, elevation_slider,
    width=250, styles={**CARD_STYLE, "height": "fit-content"}, spacing=10,
)

scatter_card = column(
    scatter_fig, legend_div,
    sizing_mode="stretch_width", styles=CARD_STYLE,
)

hist_card = column(hist_fig, sizing_mode="stretch_width", styles=CARD_STYLE)

table_card = column(
    Div(text=f'<div style="color:{TEXT_DIM};font-family:{FONT_SANS};font-size:11px;'
             f'font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">'
             f"Flagged flows &middot; top-quantile by selected score</div>"),
    data_table, sizing_mode="stretch_width", styles=CARD_STYLE,
)

footer = Div(text=f"""
<div style="color:{TEXT_DIM};font-family:{FONT_SANS};font-size:11.5px;padding:6px 2px 20px;">
  Built with scikit-learn, PyTorch, UMAP, HDBSCAN, Optuna and Bokeh &middot;
  <a href="https://github.com/fragompul/unsupervised-anomaly-network-intrusion" target="_blank"
     style="color:{TEXT_DIM};">github.com/fragompul/unsupervised-anomaly-network-intrusion</a>
</div>
""", sizing_mode="stretch_width")

layout = column(
    header,
    kpi_row,
    row(controls_card, scatter_card, sizing_mode="stretch_width", spacing=18),
    row(hist_card, table_card, sizing_mode="stretch_width", spacing=18),
    footer,
    sizing_mode="stretch_width",
    styles={"max-width": "1400px", "margin": "0 auto", "padding": "26px 28px 10px"},
)

curdoc().add_root(layout)
curdoc().title = "Unsupervised IDS Explorer"
