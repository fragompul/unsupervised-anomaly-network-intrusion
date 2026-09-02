"""Bokeh server dashboard: interactive 3D embedding explorer + anomaly drill-down table.

Run with:
    bokeh serve --show dashboard/app.py

Architecture: rotation (azimuth/elevation sliders) is pure client-side CustomJS for a responsive
drag; embedding choice, coloring, score selection and the flagging threshold trigger Python
server callbacks that recompute the projected columns and the table's row filter. Scatter and
table share one ColumnDataSource, so box/lasso-selecting points highlights the matching table
rows and vice versa for free.
"""

from __future__ import annotations

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
    LinearColorMapper,
    Select,
    Slider,
    Span,
    TableColumn,
)
from bokeh.palettes import Turbo256
from bokeh.plotting import figure

from data import EMBEDDING_KEYS, SCORE_KEYS, Artifacts
from projection import ROTATE_PROJECT_JS, normalize_embedding, rotate_project

artifacts = Artifacts()
NORM_EMBEDDINGS = {name: normalize_embedding(artifacts.embeddings[name]) for name in EMBEDDING_KEYS}

CATEGORICAL_FIELDS = ["attack_category", "label", "is_attack"]
COLOR_FIELDS = CATEGORICAL_FIELDS + list(SCORE_KEYS.keys())
CATEGORY_ORDER = ["normal", "dos", "probe", "r2l", "u2r", "unknown_attack"]

state = {"embedding": "umap", "color_by": "attack_category", "score_field": "if_score",
         "azimuth": 0.6, "elevation": 0.35, "quantile": 0.20}


def _color_values(field: str) -> np.ndarray:
    if field == "attack_category":
        codes = {c: i for i, c in enumerate(CATEGORY_ORDER)}
        return np.array([codes.get(c, len(CATEGORY_ORDER)) for c in artifacts.attack_category], dtype=float)
    if field == "label":
        uniq = sorted(set(artifacts.label.tolist()))
        codes = {c: i for i, c in enumerate(uniq)}
        return np.array([codes[c] for c in artifacts.label], dtype=float)
    if field == "is_attack":
        return artifacts.is_attack.astype(float)
    return artifacts.scores[field]


def _build_columns(embedding: str, color_by: str, azimuth: float, elevation: float) -> dict:
    x3, y3, z3 = (NORM_EMBEDDINGS[embedding][:, i] for i in range(3))
    x, y, size, alpha = rotate_project(x3, y3, z3, azimuth, elevation)
    cols = dict(artifacts.base_columns())
    cols.update(x3=x3, y3=y3, z3=z3, x=x, y=y, size=size, alpha=alpha, color_value=_color_values(color_by))
    return cols


source = ColumnDataSource(data=_build_columns(state["embedding"], state["color_by"], state["azimuth"], state["elevation"]))

color_mapper = LinearColorMapper(palette=Turbo256, low=float(np.nanmin(source.data["color_value"])),
                                  high=float(np.nanmax(source.data["color_value"])))

scatter_fig = figure(
    title="3D embedding explorer (drag sliders to rotate)",
    width=760, height=560, tools="pan,box_select,lasso_select,tap,wheel_zoom,reset",
    toolbar_location="above",
)
scatter_fig.xaxis.visible = False
scatter_fig.yaxis.visible = False
scatter_fig.grid.visible = False
scatter_fig.outline_line_color = "#333"

scatter_renderer = scatter_fig.scatter(
    "x", "y", source=source, size="size", alpha="alpha",
    # Share the same mapper instance with the ColorBar below (not bokeh.transform.linear_cmap,
    # which would build its own separate mapper) -- otherwise updating color_mapper.low/high in
    # the embedding/color-change callback would move the color bar without moving the actual
    # point colors, since the two would silently drift out of sync.
    fill_color={"field": "color_value", "transform": color_mapper},
    line_color=None,
)

color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), width=10)
scatter_fig.add_layout(color_bar, "right")

hover = HoverTool(renderers=[scatter_renderer], tooltips=[
    ("index", "@index"), ("label", "@label"), ("category", "@attack_category"),
    ("is_attack", "@is_attack"), ("IF score", "@if_score{0.000}"),
    ("OCSVM score", "@ocsvm_score{0.000}"), ("SVDD score", "@svdd_score{0.000}"),
    ("AE recon err", "@ae_recon_error{0.000}"),
])
scatter_fig.add_tools(hover)

legend_div = Div(text="", width=760)


def _update_legend(color_by: str) -> None:
    if color_by == "attack_category":
        items = ", ".join(f"{i}={c}" for i, c in enumerate(CATEGORY_ORDER))
        legend_div.text = f"<b>Color codes (attack_category):</b> {items}"
    elif color_by == "label":
        legend_div.text = "<b>Color:</b> fine-grained label, encoded alphabetically (see hover)."
    elif color_by == "is_attack":
        legend_div.text = "<b>Color:</b> 0 = normal, 1 = attack."
    else:
        legend_div.text = f"<b>Color:</b> {SCORE_KEYS[color_by]} (continuous, brighter = higher)."


_update_legend(state["color_by"])

# --- Score histogram with the current flagging threshold ---------------------------------------
hist_fig = figure(title="Anomaly score distribution", width=760, height=220, tools="")
hist_source = ColumnDataSource(data=dict(top=[], left=[], right=[]))
hist_fig.quad(top="top", bottom=0, left="left", right="right", source=hist_source,
               fill_color="#3288bd", line_color="white", alpha=0.85)
threshold_span = Span(location=0, dimension="height", line_color="crimson", line_width=2)
hist_fig.add_layout(threshold_span)


def _update_histogram(score_field: str, quantile: float) -> None:
    values = artifacts.scores[score_field]
    hist, edges = np.histogram(values, bins=40)
    hist_source.data = dict(top=hist, left=edges[:-1], right=edges[1:])
    threshold_span.location = float(np.quantile(values, 1 - quantile))
    hist_fig.title.text = f"{SCORE_KEYS[score_field]} distribution (red = top {quantile:.0%} threshold)"


_update_histogram(state["score_field"], state["quantile"])

# --- Anomaly drill-down table, filtered to the top-quantile flow of the selected score ---------
table_columns = [
    TableColumn(field="index", title="#"),
    TableColumn(field="label", title="True label"),
    TableColumn(field="attack_category", title="Category"),
    TableColumn(field="is_attack", title="Attack?"),
    TableColumn(field="if_score", title="IF score"),
    TableColumn(field="ocsvm_score", title="OCSVM score"),
    TableColumn(field="svdd_score", title="SVDD score"),
    TableColumn(field="ae_recon_error", title="AE recon err"),
]


def _top_indices(score_field: str, quantile: float) -> list[int]:
    values = artifacts.scores[score_field]
    threshold = np.quantile(values, 1 - quantile)
    return np.where(values >= threshold)[0].tolist()


table_view = CDSView(filter=IndexFilter(indices=_top_indices(state["score_field"], state["quantile"])))
data_table = DataTable(source=source, view=table_view, columns=table_columns, width=760, height=260)

# --- Widgets -------------------------------------------------------------------------------------
embedding_select = Select(title="Embedding", value=state["embedding"],
                           options=[("pca", "PCA"), ("umap", "UMAP"), ("autoencoder", "Autoencoder"), ("tsne", "t-SNE (subsample)")])
color_select = Select(title="Color by", value=state["color_by"],
                       options=[(f, SCORE_KEYS.get(f, f)) for f in COLOR_FIELDS])
score_select = Select(title="Anomaly score (histogram + table)", value=state["score_field"],
                       options=[(k, v) for k, v in SCORE_KEYS.items()])
quantile_slider = Slider(title="Flag top X% as anomalous", start=0.01, end=0.50, step=0.01, value=state["quantile"])
azimuth_slider = Slider(title="Rotate: azimuth", start=-3.14, end=3.14, step=0.02, value=state["azimuth"])
elevation_slider = Slider(title="Rotate: elevation", start=-1.5, end=1.5, step=0.02, value=state["elevation"])


def _on_embedding_or_color_change(attr, old, new) -> None:
    state["embedding"] = embedding_select.value
    state["color_by"] = color_select.value
    new_cols = _build_columns(state["embedding"], state["color_by"], azimuth_slider.value, elevation_slider.value)
    source.data = new_cols
    color_mapper.low = float(np.nanmin(new_cols["color_value"]))
    color_mapper.high = float(np.nanmax(new_cols["color_value"]))
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

header = Div(text="""
<h2 style="margin-bottom:0">Unsupervised Network Intrusion Detection &mdash; Live Explorer</h2>
<p style="color:#666;margin-top:4px">NSL-KDD test split, 41-feature flows reduced to 3D via PCA /
UMAP / Autoencoder / t-SNE, scored by four independent anomaly detectors. Rotate the embedding,
recolor by ground-truth category or by any detector's score, and drill into the flows each
method flags as most anomalous.</p>
""", width=1200)

controls = column(embedding_select, color_select, score_select, quantile_slider,
                   azimuth_slider, elevation_slider, width=260)

layout = column(
    header,
    row(controls, column(scatter_fig, legend_div)),
    row(hist_fig, data_table),
)

curdoc().add_root(layout)
curdoc().title = "Unsupervised IDS Explorer"
