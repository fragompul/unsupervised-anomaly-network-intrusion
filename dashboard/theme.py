"""Design system for the dashboard: color tokens, a dark Bokeh Theme for every figure, and the
categorical palettes used when coloring by a label field instead of a continuous score.

Kept in one place so the Bokeh `Theme` (which only styles plot-level models: backgrounds, grids,
axes) and the hand-picked categorical colors (which the app applies directly as a per-row hex
column, since Bokeh has no theme hook for "this categorical value gets this color") stay visually
consistent with each other and with the page chrome in ``templates/index.html``.
"""

from __future__ import annotations

from bokeh.themes import Theme

# --- Core tokens ---------------------------------------------------------------------------------
BG = "#0a0e17"
BG_ELEVATED = "#0d1220"
CARD = "#131829"
CARD_BORDER = "#232b42"
TEXT = "#eef1f8"
TEXT_DIM = "#8b93ab"
TEXT_FAINT = "#5a6484"
GRID_LINE = "#1f2740"
AXIS_LINE = "#2a3350"

ACCENT = "#22d3ee"  # cyan -- normal / primary accent
DANGER = "#fb4570"  # rose -- attack / alert
WARNING = "#fbbf24"  # amber
VIOLET = "#a78bfa"
EMERALD = "#34d399"

FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, monospace"

# --- Categorical palettes --------------------------------------------------------------------------
CATEGORY_COLORS: dict[str, str] = {
    "normal": ACCENT,
    "dos": DANGER,
    "probe": WARNING,
    "r2l": VIOLET,
    "u2r": "#ec4899",  # magenta, kept distinct from dos' rose
    "unknown_attack": TEXT_FAINT,
}

IS_ATTACK_COLORS: dict[str, str] = {"False": ACCENT, "True": DANGER}


def hsl_palette(n: int) -> list[str]:
    """Evenly-spaced hues for fields with too many distinct values (e.g. fine-grained ``label``)
    to hand-pick a color for each one -- fixed saturation/lightness tuned to stay legible on the
    dashboard's dark background.
    """
    return [f"hsl({round(360 * i / max(n, 1))}, 70%, 62%)" for i in range(n)]


# --- Bokeh figure theme -----------------------------------------------------------------------------
DASHBOARD_THEME = Theme(
    json={
        "attrs": {
            # Bokeh's `figure()` factory returns an instance of the (lowercase) `figure` class,
            # which subclasses `Plot` -- theme lookups match by declared class name, and `Plot`
            # is what actually carries background_fill_color, so that is the key that works here
            # (the capitalized `Figure` name that appears in bokeh.plotting's public API is not
            # a real Python class Theme can match against).
            "Plot": {
                "background_fill_color": CARD,
                "border_fill_color": CARD,
                "outline_line_color": CARD_BORDER,
                "outline_line_alpha": 1,
            },
            "Grid": {
                "grid_line_color": GRID_LINE,
                "grid_line_alpha": 0.6,
            },
            "Axis": {
                "axis_line_color": AXIS_LINE,
                "major_tick_line_color": AXIS_LINE,
                "minor_tick_line_color": None,
                "major_label_text_color": TEXT_DIM,
                "major_label_text_font": FONT_MONO,
                "major_label_text_font_size": "10px",
                "axis_label_text_color": TEXT_DIM,
                "axis_label_text_font": FONT_SANS,
            },
            "Title": {
                "text_color": TEXT,
                "text_font": FONT_SANS,
                "text_font_size": "14px",
                "text_font_style": "bold",
            },
            "Legend": {
                "background_fill_color": CARD,
                "background_fill_alpha": 0.92,
                "border_line_color": CARD_BORDER,
                "label_text_color": TEXT_DIM,
                "label_text_font": FONT_SANS,
            },
            "ColorBar": {
                "background_fill_color": CARD,
                "major_label_text_color": TEXT_DIM,
                "major_label_text_font": FONT_MONO,
                "title_text_color": TEXT_DIM,
                "border_line_color": None,
            },
        }
    }
)

CARD_STYLE = {
    "background": CARD,
    "border": f"1px solid {CARD_BORDER}",
    "border-radius": "14px",
    "padding": "18px 20px",
    "box-shadow": "0 4px 24px rgba(0,0,0,0.35)",
}

# Widget-level CSS injected via each model's `.stylesheets`, since Bokeh's Theme only reaches
# plot-level models -- Select/Slider/DataTable render their own shadow DOM the Theme can't touch.
SELECT_CSS = f"""
:host {{ --font: {FONT_SANS}; }}
label {{ color: {TEXT_DIM}; font-family: {FONT_SANS}; font-size: 12px; font-weight: 600;
         text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }}
select.bk-input {{
  background-color: {BG_ELEVATED}; color: {TEXT}; border: 1px solid {CARD_BORDER};
  border-radius: 8px; font-family: {FONT_SANS}; font-size: 13px; padding: 7px 10px;
}}
select.bk-input:focus {{ border-color: {ACCENT}; outline: none;
  box-shadow: 0 0 0 3px rgba(34,211,238,0.15); }}
"""

SLIDER_CSS = f"""
label {{ color: {TEXT_DIM}; font-family: {FONT_SANS}; font-size: 12px; font-weight: 600;
         text-transform: uppercase; letter-spacing: 0.04em; }}
.bk-slider-title {{ color: {TEXT_DIM}; font-family: {FONT_SANS}; font-size: 12px; font-weight: 600; }}
.noUi-target {{ background: {BG_ELEVATED}; border: 1px solid {CARD_BORDER}; box-shadow: none; }}
.noUi-connect {{ background: {ACCENT}; }}
.noUi-handle {{
  background: {TEXT}; border: 2px solid {ACCENT}; box-shadow: 0 0 8px rgba(34,211,238,0.5);
}}
.noUi-tooltip {{
  background: {BG_ELEVATED}; color: {TEXT}; border: 1px solid {CARD_BORDER};
  font-family: {FONT_MONO}; font-size: 11px;
}}
"""

TABLE_CSS = f"""
.slick-header-columns {{ background: {BG_ELEVATED}; border-bottom: 1px solid {CARD_BORDER}; }}
.slick-header-column {{
  color: {TEXT_DIM} !important; font-family: {FONT_SANS}; font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.03em; background: {BG_ELEVATED} !important;
  border-color: {CARD_BORDER} !important;
}}
.slick-row {{ background: {CARD} !important; color: {TEXT}; font-family: {FONT_MONO}; font-size: 12px; }}
.slick-row.odd {{ background: {BG_ELEVATED} !important; }}
.slick-row:hover {{ background: rgba(34,211,238,0.08) !important; }}
.slick-cell {{ border-color: {CARD_BORDER} !important; }}
.slick-row.selected .slick-cell {{ background: rgba(34,211,238,0.18) !important; }}
"""
