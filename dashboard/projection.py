"""Shared math between the Python-side initial render and the client-side rotation callback:
an orthographic projection of 3D embedding coordinates with a simple depth cue (size + alpha
falloff), rotated by azimuth (around Y) then elevation (around X).

Kept in pure NumPy/JS (no Three.js / WebGL dependency) so the dashboard stays a single `bokeh
serve` process with no build step -- the rotation itself runs entirely client-side via CustomJS
for a responsive drag, while every other control (embedding choice, coloring, thresholding)
round-trips to the Bokeh server, which is the right split: rotation must feel instantaneous,
everything else is infrequent enough that a server callback is simpler and plenty fast.
"""

from __future__ import annotations

import numpy as np

# Mirrors the JS in ROTATE_PROJECT_JS below field-for-field; keep the two in sync.
ROTATE_PROJECT_JS = """
const az = az_slider.value;
const el = el_slider.value;
const x3 = source.data['x3'];
const y3 = source.data['y3'];
const z3 = source.data['z3'];
const n = x3.length;
const x = new Array(n);
const y = new Array(n);
const size = new Array(n);
const alpha = new Array(n);
const cosA = Math.cos(az), sinA = Math.sin(az);
const cosE = Math.cos(el), sinE = Math.sin(el);
for (let i = 0; i < n; i++) {
  const xi = x3[i], yi = y3[i], zi = z3[i];
  if (Number.isNaN(xi)) { x[i] = NaN; y[i] = NaN; size[i] = 4; alpha[i] = 0; continue; }
  const x1 = xi * cosA + zi * sinA;
  const z1 = -xi * sinA + zi * cosA;
  const y1 = yi;
  const y2 = y1 * cosE - z1 * sinE;
  const z2 = y1 * sinE + z1 * cosE;
  x[i] = x1;
  y[i] = y2;
  const depth = Math.max(0, Math.min(1, (z2 + 3) / 6));
  size[i] = 3 + depth * 8;
  alpha[i] = 0.25 + depth * 0.65;
}
source.data['x'] = x;
source.data['y'] = y;
source.data['size'] = size;
source.data['alpha'] = alpha;
source.change.emit();
"""


def normalize_embedding(arr: np.ndarray) -> np.ndarray:
    """Center + scale each embedding to a comparable ~unit-variance range so the fixed depth
    normalization in the projection (assuming roughly [-3, 3]) is meaningful across PCA / UMAP /
    autoencoder / t-SNE, which otherwise live on wildly different native scales.
    """
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0
    return (arr - mean) / std


def rotate_project(
    x3: np.ndarray, y3: np.ndarray, z3: np.ndarray, azimuth: float, elevation: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
    cos_e, sin_e = np.cos(elevation), np.sin(elevation)
    x1 = x3 * cos_a + z3 * sin_a
    z1 = -x3 * sin_a + z3 * cos_a
    y1 = y3
    y2 = y1 * cos_e - z1 * sin_e
    z2 = y1 * sin_e + z1 * cos_e
    depth = np.clip((z2 + 3) / 6, 0, 1)
    size = 3 + depth * 8
    alpha = 0.25 + depth * 0.65
    nan_mask = np.isnan(x3)
    alpha = np.where(nan_mask, 0.0, alpha)
    return x1, y2, size, alpha
