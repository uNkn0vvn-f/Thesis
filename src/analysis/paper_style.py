"""Shared, print-safe styling for the thesis's static research figures.

Figures are authored near their final manuscript width, with embedded PDF fonts.
Series identity uses line styles and marker shapes as well as grey levels.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

WIDTH = 6.27
INK = '#111111'
MID = '#555555'
LIGHT = '#888888'
GRID = '#dddddd'
ROLE_COLORS = {'developer': INK, 'enabler': MID, 'adopter': '#777777'}
ROLE_MARKERS = {'developer': 'o', 'enabler': 's', 'adopter': '^'}
ROLE_LINES = {'developer': '-', 'enabler': '--', 'adopter': '-.'}


def set_style():
    """Matplotlib defaults for the thesis figures: serif text, thin black axes, embedded
    fonts."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = [font for font in ('Times New Roman', 'Liberation Serif', 'DejaVu Serif') if font in available]
    plt.rcParams.update({
        'font.family': family, 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'axes.titleweight': 'normal',
        'axes.titlepad': 8, 'axes.linewidth': .6, 'axes.edgecolor': INK,
        'axes.labelcolor': INK, 'text.color': INK,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'legend.fontsize': 8, 'legend.frameon': False,
        'legend.handlelength': 2.6, 'legend.columnspacing': 1.3,
        'lines.linewidth': 1.1, 'lines.markersize': 3.5,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'hatch.linewidth': .35,
    })


def style_axes(ax, grid='y'):
    """Keep only the bottom and left spines and draw a light grid along `grid` behind the
    data."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(.6)
        ax.spines[side].set_color(INK)
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.4)
    ax.set_axisbelow(True)
    ax.tick_params(width=.6, length=3)


def save_figure(fig, path):
    """Save matching high-resolution raster and vector files without layout recropping.

    The PDF carries no creation date, so a rerun on the same inputs reproduces the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=400, facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), dpi=400, facecolor='white',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
