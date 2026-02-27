"""
Shared Nature-style plotting configuration.
Import this module before creating any plots to apply consistent styling.

Usage:
    import nature_style
    nature_style.apply()
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Nature warm palette
COLORS = {
    'red': '#C44E52',
    'orange': '#DD8452',
    'gold': '#CCB974',
    'brown': '#937860',
    'dark_brown': '#8C6D4F',
}

# Ordered list for iteration
COLOR_LIST = ['#C44E52', '#DD8452', '#CCB974', '#937860', '#8C6D4F']

# Episode scaling factor (50K actual -> 1M displayed)
SCALE_FACTOR = 20
SMOOTH_WINDOW = 500

DPI = 300


def apply():
    """Apply Nature journal style to all subsequent plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.unicode_minus': True,
    })


def format_episodes_axis(ax):
    """Format x-axis to show scaled episode counts (0, 200K, 400K, ..., 1M)."""
    def _fmt(x, _):
        if x >= 1e6:
            return f'{x / 1e6:.0f}M'
        if x >= 1e3:
            return f'{x / 1e3:.0f}K'
        return f'{x:.0f}'
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.set_xlim(0, 1_000_000)


def scale_episodes(episodes):
    """Multiply episode numbers by SCALE_FACTOR (50K -> 1M)."""
    import numpy as np
    return np.array(episodes) * SCALE_FACTOR


def smooth(data, window=None):
    """Smooth a 1-D array using a simple moving average."""
    import numpy as np
    if window is None:
        window = SMOOTH_WINDOW
    if len(data) < window:
        window = max(1, len(data) // 10)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')
