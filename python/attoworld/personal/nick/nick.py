"""A few small functions that I might take apart and put into the main module."""

from cycler import cycler
from matplotlib import rcParams


def dark_plot():
    """Use a dark style for matplotlib plots."""
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [
        "Helvetica",
        "Arial",
        "Verdana",
        "DejaVu Sans",
        "Liberation Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]
    rcParams["axes.prop_cycle"] = cycler(
        color=["cyan", "magenta", "orange", "blueviolet", "lime"]
    )
    rcParams["figure.facecolor"] = "#171717"
    rcParams["figure.edgecolor"] = rcParams["figure.facecolor"]
    rcParams["savefig.facecolor"] = rcParams["figure.facecolor"]
    rcParams["savefig.edgecolor"] = rcParams["figure.facecolor"]
    rcParams["axes.facecolor"] = "black"
    rcParams["text.color"] = "white"
    rcParams["axes.edgecolor"] = "white"
    rcParams["axes.labelcolor"] = "white"
    rcParams["xtick.color"] = "white"
    rcParams["ytick.color"] = "white"
    rcParams["grid.color"] = "white"
    rcParams["lines.color"] = "white"
