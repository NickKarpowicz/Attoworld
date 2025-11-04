"""Tools for setting the style and appearance of plots."""

import io
from dataclasses import dataclass

import marimo as mo
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes


def showmo():
    """Display a plot as an svg in marimo notebooks."""
    svg_buffer = io.StringIO()
    plt.savefig(svg_buffer, format="svg")
    return mo.output.append(mo.Html(svg_buffer.getvalue()))


def set_style(theme: str = "light", font_size: int = 11):
    """Set colors and fonts for matplotlib plots.

    Args:
        theme (str): Select color theme.
                    Options:
                        ```light```: color-blind friendly colors (default)
                        ```nick_dark```: dark mode that matches Nick's slides
        font_size (int): Font size to apply to all plots.

    """
    plt.rcParams.update(
        {
            "font.size": font_size,
            "xtick.labelsize": 0.9 * font_size,
            "ytick.labelsize": 0.9 * font_size,
            "legend.fontsize": 0.9 * font_size,
            "figure.autolayout": True,
        }
    )
    match theme:
        case "nick_dark":
            plt.rcParams.update(
                {
                    "font.sans-serif": [
                        "Helvetica",
                        "Nimbus Sans L",
                        "Arial",
                        "Verdana",
                        "Nimbus Sans L",
                        "DejaVu Sans",
                        "Liberation Sans",
                        "Bitstream Vera Sans",
                        "sans-serif",
                    ],
                    "font.family": "sans-serif",
                    "axes.prop_cycle": cycler(
                        color=["cyan", "magenta", "orange", "blueviolet", "lime"]
                    ),
                    "figure.facecolor": "#171717",
                    "figure.edgecolor": "#171717",
                    "savefig.facecolor": "#171717",
                    "savefig.edgecolor": "#171717",
                    "axes.facecolor": "black",
                    "text.color": "white",
                    "axes.edgecolor": "white",
                    "axes.labelcolor": "white",
                    "xtick.color": "white",
                    "ytick.color": "white",
                    "grid.color": "white",
                    "lines.color": "white",
                    "image.cmap": "magma",
                }
            )
        # Light case is combined with _, which will capture anything else that didn't match.
        # It must be the last case, for that reason.
        case "light" | _:
            plt.rcParams.update(
                {
                    "font.sans-serif": [
                        "Helvetica",
                        "Nimbus Sans L",
                        "Arial",
                        "Verdana",
                        "DejaVu Sans",
                        "Liberation Sans",
                        "Bitstream Vera Sans",
                        "sans-serif",
                    ],
                    "font.family": "sans-serif",
                    # colorblind-friendly color cycle from https://gist.github.com/thriveth/8560036
                    "axes.prop_cycle": cycler(
                        color=[
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    "figure.facecolor": "white",
                    "figure.edgecolor": "white",
                    "savefig.facecolor": "white",
                    "savefig.edgecolor": "white",
                    "axes.facecolor": "white",
                    "text.color": "black",
                    "axes.edgecolor": "black",
                    "axes.labelcolor": "black",
                    "xtick.color": "black",
                    "ytick.color": "black",
                    "grid.color": "black",
                    "lines.color": "black",
                    "image.cmap": "viridis",
                }
            )


def label_letter(
    letter: str = "a",
    axis: Axes = plt.gca(),
    style: str = "Nature",
    x_position: float = -0.14,
    y_position: float = 1.08,
):
    """Put a letter in the corner of a set of axes to label them.

    Args:
        letter (str): The letter to use
        axis: The axes to use (default is current ones)
        style (str): The journal style to apply. Options are ```Nature```, ```Science```, and ```OSA```
        x_position (float): where to put the label horizontally relative to the axes of the figure
        y_position (float): vertical position

    """
    letter_string = f"{letter}"
    fontweight = "normal"
    match style:
        case "Nature":
            letter_string = letter_string.lower()
            fontweight = "bold"
        case "Science":
            letter_string = letter_string.upper()
            fontweight = "bold"
        case "OSA":
            letter_string = "(" + letter_string.lower() + ")"
            fontweight = "bold"

    axis.text(
        x_position,
        y_position,
        letter_string,
        ha="center",
        transform=axis.transAxes,
        fontweight=fontweight,
    )


@dataclass
class Char:
    """Contains useful characters and strings for plot labels.

    Examples:
        >>> plt.xlabel(aw.plot.Char.wavelength_micron)
        >>> plt.ylabel(f'{aw.plot.Char.theta} ({aw.plot.Char.degrees})')

    """

    mu: str = "\u03bc"
    phi: str = "\u03a6"
    lam: str = "\u03bb"
    theta: str = "\u03b8"
    micron: str = "\u03bcm"
    degrees: str = "\u00b0"
    angstrom: str = "\u212b"
    wavelength_micron: str = "Wavelength (\u03bcm)"
    energy_microjoule: str = "Energy (\u03bcJ)"
