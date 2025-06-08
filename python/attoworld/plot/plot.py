import marimo as mo
import io
import matplotlib.pyplot as plt
from cycler import cycler

def showmo():
    """
    Helper function to plot as an svg to have vector plots in marimo notebooks
    """
    svg_buffer = io.StringIO()
    plt.savefig(svg_buffer, format='svg')
    return mo.Html(svg_buffer.getvalue())

def set_style(mode: str = 'light'):
    """
    Set colors and fonts for matplotlib plots

    Args:
        mode (str): Select font and colors.
                    Options:
                        ```light```: color-blind friendly colors (default)
                        ```nick_dark```: dark mode that matches Nick's slides
    """
    match mode:
        case 'light':

            plt.rcParams.update({'font.sans-serif': ['Helvetica', 'Arial', 'Verdana', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']})
            plt.rcParams.update({'font.family': 'sans-serif'})
            #colorblind-friendly color cycle from https://gist.github.com/thriveth/8560036
            plt.rcParams.update({'axes.prop_cycle': cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])})
            plt.rcParams.update({'figure.facecolor': 'white'})
            plt.rcParams.update({'figure.edgecolor':'white'})
            plt.rcParams.update({'savefig.facecolor': 'white'})
            plt.rcParams.update({'savefig.edgecolor': 'white'})
            plt.rcParams.update({'axes.facecolor': 'white'})
            plt.rcParams.update({'text.color': 'black'})
            plt.rcParams.update({'axes.edgecolor': 'black'})
            plt.rcParams.update({'axes.labelcolor': 'black'})
            plt.rcParams.update({'xtick.color': 'black'})
            plt.rcParams.update({'ytick.color': 'black'})
            plt.rcParams.update({'grid.color': 'black'})
            plt.rcParams.update({'lines.color': 'black'})

        case 'nick_dark':
            plt.rcParams.update({'font.sans-serif': ['Helvetica', 'Arial', 'Verdana', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']})
            plt.rcParams.update({'font.family': 'sans-serif'})
            plt.rcParams.update({'axes.prop_cycle': cycler(color=["cyan", "magenta", "orange", "blueviolet", "lime"])})
            plt.rcParams.update({'figure.facecolor': '#171717'})
            plt.rcParams.update({'figure.edgecolor': '#171717'})
            plt.rcParams.update({'savefig.facecolor': '#171717'})
            plt.rcParams.update({'savefig.edgecolor': '#171717'})
            plt.rcParams.update({'axes.facecolor': 'black'})
            plt.rcParams.update({'text.color': 'white'})
            plt.rcParams.update({'axes.edgecolor': 'white'})
            plt.rcParams.update({'axes.labelcolor': 'white'})
            plt.rcParams.update({'xtick.color': 'white'})
            plt.rcParams.update({'ytick.color': 'white'})
            plt.rcParams.update({'grid.color': 'white'})
            plt.rcParams.update({'lines.color': 'white'})
