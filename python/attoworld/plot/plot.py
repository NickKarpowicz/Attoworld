import marimo as mo
import io
import matplotlib.pyplot as plt

def showmo():
    """
    Helper function to plot as an svg to have vector plots in marimo notebooks
    """
    # Contributed by Nick Karpowicz
    svg_buffer = io.StringIO()
    plt.savefig(svg_buffer, format='svg')
    return mo.Html(svg_buffer.getvalue())
