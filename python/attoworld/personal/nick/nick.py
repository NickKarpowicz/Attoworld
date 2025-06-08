from matplotlib import rcParams, cycler

def dark_plot():
    """
    Use a dark style for matplotlib plots.
    """
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Verdana', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    rcParams['axes.prop_cycle'] = cycler(color=["cyan", "magenta", "orange", "purple"])
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Verdana', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    rcParams['figure.facecolor'] = 'black'
    rcParams['figure.edgecolor'] = 'black'
    rcParams['savefig.facecolor'] = 'black'
    rcParams['savefig.edgecolor'] = 'black'
    rcParams['axes.facecolor'] = 'black'
    rcParams['text.color'] = 'white'
    rcParams['axes.edgecolor'] = 'white'
    rcParams['axes.labelcolor'] = 'white'
    rcParams['xtick.color'] = 'white'
    rcParams['ytick.color'] = 'white'
    rcParams['grid.color'] = 'white'
    rcParams['lines.color'] = 'white'
