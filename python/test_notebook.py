import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import attoworld as aw
    import matplotlib.pyplot as plt
    return aw, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""## Check convergence of numerical derivatives""")
    return


@app.cell
def _(aw, np, plt):
    def convergence_check(order, N_pts):
        x = np.linspace(0.0, 2*np.pi, N_pts+1)[0:-1]
        dx = x[1]-x[0]
        y = np.cos(x)
        y_derivative_analytic = -np.sin(x)
        y_derivative = aw.numeric.uniform_derivative(y,1,order,boundary='periodic')/dx
        return np.max(np.abs(y_derivative - y_derivative_analytic))

    N_pts_range = [16, 32, 64, 128, 256, 512]
    order_range = range(1,6)

    convergence_data = np.array([[convergence_check(_order, _n) for _n in N_pts_range] for _order in order_range])

    lines = plt.loglog(N_pts_range,convergence_data.T)
    for order, line in zip(order_range, lines):
        line.set_label(f"{order}")
    plt.xlabel("Number of grid points")
    plt.ylabel("Mean error")
    plt.ylim(1e-15, 0.1)
    plt.legend()
    aw.plot.showmo()
    return


@app.cell
def _(mo):
    mo.md(r"""## Check interpolate()""")
    return


@app.cell
def _(aw, np, plt):
    def plot_wiggles():
        x = np.real(np.linspace(0.0,17.0,24))
        x_fine = np.real(np.linspace(0.0,17.0,1024))
        y = np.sin(x**2/10)
        y_fine = np.sin(x_fine**2/10)
        x2 = np.linspace(0.0,19.0,60)
        plt.plot(x,y,'o')
        plt.plot(x_fine,y_fine)
        plt.plot(x2,aw.numeric.interpolate(x2, x,y, 3, extrapolate=False),'x')

    plot_wiggles()
    aw.plot.showmo()
    return


@app.cell
def _(aw, np, plt):
    ix = np.linspace(-5, 5, 18) - 1.2
    _dx = ix[1]-ix[0]
    iy = np.exp(-ix**2/2)
    plt.plot(iy)

    interpolated_position, interpolated_max = aw.attoworld_rs.find_maximum_location(iy, 3)
    raw_max = np.max(iy)
    raw_argmax = np.argmax(iy)

    first_intercept = aw.attoworld_rs.find_first_intercept(iy, 0.5, 2)
    last_intercept = aw.attoworld_rs.find_last_intercept(iy, 0.5, 2)
    fwhm = aw.attoworld_rs.fwhm(iy, _dx)

    print(f"find_maximum_location gives maximum at ({interpolated_position},{interpolated_max})")
    print(f"raw indexing gives maximum at ({raw_argmax},{raw_max})")
    print(f"first intercept float index: {first_intercept}")
    print(f"fwhm: {_dx * (last_intercept - first_intercept)}")
    print(f"fwhm (from function): {fwhm}")
    print(f"last intercept float index: {last_intercept}")

    aw.plot.showmo()
    return


@app.cell
def _(aw, np, plt):
    _y = np.array([0.00533447922478714, 0.030093074775186003, 0.12010641198185724, 0.3391492920920331, 0.6775490323319323, 0.957669456847167])
    _x = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    _sten =  np.array([-3.842395583015742, 5.937950230146847, -2.8755923850777814, 1.5170488819154688, 0.28266188230301, -0.019673026271801123])
    print(np.sum(_y * _sten))
    print(np.sum(_x*_sten))
    print(np.sum(aw.numeric.fornberg_stencil(0,_y[1::],0.5) * _x[1::]))
    plt.plot(_y,_x)
    aw.plot.showmo()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
