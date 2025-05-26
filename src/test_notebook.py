import marimo

__generated_with = "0.13.11"
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
def _(aw, np, plt):
    def interpolate(x_in, y_in, x_out, neighbors, extrapolate: bool = False):
        sort_order = np.argsort(x_in)
        x_in_sorted = x_in[sort_order]
        y_in_sorted = y_in[sort_order]
        y_out = np.zeros(x_out.shape)

        def interpolate_point(x):
            location = np.searchsorted(x_in_sorted, x_out[_i], side='left')

            if (((location == 0) and x != x_in_sorted[0]) or (location >= len(x_in_sorted))):
                #points outside the range of x_in (extrapolation)
                if extrapolate:
                    return 3.0
                else:
                    return 0.0
        
            elif x == x_in_sorted[location]:
                #case if x is exactly in the x_in array
                #check if the next point also matches, careful not to go outside the array
                return 1.0
            
            else:
                #normal interior point
                return 2.0
                #x is not in the array

        for _i in range(len(x_out)):
            y_out[_i] = interpolate_point(x_out[_i])

        return y_out

    x = np.sqrt(np.linspace(0.0,10.0,16))
    y = x**4
    plt.plot(interpolate(x,y,np.array([0, 0.5, 1.0, 2.0, 4]), 1),'x')
    aw.plot.showmo()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
