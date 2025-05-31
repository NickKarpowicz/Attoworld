import numpy as np
from matplotlib import pyplot as plt
import pandas
import scipy.optimize as opt
import scipy.signal

e = 1.602176462e-19
hbar = 1.05457159682e-34
m = 9.1093818872e-31
eps0 = 8.854187817e-12
kB = 1.380650324e-23
c = 299792458


def get_fwhm(t, x, no_envelope: bool = False):
    """
    This function calculates the full-width-at-half-maximum:
    FWHM is the full-width of the intensity profile I(t) = |A(t)|^2 at half-maximum, where A(t) is the envelope of the signal.

    Args:
    x: signal in time domain vector
    t: time vector

    RETURNS:
    fwhm: fwhm_val[0]

    if no_envelope == True it computes the FWHM without taking the square modulus and without computing envelope (assuming input is intensity envelope already)
    """
    if no_envelope and np.min(x) < 0:
        raise ValueError('tried to compute FWHM of an envelope with negative values')
    envelope = np.abs(scipy.signal.hilbert(x) ** 2)
    half_max = np.max(envelope) / 2
    finet = np.linspace(t[0], t[-1], len(t)*50)
    fineEnvelope = np.interp(finet, t, envelope)
    if no_envelope:
        fineEnvelope = np.interp(finet, t, x)
        half_max = np.max(x) / 2

    signs = np.sign(fineEnvelope - half_max)
    fwhm_val = np.abs(finet[np.argwhere(signs == 1)[0]] - finet[np.argwhere(signs == 1)[-1]])
    return fwhm_val[0]

def gaussian(height, center_x, center_y, width_x, width_y, offset):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: offset + height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def eliminate_broken_pixels(data):
    """Eliminates the broken pixels of the camera rayCi I was using"""
    data[395:409, 950:1060] = 0
    return data

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, np.min(data)

def moments_peak(data):
    """Returns (height, x, y, width_x, width_y, np.min(data)), that is,
    the gaussian parameters of a 2D distribution by calculating its
    peak and moments"""
    total = data.sum()
    X, Y = np.indices(data.shape)
    maxindex = np.unravel_index(np.argmax(data), data.shape)
    x, y = maxindex[0], maxindex[1]
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, np.min(data)

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments_peak(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = opt.leastsq(errorfunction, params)
    return p

def plot_crosssect(data, fitfunct=None):
    maxindex = np.unravel_index(np.argmax(data), data.shape)
    y, x = maxindex[0], maxindex[1]
    Y, X = np.indices(data.shape)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(X[y], data[y])
    if fitfunct is not None:
        ax[0].plot(X[y], fitfunct(X[y]*0. + y, X[y]))
    ax[1].plot(Y[:,x], data[:,x])
    if fitfunct is not None:
        ax[1].plot(Y[:,x], fitfunct(Y[:,x], Y[:,x]*0. + x))
    fig.show()

def cut_tail(data, radius, pixelsize=1., center=None):
    radius=radius/pixelsize
    if center == None:
        center = np.unravel_index(np.argmax(data), data.shape)
    y, x = center[0], center[1]
    Y, X = np.indices(data.shape)
    cut_data = data.copy()
    cut_data[np.sqrt((Y-y)**2+(X-x)**2) > radius] = 0.
    return cut_data

def cut_trace(t, x, radius, center=None):
    if center == None:
        center = np.argmax(abs(x))
    cut_trace = x.copy()
    cut_trace[abs(t-t[center])>radius] = 0
    return cut_trace

def integral2d(data, pixelsize=1.):
    return data.sum()*pixelsize**2

def trace_integral(t, x):
    dt = np.gradient(t)
    f = x #np.array([(i+j)/2 for (i,j) in zip(x[:-1], x[1:])])
    return (dt*f).sum()

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()


def profile_analysis(profile_file, trace_file = None, magnification=1., pixelsize=5., power=20., reprate=4., ROI_diam:int=100, forced_background=None, trace_cutoff_radius = 50., cutoff_gaus_fit_profile = 25.):
    """if trace_file is None, this function only loads, fits and plots the beam profile; if trace_file is given, the function additionally calculates the peak intensity and field.

    ARGUMENTS:
        profile_file: name of the file containing the beam profile; the file (txt or csv) contains the 2D array of the beam profile

    OPTIONAL ARGUMENTS:
        trace_file: None; name of the file containing the trace; the file (txt or csv) contains the time and field arrays as tab-separated columns
        magnification: 1.; ratio of lens-camera/lens-z0 lengths; This is used to give the correct size of the beam profile at z0 in um
        pixelsize: 5. um; pixel size of the camera
        power: 20. mW; power of the laser beam; this is used to calculate the peak intensity
        reprate: 4 kHz; repetition rate of the laser; this is used to calculate the peak intensity
        ROI_diam: 100; diameter of the region of interest in pixels; the center of the ROI is the maximum of the camera data (which is assumed to be not saturated)
        cutoff_gaus_fit_profile: 25. pixels; the 2d gaussian is fitted to the beam profile up to this distance from the peak
        forced_background: None; if not None, the subtracted background is set to this value; otherwise it's computed from the outer ring of the ROI (camera data)
        trace_cutoff_radius: 50. fs; for intensity calculation, the integral of the trace is computed only in the range t[peak]-trace_cutoff_radius < t < t[peak]+trace_cutoff_radius"""

    pixelsize=pixelsize/magnification
    data=pandas.read_csv(profile_file)

    dataval=np.array(data.values)
    #print('eliminating fixed broken pixels rayCi camera')
    #dataval=eliminate_broken_pixels(dataval)
    maxindex = np.unravel_index(np.argmax(dataval), dataval.shape)
    print('diameter of Region Of Interest (ROI): ', ROI_diam, ' pixels')
    dataval=np.array(data.values[maxindex[0]-int(ROI_diam/2):maxindex[0]+int(ROI_diam/2),maxindex[1]-int(ROI_diam/2):maxindex[1]+int(ROI_diam/2)])
    maxindex = np.unravel_index(np.argmax(dataval), dataval.shape)

    background = 0.
    count = 0
    print('calculating background intensity outside a circle whose diameter is 3/4 of the ROI diameter')
    for i, j in np.ndindex(dataval.shape):
        if (i-maxindex[0])**2 + (j-maxindex[1])**2 > (ROI_diam*3/8)**2:
            background += dataval[i,j]
            count += 1
    background /= count

    if forced_background is not None:
        background = forced_background
        print('subtracting background intensity: ', background)
    else:
        print('subtracting background intensity: ', background)

    dataval=np.maximum(dataval-background, dataval*0.)
    maxval = np.max(dataval)

    # gaussian fit
    param = moments_peak(dataval)
    param = fitgaussian(cut_tail(dataval, cutoff_gaus_fit_profile))
    X, Y = np.indices(dataval.shape)
    fitted_funct = gaussian(*param)(X, Y)
    print('beam w [um], resp. in x and y axis:', 2*param[3]*pixelsize, 2*param[4]*pixelsize)
    print('beam 1/e^2 diameter [um]: ', 4*param[3]*pixelsize, 4*param[4]*pixelsize)

    if  trace_file is not None:

        trace = pandas.read_csv(trace_file, sep='\t')

        # time in fs, field in a.u.
        intensity = power *1e-3/reprate *1e-3\
                / integral2d(fitted_funct, pixelsize=pixelsize) * np.max(fitted_funct) *1e8 \
                / trace_integral(trace['delay (fs)'], cut_trace(trace['delay (fs)'], trace['field (a.u.)']**2, trace_cutoff_radius)) * np.max(trace['field (a.u.)']**2) *1e15

        intensityFromRawCameraData = power *1e-3/reprate *1e-3\
                / integral2d(dataval[maxindex[0]-int(ROI_diam/2):maxindex[0]+int(ROI_diam/2),maxindex[1]-int(ROI_diam/2):maxindex[1]+int(ROI_diam/2)], pixelsize=pixelsize) * np.max(dataval) *1e8 \
                / trace_integral(trace['delay (fs)'], cut_trace(trace['delay (fs)'], trace['field (a.u.)']**2, trace_cutoff_radius)) * np.max(trace['field (a.u.)']**2) *1e15

        temporal_fwhm = get_fwhm(np.array(trace['delay (fs)']), np.array(trace['field (a.u.)']))
        intensityFromTemporalFWHM = power *1e-3/reprate *1e-3\
                / integral2d(dataval[maxindex[0]-int(ROI_diam/2):maxindex[0]+int(ROI_diam/2),maxindex[1]-int(ROI_diam/2):maxindex[1]+int(ROI_diam/2)], pixelsize=pixelsize) * np.max(dataval) *1e8 \
                / (np.sqrt(2*np.pi)*temporal_fwhm/np.sqrt(8*np.log(2))) *1e15

        print('\nusing the gaussian fit of the beam profile and the integral of the trace^2:\n '
            'peak intensity (sub-cycle, W/cm2): {i:.3e}'.format(**{'i':intensity}))
        peak_field = np.sqrt(intensity*1e4 /c/eps0)
        print('peak field (sub-cycle, V/m): {f:.3e}'.format(**{'f':peak_field}))

        print('\nusing directly camera data, and the integral of the trace^2:\n '
              'peak intensity (sub-cycle, W/cm2): {i:.3e}'.format(**{'i':intensityFromRawCameraData}))
        peak_field = np.sqrt(intensityFromRawCameraData*1e4 /c/eps0)
        print('peak field (sub-cycle, V/m): {f:.3e}'.format(**{'f':peak_field}))

        print('\nusing directly camera data, and the temporal FWHM retrieved from the trace:\n '
              'peak intensity (sub-cycle, W/cm2): {i:.3e}'.format(**{'i':intensityFromTemporalFWHM}))
        peak_field = np.sqrt(intensityFromTemporalFWHM*1e4 /c/eps0)
        print('peak field (sub-cycle, V/m): {f:.3e}'.format(**{'f':peak_field}))

        fig, ax = plt.subplots(1, 1)
        time = trace['delay (fs)']
        field = trace['field (a.u.)']
        i_peak = np.argmax(field**2)
        ax.plot(time[np.abs(time-time[i_peak])<trace_cutoff_radius],
                field[np.abs(time-time[i_peak])<trace_cutoff_radius]**2)
        plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(dataval)
    plt.show()

    plot_crosssect(dataval, fitfunct=gaussian(*param))