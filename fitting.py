#coding = utf8

import sys, os
import numpy as np 
from   scipy.optimize import curve_fit

def fits(fit_func, xdata, ydata, rangea = 0, rangeb = 0, p0 = [], bounds = ()):
    """
    This function is used to fit existing data in numpy array
    fiting function and (X, Y) should be provided
    """
    if   len(p0) >= 1 and len(bounds) >= 1:
        popt, pcov = curve_fit(fit_func, xdata, ydata, maxfev = 5000000, p0 = p0, bounds = bounds)
    elif len(p0) >= 1 and len(bounds) == 0:
        popt, pcov = curve_fit(fit_func, xdata, ydata, maxfev = 5000000, p0 = p0)
    elif len(p0) == 0 and len(bounds) >= 1:
        popt, pcov = curve_fit(fit_func, xdata, ydata, maxfev = 5000000, bounds = bounds)
    else:
        popt, pcov = curve_fit(fit_func, xdata, ydata, maxfev = 5000000)

    perr = np.sqrt(np.diag(pcov))
    residuals = ydata - fit_func(xdata, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.square(ydata - ydata.mean()).sum()
    R2 = 1 - (ss_res / ss_tot)
    print ('fitting R^2 = %.6f' %R2)
    print ('fitting parameters values: ' + ' '.join(map('{:.6f}'.format, popt)))
    print ('fitting parameters errors: ' + ' '.join(map('{:.6f}'.format, perr)))
    
    if rangeb == 0:
        xfit = np.linspace(xdata.min(), xdata.max(), 10000)
    else:
        xfit = np.linspace(rangea, rangeb, 10000)
    yfit = fit_func(xfit, *popt)
    return (popt, perr, xfit, yfit)