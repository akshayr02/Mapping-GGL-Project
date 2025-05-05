# Third-party imports
import numpy as np
import os
import re
from numba import njit

import pickle
from scipy.interpolate import CubicSpline
from scipy import interpolate

def derivative_from_samples(x:float, xs:np.array, fs:np.array) -> float:
    '''
    Calculates the derivative of the function f(x) which is sampled as fs at values xs
    Approximates the function as quadratic using the samples and Legendre polynomials
    Args:
        x: Point at which to calculate the derivative
        xs: Sample locations
        fs: Value of function at sample locations
    '''
    from scipy.interpolate import lagrange
    ix, _ = find_closest_index_value(x, xs)
    if ix == 0:
        imin, imax = (0, 1) if x < xs[0] else (0, 2) # Tuple braces seem to be necessarry here
    elif ix == len(xs)-1:
        nx = len(xs)
        imin, imax = (nx-2, nx-1) if x > xs[-1] else (nx-3, nx-1) # Tuple braces seem to be necessarry here
    else:
        imin, imax = ix-1, ix+1
    poly = lagrange(xs[imin:imax+1], fs[imin:imax+1])
    return poly.deriv()(x)


def logspace(xmin:float, xmax:float, nx:int) -> np.ndarray:
    '''
    Return a logarithmically spaced range of numbers
    '''
    return np.logspace(np.log10(xmin), np.log10(xmax), nx)


def find_closest_index_value(x:float, xs:np.array) -> tuple:
    '''
    Find the index, value pair of the closest values in array 'arr' to value 'x'
    '''
    idx = (np.abs(xs-x)).argmin()
    return idx, xs[idx]


def is_array_monotonic(x:np.array) -> bool:
    '''
    Returns True iff the array contains monotonically increasing values
    '''
    return np.all(np.diff(x) > 0.)


def is_array_linear(x:np.array, atol=1e-8) -> bool:
    '''
    Returns True iff the array is linearly spaced
    '''
    return np.isclose(np.all(np.diff(x)-np.diff(x)[0]), 0., atol=atol)



def readTableFile(filename):
    """
    Reads a multi-column file and returns a dictionary with the first column as keys
    and the remaining columns as numpy array values. Important: We are storing the values as longdouble for the adequate precision!
    
    Parameters:
    - filename (str): The path to the file to read.
    
    Returns:
    - dict: A dictionary where keys are from the first column and values are numpy arrays of remaining columns.
    """
    data_dict = {}

    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist!")
    
    with open(filename, 'r') as file:
        for line in file:
            # Split by any whitespace (tab or space)
            columns = re.split(r'\s+', line.strip())
            # Ensure there are at least two values (key + at least one value)
            if len(columns) >= 2:
                key = int(columns[0])
                values = np.array(list(map(np.longdouble, columns[1:])))
                data_dict[key] = values
            else:
                # Optionally, log or print a message about the skipped line
                print(f"Skipping line due to incorrect format: {line.strip()}")
                
    return data_dict






def save_splines(splines_dict, filename):
    """
    Save a dictionary of CubicSpline objects to a file.
    
    Parameters:
    - splines_dict (dict): Dictionary where each value is a CubicSpline object.
    - filename (str): The file path to save the data to.
    """
    # Prepare a dictionary to store serializable data
    serializable_dict = {}
    for key, spline in splines_dict.items():
        # Store x and y values, which is enough to reconstruct the spline
        serializable_dict[key] = {
            'x': spline.x,
            'y': spline(spline.x),  # Evaluating spline at its knots to get y values
        }
    
    # Save the dictionary to a file
    with open(filename, 'wb') as file:
        pickle.dump(serializable_dict, file)


def load_splines(filename):
    """
    Load a dictionary of CubicSpline objects from a file.
    
    Parameters:
    - filename (str): The file path to load the data from.
    
    Returns:
    - dict: Dictionary where each value is a restored CubicSpline object.
    """
    try:
        data = np.load(filename,allow_pickle=True).item()
    except:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    
    # Reconstruct the splines from the stored data
    splines_dict = {}
    for key, spline_data in data.items():
        # splines_dict[key] = CubicSpline(x=spline_data['x'],y=spline_data['y'])
        splines_dict[key] = interpolate.interp1d(x=spline_data['x'],y=spline_data['y'],kind='linear')
    
    return splines_dict



@njit
def custom_interpolate(ell1_values, ell2_values, phi_values, bkappa_grid, ell1, ell2, phi):
    """
    Performs trilinear interpolation on a 3D grid of `bkappa` values based on the given 
    `ell1`, `ell2`, and `phi` coordinates. 

    Parameters
    ----------
    ell1_values : numpy.ndarray
        1D array of `ell1` coordinate values in ascending order.
    ell2_values : numpy.ndarray
        1D array of `ell2` coordinate values in ascending order.
    phi_values : numpy.ndarray
        1D array of `phi` coordinate values in ascending order.
    bkappa_grid : numpy.ndarray
        3D array containing `bkappa` values, where each entry corresponds to a specific 
        combination of `ell1`, `ell2`, and `phi` values.
    ell1 : float
        Target `ell1` coordinate for interpolation.
    ell2 : float
        Target `ell2` coordinate for interpolation.
    phi : float
        Target `phi` coordinate for interpolation.

    Returns
    -------
    float
        Interpolated `bkappa` value at the specified (`ell1`, `ell2`, `phi`) point.

    Notes
    -----
    This function assumes that `ell1_values`, `ell2_values`, and `phi_values` are 
    uniformly spaced and ordered in ascending fashion. The method will perform 
    boundary checks and handle cases where the coordinates fall at the edges of 
    the grid.
    """
    i = np.searchsorted(ell1_values, ell1) - 1
    j = np.searchsorted(ell2_values, ell2) - 1
    k = np.searchsorted(phi_values, phi) - 1
    
    # Handle boundaries
    i = max(min(i, len(ell1_values) - 2), 0)
    j = max(min(j, len(ell2_values) - 2), 0)
    k = max(min(k, len(phi_values) - 2), 0)
    
    x1, x2 = ell1_values[i], ell1_values[i+1]
    y1, y2 = ell2_values[j], ell2_values[j+1]
    z1, z2 = phi_values[k], phi_values[k+1]
    
    # Trilinear interpolation
    f111 = bkappa_grid[i, j, k]
    f112 = bkappa_grid[i, j, k+1]
    f121 = bkappa_grid[i, j+1, k]
    f122 = bkappa_grid[i, j+1, k+1]
    f211 = bkappa_grid[i+1, j, k]
    f212 = bkappa_grid[i+1, j, k+1]
    f221 = bkappa_grid[i+1, j+1, k]
    f222 = bkappa_grid[i+1, j+1, k+1]
    
    f11 = (f111 * (z2 - phi) + f112 * (phi - z1)) / (z2 - z1)
    f12 = (f121 * (z2 - phi) + f122 * (phi - z1)) / (z2 - z1)
    f21 = (f211 * (z2 - phi) + f212 * (phi - z1)) / (z2 - z1)
    f22 = (f221 * (z2 - phi) + f222 * (phi - z1)) / (z2 - z1)
    
    f1 = (f11 * (y2 - ell2) + f12 * (ell2 - y1)) / (y2 - y1)
    f2 = (f21 * (y2 - ell2) + f22 * (ell2 - y1)) / (y2 - y1)
    
    return (f1 * (x2 - ell1) + f2 * (ell1 - x1)) / (x2 - x1)


@njit
def custom_interpolate_1d(x_values, y_values, x):
    """
    Performs linear interpolation on a 1D grid of `y` values based on the given `x` coordinate.

    Parameters
    ----------
    x_values : numpy.ndarray
        1D array of `x` coordinate values in ascending order.
    y_values : numpy.ndarray
        1D array of `y` values corresponding to each `x` coordinate in `x_values`.
    x : float
        Target `x` coordinate for interpolation.

    Returns
    -------
    float
        Interpolated `y` value at the specified `x` point.

    Notes
    -----
    This function uses linear interpolation to estimate the `y` value at the given `x` coordinate.
    The function assumes `x_values` are in ascending order. Boundary cases are handled by clamping
    the interpolation interval to the edges of the `x_values` array if `x` is outside the range.
    """
    # Find the interval in x_values that contains x
    i = np.searchsorted(x_values, x) - 1

    # Handle boundaries: make sure i is within the valid range
    i = max(min(i, len(x_values) - 2), 0)
    
    # Get the x-values and y-values at the interval boundaries
    x1, x2 = x_values[i], x_values[i+1]
    y1, y2 = y_values[i], y_values[i+1]

    # Linear interpolation formula
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)






@njit
def cubic_interp1d(x, x_vals, y_vals):
    """
    Perform 1D cubic interpolation on a given set of points.
    
    Parameters
    ----------
    x : float
        Target point for interpolation.
    x_vals : numpy.ndarray
        1D array of x-coordinates, must be sorted in ascending order.
    y_vals : numpy.ndarray
        1D array of y-coordinates corresponding to x_vals.
    
    Returns
    -------
    float
        Interpolated value at x.
    """
    n = len(x_vals)
    
    # Find the interval for x
    i = np.searchsorted(x_vals, x) - 1
    i = max(min(i, n - 2), 1)  # Ensure valid indices for cubic interpolation
    
    # Select four surrounding points for cubic interpolation
    x0 = x_vals[i - 1] if i > 0 else x_vals[i]
    x1 = x_vals[i]
    x2 = x_vals[i + 1]
    x3 = x_vals[i + 2] if i + 2 < n else x_vals[i + 1]
    
    y0 = y_vals[i - 1] if i > 0 else y_vals[i]
    y1 = y_vals[i]
    y2 = y_vals[i + 1]
    y3 = y_vals[i + 2] if i + 2 < n else y_vals[i + 1]
    
    # Compute cubic interpolation
    t = (x - x1) / (x2 - x1)
    a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    b = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3
    c = -0.5 * y0 + 0.5 * y2
    d = y1
    
    return a * t**3 + b * t**2 + c * t + d

@njit
def cubic_interpolate_3d(ell1_values, ell2_values, phi_values, bkappa_grid, ell1, ell2, phi):
    """
    Perform cubic interpolation in a 3D grid.
    
    Parameters
    ----------
    ell1_values, ell2_values, phi_values : numpy.ndarray
        1D arrays of grid coordinates for each axis.
    bkappa_grid : numpy.ndarray
        3D array of values to interpolate.
    ell1, ell2, phi : float
        Target coordinates for interpolation.
    
    Returns
    -------
    float
        Interpolated value at the target coordinates.
    """
    # Interpolate along phi for each fixed (ell1, ell2)
    phi_slices = np.zeros((len(ell1_values), len(ell2_values)))
    for i in range(len(ell1_values)):
        for j in range(len(ell2_values)):
            phi_slices[i, j] = cubic_interp1d(phi, phi_values, bkappa_grid[i, j, :])
    
    # Interpolate along ell2 for each fixed ell1
    ell2_slices = np.zeros(len(ell1_values))
    for i in range(len(ell1_values)):
        ell2_slices[i] = cubic_interp1d(ell2, ell2_values, phi_slices[i, :])
    
    # Interpolate along ell1
    return cubic_interp1d(ell1, ell1_values, ell2_slices)