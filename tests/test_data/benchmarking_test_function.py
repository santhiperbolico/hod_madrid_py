import numpy as np
from numba import njit


def mandelbrot(c: complex, n: int = 1000) -> int:
    """
    Computes the Mandelbrot iteration count for a given complex number.

    Parameters
    ----------
    c : complex
        Complex number to evaluate.
    n : int, default=1000
        Maximum number of iterations.

    Returns
    -------
    int
        Number of iterations until divergence, or `n` if it remains bounded.
    """
    z = 0
    for i in range(n):
        z = z*z + c
        if abs(z) > 2:
            return i
    return n


@njit
def mandelbrot_numba(c: complex, n: int = 1000) -> int:
    """
   Computes the Mandelbrot iteration count for a given complex number,
   optimized with Numba JIT compilation.

   Parameters
   ----------
   c : complex
       Complex number to evaluate.
   n : int, default=1000
       Maximum number of iterations.

   Returns
   -------
   int
       Number of iterations until divergence, or `n` if it remains bounded.
   """
    z = 0 + 0j
    for i in range(n):
        z = z * z + c
        if (z.real*z.real + z.imag*z.imag) > 4.0:
            return i
    return n


@njit
def mandelbrot_grid_numba(xmin, xmax, ymin, ymax, width, height, n=1000):
    """
    Computes the Mandelbrot set on a 2D grid using Numba JIT compilation.

    Parameters
    ----------
    xmin, xmax : float
        Range of the real axis.
    ymin, ymax : float
        Range of the imaginary axis.
    width : int
        Number of points along the x-axis.
    height : int
        Number of points along the y-axis.
    n : int, default=1000
        Maximum number of iterations.

    Returns
    -------
    numpy.ndarray of shape (height, width), dtype=int32
        2D array with the number of iterations until divergence
        for each grid point.
    """
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            x = xmin + (xmax - xmin) * j / width
            y = ymin + (ymax - ymin) * i / height
            c = x + 1j * y
            result[i, j] = mandelbrot_numba(c, n)
    return result


def mandelbrot_grid(xmin, xmax, ymin, ymax, width, height, n=1000):
    """
    Computes the Mandelbrot set on a 2D grid.

    Parameters
    ----------
    xmin, xmax : float
        Range of the real axis.
    ymin, ymax : float
        Range of the imaginary axis.
    width : int
        Number of points along the x-axis.
    height : int
        Number of points along the y-axis.
    n : int, default=1000
        Maximum number of iterations.

    Returns
    -------
    numpy.ndarray of shape (height, width), dtype=int32
        2D array with the number of iterations until divergence
        for each grid point.
    """
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            x = xmin + (xmax - xmin) * j / width
            y = ymin + (ymax - ymin) * i / height
            c = x + 1j * y
            result[i, j] = mandelbrot(c, n)
    return result

