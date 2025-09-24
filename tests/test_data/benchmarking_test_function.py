import numpy as np
from numba import njit


def mandelbrot(c: complex, n: int = 1000) -> int:
    """
    Devuelve el número de iteraciones hasta que diverge,
    o n si permanece estable.
    """
    z = 0
    for i in range(n):
        z = z*z + c
        if abs(z) > 2:
            return i
    return n


@njit
def mandelbrot_numba(c: complex, n: int = 1000) -> int:
    z = 0 + 0j
    for i in range(n):
        z = z * z + c
        if (z.real*z.real + z.imag*z.imag) > 4.0:
            return i
    return n


@njit
def mandelbrot_grid_numba(xmin, xmax, ymin, ymax, width, height, n=1000):
    """
    Calcula el conjunto de Mandelbrot en una rejilla.

    Retorna: array 2D con el número de iteraciones hasta divergencia.
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
    Calcula el conjunto de Mandelbrot en una rejilla.

    Retorna: array 2D con el número de iteraciones hasta divergencia.
    """
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            x = xmin + (xmax - xmin) * j / width
            y = ymin + (ymax - ymin) * i / height
            c = x + 1j * y
            result[i, j] = mandelbrot(c, n)
    return result

