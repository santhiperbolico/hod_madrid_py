from tempfile import TemporaryDirectory

from tests.test_data.benchmarking_test_function import mandelbrot_grid, mandelbrot_grid_numba
from benchmarking_analysis import benchmarking_analysis


def test_benchmarking_analysis():
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 800, 600
    with TemporaryDirectory() as tmpdir:
        result = benchmarking_analysis(
            mandelbrot_grid,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=width, height=height,
            output_path=tmpdir,
        )
        result_numba = benchmarking_analysis(
            mandelbrot_grid_numba,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=width, height=height,
            output_path=tmpdir,
        )

    assert isinstance(result, dict)
    assert isinstance(result_numba, dict)
    assert result["runtime_s"] > result_numba["runtime_s"]
