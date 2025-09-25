import sys
import argparse
import logging
import os
import json
from typing import Any, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from memory_profiler import memory_usage
import time

from produce_hod_mock import main


def get_params(argv: list[str]) -> argparse.Namespace:
    """
    Función que genera los argumentos de download_data.

    Parameters
    ----------
    argv: list[str]
        Lista de argumentos del job.

    Returns
    -------
    args: argparse.Namespace
        Argumentos.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="output/")
    return parser.parse_args(argv)


@dataclass
class BenchStats:
    func_name: str
    runtime_s: float
    samples: int
    approx_interval_s: float
    peak_mem_mb: float
    avg_mem_mb: float
    start_mem_mb: float
    end_mem_mb: float
    delta_mem_mb: float

    def asdict(self) -> dict[str, str | float]:
        return {
            "func_name": self.func_name,
            "runtime_s": round(self.runtime_s, 6),
            "samples": self.samples,
            "approx_interval_s": round(self.approx_interval_s, 6),
            "peak_mem_mb": round(self.peak_mem_mb, 3),
            "avg_mem_mb": round(self.avg_mem_mb, 3),
            "start_mem_mb": round(self.start_mem_mb, 3),
            "end_mem_mb": round(self.end_mem_mb, 3),
            "delta_mem_mb": round(self.delta_mem_mb, 3)
        }

    def to_json(self, filepath: str) -> str:
        dic_stats = self.asdict()
        json_stats = json.dumps(dic_stats)
        with open(filepath, "w") as jsonfile:
            jsonfile.write(json_stats)
        return filepath



def benchmarking_analysis(
    func: Callable[..., Any],
    *args: Any,
    interval: float = 0.05,
    include_children: bool = True,
    output_path: str | None = None,
    **kwargs: Any,
) -> dict[str, str | float]:
    """
    Analiza memoria y tiempo de ejecución de `func` usando memory-profiler y devuelve métricas.
    Opcionalmente guarda la serie temporal en CSV y una gráfica PNG.

    Parámetros
    ----------
    func : Callable
        Función a evaluar.
    *args, **kwargs : Any
        Argumentos de la función.
    interval : float, por defecto 0.05
        Periodo de muestreo en segundos para memory-profiler.
    include_children : bool, por defecto True
        Incluye procesos hijos en la medición.
    save_csv : str | None
        Ruta donde guardar el CSV con columnas: t_s, mem_mb.
    save_plot : str | None
        Ruta donde guardar la gráfica PNG (memoria vs tiempo).

    Retorna
    -------
    dict[str, str | float]
        Diccionario con estadísticas clave y, si aplica, rutas a CSV/PNG.
    """
    start_perf = time.perf_counter()
    (series, _retval) = memory_usage(
        (func, args, kwargs),
        interval=interval,
        timeout=None,
        timestamps=True,
        retval=True,
        include_children=include_children,
        max_iterations=None,
        multiprocess=True,
        max_usage=False,
        stream=None,
    )
    end_perf = time.perf_counter()

    if not series:
        raise RuntimeError("No se obtuvo ninguna muestra de memoria.")

    t0 = series[0][0]
    times_s = [t - t0 for (t, _m) in series]
    mems_mb = [m for (_t, m) in series]

    samples = len(mems_mb)
    approx_interval_s = (times_s[-1] / (samples - 1)) if samples > 1 else interval
    peak_mem_mb = max(mems_mb)
    avg_mem_mb = sum(mems_mb) / samples
    start_mem_mb = mems_mb[0]
    end_mem_mb = mems_mb[-1]
    delta_mem_mb = peak_mem_mb - start_mem_mb

    func_name = getattr(func, "__name__", "callable")
    path = Path(os.path.join(output_path,f"{func_name}_timeserie.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("t_s,mem_mb\n")
        for t, m in zip(times_s, mems_mb):
            file.write(f"{t:.6f},{m:.6f}\n")

    path = Path(os.path.join(output_path,f"{func_name}_timeserie.png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(times_s, mems_mb)
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MiB)")
    plt.title(f"Memory Use: {func.__name__}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    stats = BenchStats(
        func_name=getattr(func, "__name__", "callable"),
        runtime_s=end_perf - start_perf,
        samples=samples,
        approx_interval_s=approx_interval_s,
        peak_mem_mb=peak_mem_mb,
        avg_mem_mb=avg_mem_mb,
        start_mem_mb=start_mem_mb,
        end_mem_mb=end_mem_mb,
        delta_mem_mb=delta_mem_mb
    )

    stats.to_json(os.path.join(output_path,f"{func_name}_stats.json"))
    return stats.asdict()

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])
    benchmarking_analysis(main, output_path=args.path)
