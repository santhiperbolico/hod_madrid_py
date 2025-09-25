import sys
import argparse
import logging
import os
import json
from typing import Any, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    interpreter_mem_mib: float
    peak_mem_mib: float
    avg_mem_mib: float
    start_mem_mib: float
    end_mem_mib: float
    delta_mem_mib: float

    def asdict(self) -> dict[str, str | float]:
        return {
            "func_name": self.func_name,
            "runtime_s": round(self.runtime_s, 6),
            "samples": self.samples,
            "approx_interval_s": round(self.approx_interval_s, 6),
            "interpreter_mem_mib": self.interpreter_mem_mib,
            "peak_mem_mib": round(self.peak_mem_mib, 3),
            "avg_mem_mib": round(self.avg_mem_mib, 3),
            "start_mem_mib": round(self.start_mem_mib, 3),
            "end_mem_mib": round(self.end_mem_mib, 3),
            "delta_mem_mib": round(self.delta_mem_mib, 3)
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
    interpreter_mem_mib = memory_usage(max_usage=True)
    series, _ = memory_usage(
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

    times_s = list(np.linspace(0, end_perf-start_perf, len(series)))
    mems_mb = [m for (m, _s) in series]

    samples = len(mems_mb)
    approx_interval_s = (times_s[-1] / (samples - 1)) if samples > 1 else interval
    peak_mem_mib = max(mems_mb)
    avg_mem_mib = sum(mems_mb) / samples
    start_mem_mib = mems_mb[0]
    end_mem_mib = mems_mb[-1]
    delta_mem_mib = peak_mem_mib - interpreter_mem_mib

    func_name = getattr(func, "__name__", "callable")
    path = Path(os.path.join(output_path,f"{func_name}_timeserie.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("t_s,mem_mib, interpreter_mem_mib\n")
        for t, m in zip(times_s, mems_mb):
            file.write(f"{t:.6f},{m:.6f},{interpreter_mem_mib:.6f}\n")

    path = Path(os.path.join(output_path,f"{func_name}_timeserie.png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    mem_array = np.array(mems_mb) - interpreter_mem_mib
    plt.plot(times_s, mem_array)
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MiB) - Interpreter Memory (MiB)")
    plt.title(f"Process Memory Used: {func.__name__}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    stats = BenchStats(
        func_name=getattr(func, "__name__", "callable"),
        runtime_s=end_perf - start_perf,
        samples=samples,
        approx_interval_s=float(approx_interval_s),
        interpreter_mem_mib=interpreter_mem_mib,
        peak_mem_mib=peak_mem_mib,
        avg_mem_mib=avg_mem_mib,
        start_mem_mib=start_mem_mib,
        end_mem_mib=end_mem_mib,
        delta_mem_mib=delta_mem_mib
    )

    stats.to_json(os.path.join(output_path,f"{func_name}_stats.json"))
    return stats.asdict()

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    args = get_params(sys.argv[1:])
    benchmarking_analysis(main, output_path=args.path)
