"""Compatibility entrypoint. Use python -m src.benchmark_latency."""

from src.benchmark_latency import run_latency_benchmark, main

__all__ = ["run_latency_benchmark", "main"]
if __name__ == "__main__":
    main()
