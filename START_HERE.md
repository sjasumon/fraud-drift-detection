# Start here — reaudited v0.4.1

1. Read [TASK_COMPLETION_AUDIT.md](TASK_COMPLETION_AUDIT.md) for the defects found,
   fixes, evidence and genuinely pending work.
2. Read [VALIDATION.md](VALIDATION.md): 122 tests passed, plus installed-command checks.
3. On Windows with Python 3.12, run `setup.bat` or `setup.ps1`.
4. Run `.\.venv\Scripts\python.exe -m src.demo` for a synthetic example.
5. Follow README.md to supply the real datasets and reproduce the corrected experiments.

Included reports (extract the ZIP before opening):
- [Synthetic example report](results/example/runs/RUN_12C56BA057/report.html)
- [Synthetic study summary](results/study_example/study.csv)

Source installation is recommended for development. An installable, tested wheel
is also included in `wheel/`. Synthetic packages remain unready for serving.
The original research documents/results are preserved as historical archives.
