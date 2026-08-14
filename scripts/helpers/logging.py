"""CSV + stderr logger for experiments.

Usage:
    log = ExperimentLogger("my_experiment")
    log.log_epoch(epoch=1, train_loss=0.5, std_acc=0.8)
    log.log_final({"std_acc": 0.85, "cert_acc": 0.35})
"""

import csv
import os
import sys
import time
from dataclasses import dataclass, field


@dataclass
class ExperimentLogger:
    name: str
    output_dir: str = "logs"
    extra_meta: dict = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        safe_name = self.name.replace(" ", "_").replace("/", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{safe_name}"
        self.csv_path = os.path.join(self.output_dir, f"{self.run_id}.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = None
        self._header_written = False
        self.start_time = time.time()
        self._print_meta()

    def _print_meta(self):
        self._console("=" * 60)
        self._console(f"  experiment: {self.name}")
        self._console(f"  run id:     {self.run_id}")
        self._console(f"  log:        {self.csv_path}")
        if self.extra_meta:
            for k, v in self.extra_meta.items():
                self._console(f"  {k}: {v}")
        self._console("=" * 60)

    def _console(self, msg: str):
        print(f"[{self.run_id}] {msg}", file=sys.stderr)

    def log_epoch(self, **kwargs):
        row = {"epoch": kwargs.get("epoch", "?"), "phase": "train"}
        row.update({k: v for k, v in kwargs.items() if k != "epoch"})
        self._write_row(row)

    def log_eval(self, **kwargs):
        row = {"epoch": kwargs.get("epoch", "?"), "phase": "eval"}
        row.update({k: v for k, v in kwargs.items() if k != "epoch"})
        self._write_row(row)

    def log_final(self, metrics: dict):
        elapsed = time.time() - self.start_time
        row = {"epoch": "final", "phase": "final"}
        row.update(metrics)
        row["elapsed_seconds"] = round(elapsed, 1)
        self._write_row(row)
        self._print_summary(metrics, elapsed)

    def _write_row(self, row: dict):
        if not self._header_written:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            self._csv_writer.writeheader()
            self._header_written = True
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _print_summary(self, metrics: dict, elapsed: float):
        self._console("-" * 60)
        self._console("  final results")
        for k, v in metrics.items():
            if isinstance(v, float):
                self._console(f"    {k}: {v:.4f}")
            else:
                self._console(f"    {k}: {v}")
        self._console(f"  elapsed: {elapsed:.1f}s")
        self._console("-" * 60)

    def close(self):
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

    def __del__(self):
        self.close()
