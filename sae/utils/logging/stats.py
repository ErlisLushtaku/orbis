"""Training statistics collection for the SAE module.

Provides phase timing, epoch-level IO tracking, and aggregated statistics
with JSON export for experiment analysis and storage strategy comparison.
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .setup import get_logger, format_duration, format_throughput, format_bytes



@dataclass
class PhaseRecord:
    """Record of a single phase's timing and metrics."""
    name: str
    duration_s: float
    start_time: float
    end_time: float
    tokens: int = 0
    bytes_processed: int = 0
    batches: int = 0
    samples: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def throughput_tokens_s(self) -> Optional[float]:
        """Tokens per second throughput."""
        if self.tokens > 0 and self.duration_s > 0:
            return self.tokens / self.duration_s
        return None

    @property
    def throughput_mb_s(self) -> Optional[float]:
        """MB per second throughput."""
        if self.bytes_processed > 0 and self.duration_s > 0:
            return (self.bytes_processed / 1e6) / self.duration_s
        return None

    @property
    def throughput_samples_s(self) -> Optional[float]:
        """Samples per second throughput."""
        if self.samples > 0 and self.duration_s > 0:
            return self.samples / self.duration_s
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        result = {
            "name": self.name,
            "duration_s": round(self.duration_s, 3),
        }
        if self.tokens > 0:
            result["tokens"] = self.tokens
            if self.throughput_tokens_s:
                result["throughput_tok_s"] = round(self.throughput_tokens_s, 1)
        if self.bytes_processed > 0:
            result["bytes"] = self.bytes_processed
            if self.throughput_mb_s:
                result["throughput_mb_s"] = round(self.throughput_mb_s, 1)
        if self.samples > 0:
            result["samples"] = self.samples
            if self.throughput_samples_s:
                result["throughput_samples_s"] = round(self.throughput_samples_s, 1)
        if self.batches > 0:
            result["batches"] = self.batches
        if self.extra:
            result.update(self.extra)
        return result


class PhaseTimer:
    """Context manager for timing code phases with throughput tracking.

    Usage:
        with PhaseTimer("caching_train", stats) as timer:
            for batch in dataloader:
                # ... process ...
                timer.update(tokens=1000, bytes_processed=4096)
    """

    def __init__(
        self,
        name: str,
        stats: "TimingStats",
        log_start: bool = True,
        log_end: bool = True,
    ):
        self.name = name
        self.stats = stats
        self.log_start = log_start
        self.log_end = log_end
        self.logger = get_logger("sae.timing")

        self.start_time: float = 0
        self.end_time: float = 0
        self.tokens: int = 0
        self.bytes_processed: int = 0
        self.batches: int = 0
        self.samples: int = 0
        self.extra: Dict[str, Any] = {}

    def __enter__(self) -> "PhaseTimer":
        self.start_time = time.perf_counter()
        if self.log_start:
            self.logger.info(f"Starting phase: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time

        record = PhaseRecord(
            name=self.name,
            duration_s=duration,
            start_time=self.start_time,
            end_time=self.end_time,
            tokens=self.tokens,
            bytes_processed=self.bytes_processed,
            batches=self.batches,
            samples=self.samples,
            extra=self.extra,
        )

        self.stats.record_phase(record)

        if self.log_end:
            msg = f"Completed phase: {self.name} in {format_duration(duration)}"
            if record.throughput_tokens_s:
                msg += f" ({format_throughput(record.throughput_tokens_s, 'tok/s')})"
            if record.throughput_mb_s:
                msg += f" ({record.throughput_mb_s:.1f} MB/s)"
            self.logger.info(msg)

    def update(
        self,
        tokens: int = 0,
        bytes_processed: int = 0,
        batches: int = 0,
        samples: int = 0,
        **extra,
    ) -> None:
        """Update counters for throughput calculation."""
        self.tokens += tokens
        self.bytes_processed += bytes_processed
        self.batches += batches
        self.samples += samples
        for k, v in extra.items():
            if k in self.extra:
                self.extra[k] += v
            else:
                self.extra[k] = v

    @property
    def elapsed(self) -> float:
        """Current elapsed time in seconds."""
        return time.perf_counter() - self.start_time



@dataclass
class EpochStats:
    """Statistics for a single training epoch, tracking IO wait vs compute time.

    Critical for storage strategy comparison: measures how much time is spent
    waiting for data vs actual GPU compute.
    """
    epoch_num: int

    # Accumulated timing
    io_wait_time: float = 0.0
    compute_time: float = 0.0
    total_time: float = 0.0

    # Counters
    num_batches: int = 0
    num_samples: int = 0
    bytes_read: int = 0

    # Metrics (NaN-safe: only finite losses are accumulated)
    loss_sum: float = 0.0
    loss_count: int = 0

    # GPU stats (filled by GPUMonitor)
    gpu_util_samples: List[float] = field(default_factory=list)
    gpu_memory_samples: List[float] = field(default_factory=list)

    def add_batch(
        self,
        io_wait: float,
        compute_time: float,
        batch_size: int = 0,
        bytes_read: int = 0,
        loss: float = 0.0,
    ) -> None:
        """Record metrics for a single batch."""
        self.io_wait_time += io_wait
        self.compute_time += compute_time
        self.total_time += io_wait + compute_time
        self.num_batches += 1
        self.num_samples += batch_size
        self.bytes_read += bytes_read
        if math.isfinite(loss):
            self.loss_sum += loss
            self.loss_count += 1

    @property
    def efficiency_index(self) -> float:
        """Compute/wall time ratio. 1.0 = no IO overhead."""
        if self.total_time > 0:
            return self.compute_time / self.total_time
        return 1.0

    @property
    def io_wait_pct(self) -> float:
        """Percentage of time spent waiting for IO."""
        if self.total_time > 0:
            return (self.io_wait_time / self.total_time) * 100
        return 0.0

    @property
    def avg_gpu_util(self) -> Optional[float]:
        """Average GPU utilization percentage."""
        if self.gpu_util_samples:
            return sum(self.gpu_util_samples) / len(self.gpu_util_samples)
        return None

    @property
    def peak_gpu_memory_gb(self) -> Optional[float]:
        """Peak GPU memory in GB."""
        if self.gpu_memory_samples:
            return max(self.gpu_memory_samples) / 1e9
        return None

    @property
    def disk_read_mb_s(self) -> Optional[float]:
        """Disk read throughput in MB/s (based on IO wait time)."""
        if self.bytes_read > 0 and self.io_wait_time > 0:
            return (self.bytes_read / 1e6) / self.io_wait_time
        return None

    @property
    def avg_loss(self) -> Optional[float]:
        """Average loss for the epoch (NaN batches excluded)."""
        if self.loss_count > 0:
            return self.loss_sum / self.loss_count
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        result = {
            "epoch": self.epoch_num,
            "duration_s": round(self.total_time, 3),
            "compute_time_s": round(self.compute_time, 3),
            "io_wait_time_s": round(self.io_wait_time, 3),
            "io_wait_pct": round(self.io_wait_pct, 2),
            "efficiency_index": round(self.efficiency_index, 3),
            "num_batches": self.num_batches,
            "num_samples": self.num_samples,
        }
        if self.bytes_read > 0:
            result["bytes_read"] = self.bytes_read
            if self.disk_read_mb_s:
                result["disk_read_mb_s"] = round(self.disk_read_mb_s, 1)
        if self.avg_gpu_util is not None:
            result["gpu_util_pct"] = round(self.avg_gpu_util, 1)
        if self.peak_gpu_memory_gb is not None:
            result["peak_gpu_memory_gb"] = round(self.peak_gpu_memory_gb, 2)
        if self.avg_loss is not None:
            result["avg_loss"] = round(self.avg_loss, 6)
        return result



class TimingStats:
    """Centralized statistics collection for SAE training runs.

    Collects phase timing, per-epoch IO tracking, GPU utilization,
    and data source information for storage comparison.
    """

    def __init__(self):
        self.logger = get_logger("sae.stats")

        # Run metadata
        self.run_name: Optional[str] = None
        self.layer: Optional[int] = None
        self.data_source: Optional[str] = None

        # Data source tracking
        self.data_source_type: str = "unknown"
        self.data_source_path: Optional[str] = None
        self.original_data_path: Optional[str] = None

        # Phase records
        self.phases: Dict[str, PhaseRecord] = {}
        self._phase_order: List[str] = []

        # Epoch records
        self.epoch_stats: List[EpochStats] = []

        # Cache, training, GPU, RAM statistics
        self.cache_stats: Dict[str, Any] = {}
        self.training_stats: Dict[str, Any] = {}
        self.gpu_stats: Dict[str, Any] = {}
        self.ram_stats: Dict[str, Any] = {}

        # Timing
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def set_run_info(self, run_name: str, layer: int, data_source: str) -> None:
        """Set run metadata."""
        self.run_name = run_name
        self.layer = layer
        self.data_source = data_source
        self._start_time = time.time()

    def set_data_source(
        self, source_type: str, path: str, original_path: Optional[str] = None,
    ) -> None:
        """Set data source information for storage strategy comparison."""
        self.data_source_type = source_type
        self.data_source_path = path
        self.original_data_path = original_path
        self.logger.info(f"Data source: {source_type} ({path})")

    def record_phase(self, record: PhaseRecord) -> None:
        """Record a completed phase."""
        self.phases[record.name] = record
        if record.name not in self._phase_order:
            self._phase_order.append(record.name)

    def record_epoch(self, epoch_stats: EpochStats) -> None:
        """Record completed epoch statistics."""
        self.epoch_stats.append(epoch_stats)
        self.logger.info(
            f"Epoch {epoch_stats.epoch_num}: "
            f"{format_duration(epoch_stats.total_time)} "
            f"(IO wait: {epoch_stats.io_wait_pct:.1f}%, "
            f"efficiency: {epoch_stats.efficiency_index:.2f})"
        )

    def set_cache_stats(
        self,
        train_batches: int,
        train_tokens: int,
        train_bytes: int,
        val_batches: int,
        val_tokens: int,
        val_bytes: int,
        dtype: str,
    ) -> None:
        """Set cache statistics."""
        self.cache_stats = {
            "train_batches": train_batches,
            "train_tokens": train_tokens,
            "train_bytes": train_bytes,
            "val_batches": val_batches,
            "val_tokens": val_tokens,
            "val_bytes": val_bytes,
            "dtype": dtype,
        }

    def set_training_stats(
        self, num_epochs: int, batches_per_epoch: int, final_loss: float,
    ) -> None:
        """Set training statistics."""
        self.training_stats = {
            "num_epochs": num_epochs,
            "batches_per_epoch": batches_per_epoch,
            "final_loss": final_loss,
        }

    def set_gpu_stats(self, gpu_stats: Dict[str, Any]) -> None:
        """Set GPU statistics from GPUMonitor."""
        self.gpu_stats = gpu_stats

    def set_ram_stats(self, ram_stats: Dict[str, Any]) -> None:
        """Set RAM statistics from GPUMonitor."""
        self.ram_stats = ram_stats

    def finalize(self) -> None:
        """Finalize statistics collection."""
        self._end_time = time.time()

    @property
    def total_duration_s(self) -> float:
        """Total wall-clock time."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.time()
        return end - self._start_time

    def _get_io_efficiency_stats(self) -> Dict[str, Any]:
        """Calculate IO efficiency statistics with cold/hot separation."""
        if not self.epoch_stats:
            return {}

        cold_epochs = [e for e in self.epoch_stats if e.epoch_num == 1]
        hot_epochs = [e for e in self.epoch_stats if e.epoch_num > 1]

        result: Dict[str, Any] = {}

        if cold_epochs:
            cold = cold_epochs[0]
            cold_dict: Dict[str, Any] = {
                "duration_s": round(cold.total_time, 3),
                "compute_time_s": round(cold.compute_time, 3),
                "io_wait_s": round(cold.io_wait_time, 3),
                "io_wait_pct": round(cold.io_wait_pct, 2),
                "efficiency_index": round(cold.efficiency_index, 3),
            }
            if cold.avg_gpu_util is not None:
                cold_dict["gpu_util_pct"] = round(cold.avg_gpu_util, 1)
            if cold.disk_read_mb_s is not None:
                cold_dict["disk_read_mb_s"] = round(cold.disk_read_mb_s, 1)
            result["epoch_1_cold"] = cold_dict

        if hot_epochs:
            avg_duration = sum(e.total_time for e in hot_epochs) / len(hot_epochs)
            avg_compute = sum(e.compute_time for e in hot_epochs) / len(hot_epochs)
            avg_io_wait = sum(e.io_wait_time for e in hot_epochs) / len(hot_epochs)
            avg_efficiency = sum(e.efficiency_index for e in hot_epochs) / len(hot_epochs)

            hot_dict: Dict[str, Any] = {
                "avg_duration_s": round(avg_duration, 3),
                "avg_compute_time_s": round(avg_compute, 3),
                "avg_io_wait_s": round(avg_io_wait, 3),
                "io_wait_pct": round((avg_io_wait / avg_duration * 100) if avg_duration > 0 else 0, 2),
                "efficiency_index": round(avg_efficiency, 3),
                "num_epochs": len(hot_epochs),
            }

            gpu_utils = [e.avg_gpu_util for e in hot_epochs if e.avg_gpu_util is not None]
            if gpu_utils:
                hot_dict["gpu_util_pct"] = round(sum(gpu_utils) / len(gpu_utils), 1)

            disk_reads = [e.disk_read_mb_s for e in hot_epochs if e.disk_read_mb_s is not None]
            if disk_reads:
                hot_dict["disk_read_mb_s"] = round(sum(disk_reads) / len(disk_reads), 1)

            result["epochs_2_to_n_hot"] = hot_dict

        if self.epoch_stats:
            total_duration = sum(e.total_time for e in self.epoch_stats)
            total_compute = sum(e.compute_time for e in self.epoch_stats)
            total_io_wait = sum(e.io_wait_time for e in self.epoch_stats)

            result["overall"] = {
                "avg_duration_s": round(total_duration / len(self.epoch_stats), 3),
                "total_compute_s": round(total_compute, 3),
                "total_io_wait_s": round(total_io_wait, 3),
                "io_wait_pct": round((total_io_wait / total_duration * 100) if total_duration > 0 else 0, 2),
                "efficiency_index": round((total_compute / total_duration) if total_duration > 0 else 1.0, 3),
            }

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert all statistics to dictionary for JSON export."""
        result: Dict[str, Any] = {
            "run_name": self.run_name,
            "layer": self.layer,
            "data_source": self.data_source,
            "data_source_type": self.data_source_type,
            "data_source_path": self.data_source_path,
        }

        if self.original_data_path:
            result["original_data_path"] = self.original_data_path

        result["phases"] = {
            name: self.phases[name].to_dict()
            for name in self._phase_order
            if name in self.phases
        }

        io_efficiency = self._get_io_efficiency_stats()
        if io_efficiency:
            result["io_efficiency"] = io_efficiency

        if self.epoch_stats:
            result["epochs"] = [e.to_dict() for e in self.epoch_stats]

        if self.cache_stats:
            result["cache_stats"] = self.cache_stats
        if self.training_stats:
            result["training_stats"] = self.training_stats
        if self.gpu_stats:
            result["gpu_stats"] = self.gpu_stats
        if self.ram_stats:
            result["ram_stats"] = self.ram_stats

        result["total_duration_s"] = round(self.total_duration_s, 3)
        return result

    def save_json(self, path: Path) -> None:
        """Save statistics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        self.logger.info(f"Statistics saved to {path}")

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        self.finalize()

        width = 80
        sep = "=" * width
        sep_thin = "-" * width

        lines = [
            sep,
            "SAE TRAINING STATISTICS SUMMARY".center(width),
            sep,
        ]

        # Run info
        info_parts = []
        if self.run_name:
            info_parts.append(f"Run: {self.run_name}")
        if self.layer is not None:
            info_parts.append(f"Layer: {self.layer}")
        if self.data_source:
            info_parts.append(f"Data: {self.data_source}")
        if info_parts:
            lines.append(" | ".join(info_parts))

        if self.data_source_type != "unknown":
            lines.append(f"Data Source: {self.data_source_type} ({self.data_source_path})")

        lines.append("")

        # Timing breakdown
        lines.append("TIMING BREAKDOWN:")
        lines.append(sep_thin)
        lines.append(f"{'Phase':<35} {'Duration':>12} {'%Total':>10} {'Throughput':>18}")
        lines.append(sep_thin)

        total_duration = self.total_duration_s
        for name in self._phase_order:
            if name not in self.phases:
                continue
            record = self.phases[name]

            pct = (record.duration_s / total_duration * 100) if total_duration > 0 else 0
            pct_str = f"{pct:.1f}%" if pct >= 0.1 else "<0.1%"

            throughput_parts = []
            if record.throughput_tokens_s:
                throughput_parts.append(format_throughput(record.throughput_tokens_s, "tok/s"))
            if record.throughput_mb_s:
                throughput_parts.append(f"{record.throughput_mb_s:.1f} MB/s")
            if record.throughput_samples_s:
                throughput_parts.append(format_throughput(record.throughput_samples_s, "samp/s"))
            throughput_str = " | ".join(throughput_parts) if throughput_parts else "-"

            lines.append(f"{name:<35} {format_duration(record.duration_s):>12} {pct_str:>10} {throughput_str:>18}")

        lines.append(sep_thin)
        lines.append(f"{'Total Wall-Clock Time:':<35} {format_duration(total_duration):>12}")
        lines.append("")

        # IO Efficiency
        io_stats = self._get_io_efficiency_stats()
        if io_stats:
            lines.append("IO EFFICIENCY (Key for Storage Comparison):")
            lines.append(sep_thin)

            header = f"{'':>20} {'Epoch 1 (Cold)':>18} {'Epochs 2-N (Hot)':>18} {'Average':>12}"
            lines.append(header)
            lines.append(sep_thin)

            cold = io_stats.get("epoch_1_cold", {})
            hot = io_stats.get("epochs_2_to_n_hot", {})
            overall = io_stats.get("overall", {})

            def fmt_row(label, cold_key, hot_key, overall_key, fmt=".1f", suffix=""):
                cold_val = cold.get(cold_key, "-")
                hot_val = hot.get(hot_key, "-")
                overall_val = overall.get(overall_key, "-")

                def fmt_val(v):
                    if v == "-":
                        return "-"
                    return f"{v:{fmt}}{suffix}"

                return f"{label:>20} {fmt_val(cold_val):>18} {fmt_val(hot_val):>18} {fmt_val(overall_val):>12}"

            lines.append(fmt_row("Epoch Duration", "duration_s", "avg_duration_s", "avg_duration_s", ".1f", "s"))
            lines.append(fmt_row("  - Compute Time", "compute_time_s", "avg_compute_time_s", "total_compute_s", ".1f", "s"))
            lines.append(fmt_row("  - IO Wait Time", "io_wait_s", "avg_io_wait_s", "total_io_wait_s", ".1f", "s"))
            lines.append(fmt_row("IO Wait %", "io_wait_pct", "io_wait_pct", "io_wait_pct", ".1f", "%"))

            if "disk_read_mb_s" in cold or "disk_read_mb_s" in hot:
                lines.append(fmt_row("Disk Read Speed", "disk_read_mb_s", "disk_read_mb_s", "", ".0f", " MB/s"))

            if "gpu_util_pct" in cold or "gpu_util_pct" in hot:
                lines.append(fmt_row("GPU Utilization", "gpu_util_pct", "gpu_util_pct", "", ".0f", "%"))

            lines.append(fmt_row("Efficiency Index", "efficiency_index", "efficiency_index", "efficiency_index", ".2f", ""))
            lines.append(sep_thin)
            lines.append("")

        # Cache statistics
        if self.cache_stats:
            lines.append("CACHE STATISTICS:")
            lines.append(sep_thin)
            cs = self.cache_stats
            lines.append(
                f"Train cache: {cs.get('train_batches', 0):,} batches | "
                f"{format_throughput(cs.get('train_tokens', 0), 'tokens')} | "
                f"{format_bytes(cs.get('train_bytes', 0))}"
            )
            lines.append(
                f"Val cache:   {cs.get('val_batches', 0):,} batches | "
                f"{format_throughput(cs.get('val_tokens', 0), 'tokens')} | "
                f"{format_bytes(cs.get('val_bytes', 0))}"
            )
            lines.append(f"Cache dtype: {cs.get('dtype', 'unknown')}")
            lines.append("")

        # Training statistics
        if self.training_stats:
            lines.append("TRAINING STATISTICS:")
            lines.append(sep_thin)
            ts = self.training_stats
            lines.append(
                f"Epochs: {ts.get('num_epochs', 0)} | "
                f"Batches/epoch: {ts.get('batches_per_epoch', 0):,} | "
                f"Final loss: {ts.get('final_loss', 0):.4f}"
            )

            if self.epoch_stats:
                avg_epoch_time = sum(e.total_time for e in self.epoch_stats) / len(self.epoch_stats)
                total_batches = sum(e.num_batches for e in self.epoch_stats)
                total_time = sum(e.total_time for e in self.epoch_stats)
                avg_batch_time = (total_time / total_batches * 1000) if total_batches > 0 else 0
                lines.append(
                    f"Avg epoch time: {format_duration(avg_epoch_time)} | "
                    f"Avg batch time: {avg_batch_time:.2f}ms"
                )
            lines.append("")

        # GPU statistics
        if self.gpu_stats:
            gpu = self.gpu_stats
            lines.append(
                f"Peak GPU Memory: {gpu.get('peak_memory_gb', 0):.1f} GB | "
                f"Avg GPU Utilization: {gpu.get('avg_util_pct', 0):.0f}%"
            )

        # RAM statistics
        if self.ram_stats:
            ram = self.ram_stats
            sys_total = ram.get("system_ram_total_gb", 0)
            sys_peak = ram.get("system_ram_peak_gb", 0)
            sys_pct = (sys_peak / sys_total * 100) if sys_total > 0 else 0
            lines.append(
                f"Job RAM (peak): {ram.get('job_ram_peak_gb', 0):.1f} GB | "
                f"System RAM: {sys_peak:.1f} / {sys_total:.1f} GB ({sys_pct:.1f}%)"
            )

        lines.append(sep)
        print("\n".join(lines))



stats = TimingStats()
