"""GPU and RAM monitoring for the SAE module.

Provides a background sampling thread that tracks GPU utilization,
GPU memory, job-level RAM (main process + workers), and system-wide RAM.
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional

import psutil
import torch

from .setup import get_logger

logger = get_logger(__name__)


class GPUMonitor:
    """Background thread for monitoring GPU utilization and memory.

    Low GPU utilization during training indicates an IO bottleneck.

    Usage:
        monitor = GPUMonitor(sample_interval_s=1.0)
        monitor.start()
        # ... training ...
        stats = monitor.get_stats()
        monitor.stop()
    """

    def __init__(
        self,
        sample_interval_s: float = 1.0,
        device_id: int = 0,
    ):
        self.sample_interval_s = sample_interval_s
        self.device_id = device_id
        self.logger = get_logger("sae.gpu_monitor")

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # GPU samples
        self._util_samples: List[float] = []
        self._memory_samples: List[float] = []
        self._timestamps: List[float] = []

        # RAM samples
        self._process = psutil.Process(os.getpid())
        self._ram_rss_samples: List[float] = []        # Main process RSS in bytes
        self._ram_job_samples: List[float] = []        # Main + children RSS in bytes
        self._ram_system_samples: List[float] = []     # System used RAM in bytes
        self._ram_system_total: float = psutil.virtual_memory().total

        # Phase tracking
        self._current_phase: Optional[str] = None
        self._phase_samples: Dict[str, Dict[str, List[float]]] = {}

        # Check for pynvml availability
        self._use_pynvml = False
        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self._use_pynvml = True
            self.logger.debug("Using pynvml for GPU monitoring")
        except Exception:
            self.logger.warning(
                "pynvml not available -- GPU utilization will report 0%%. "
                "Install nvidia-ml-py for accurate monitoring."
            )

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self.logger.debug("GPU monitor started")

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.logger.debug("GPU monitor stopped")

    def set_phase(self, phase: str) -> None:
        """Set current phase for per-phase statistics."""
        with self._lock:
            self._current_phase = phase
            if phase not in self._phase_samples:
                self._phase_samples[phase] = {"util": [], "memory": []}

    def clear_phase(self) -> None:
        """Clear current phase."""
        with self._lock:
            self._current_phase = None

    def _get_job_memory(self) -> tuple:
        """Get memory for the main process and the full process tree.

        Returns:
            Tuple of (main_rss_bytes, job_total_rss_bytes).
        """
        try:
            main_rss = self._process.memory_info().rss
            job_rss = main_rss
            for child in self._process.children(recursive=True):
                try:
                    job_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return main_rss, job_rss
        except Exception:
            return 0, 0

    def _sample_loop(self) -> None:
        """Background sampling loop for GPU and RAM metrics."""
        while self._running:
            try:
                util, memory = self._get_gpu_stats()

                # RAM metrics (main process + DataLoader workers + system)
                main_rss, job_rss = self._get_job_memory()
                sys_mem = psutil.virtual_memory()
                sys_used = sys_mem.total - sys_mem.available

                timestamp = time.time()

                with self._lock:
                    self._util_samples.append(util)
                    self._memory_samples.append(memory)
                    self._timestamps.append(timestamp)
                    self._ram_rss_samples.append(main_rss)
                    self._ram_job_samples.append(job_rss)
                    self._ram_system_samples.append(sys_used)

                    if self._current_phase is not None:
                        self._phase_samples[self._current_phase]["util"].append(util)
                        self._phase_samples[self._current_phase]["memory"].append(memory)

            except Exception as e:
                self.logger.debug(f"Sampling error: {e}")

            time.sleep(self.sample_interval_s)

    def _get_gpu_stats(self) -> tuple:
        """Get current GPU utilization and memory usage."""
        if self._use_pynvml and self._nvml_handle is not None:
            import pynvml
            util_info = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return util_info.gpu, mem_info.used
        else:
            if torch.cuda.is_available():
                memory = torch.cuda.memory_allocated(self.device_id)
                return 0.0, memory
            return 0.0, 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics including GPU and RAM."""
        with self._lock:
            if not self._util_samples:
                return {}

            result: Dict[str, Any] = {
                "avg_util_pct": sum(self._util_samples) / len(self._util_samples),
                "max_util_pct": max(self._util_samples),
                "min_util_pct": min(self._util_samples),
                "peak_memory_gb": max(self._memory_samples) / 1e9,
                "avg_memory_gb": sum(self._memory_samples) / len(self._memory_samples) / 1e9,
                "num_samples": len(self._util_samples),
            }

            if self._ram_job_samples:
                result["job_ram_peak_gb"] = max(self._ram_job_samples) / 1e9
                result["main_process_ram_peak_gb"] = max(self._ram_rss_samples) / 1e9
                result["system_ram_peak_gb"] = max(self._ram_system_samples) / 1e9
                result["system_ram_total_gb"] = self._ram_system_total / 1e9

            return result

    def get_phase_stats(self, phase: str) -> Dict[str, Any]:
        """Get statistics for a specific phase."""
        with self._lock:
            if phase not in self._phase_samples:
                return {}

            samples = self._phase_samples[phase]
            if not samples["util"]:
                return {}

            return {
                "avg_util_pct": sum(samples["util"]) / len(samples["util"]),
                "peak_memory_gb": max(samples["memory"]) / 1e9 if samples["memory"] else 0,
            }
