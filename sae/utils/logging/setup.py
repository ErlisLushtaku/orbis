"""Logging configuration and formatting utilities for the SAE module.

Provides standardized logging setup matching Orbis conventions
and human-readable formatting for durations, throughput, and byte sizes.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_logging_initialized = False


def setup_sae_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """Configure logging for SAE modules.

    Args:
        log_file: Optional path to write logs to file
        level: Logging level (default: INFO)
        console: Whether to output to console (default: True)
    """
    global _logging_initialized

    if _logging_initialized:
        return

    sae_logger = logging.getLogger("sae")
    sae_logger.setLevel(level)
    sae_logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        sae_logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        sae_logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    sae_logger.propagate = False

    _logging_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for SAE modules with hierarchical naming.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    if not _logging_initialized:
        setup_sae_logging()

    # Convert module path to hierarchical logger name under "sae"
    if name.startswith("sae."):
        logger_name = name
    elif "sae" in name:
        parts = name.split(".")
        try:
            sae_idx = parts.index("sae")
            logger_name = ".".join(parts[sae_idx:])
        except ValueError:
            logger_name = f"sae.{name}"
    else:
        logger_name = f"sae.{name}"

    return logging.getLogger(logger_name)



def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        if minutes > 0:
            return f"{hours:.0f}h {minutes:.0f}m"
        return f"{hours:.1f}h"


def format_throughput(value: float, unit: str) -> str:
    """Format throughput with appropriate unit prefix."""
    if value >= 1e9:
        return f"{value / 1e9:.2f}G {unit}"
    elif value >= 1e6:
        return f"{value / 1e6:.2f}M {unit}"
    elif value >= 1e3:
        return f"{value / 1e3:.2f}K {unit}"
    else:
        return f"{value:.2f} {unit}"


def format_bytes(num_bytes: float) -> str:
    """Format bytes in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"
