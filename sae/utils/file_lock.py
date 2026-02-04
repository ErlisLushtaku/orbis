"""
Atomic file locking utilities for safe concurrent JSON operations.

This module provides a context manager for atomic JSON file operations with
file locking to prevent race conditions when multiple processes access the
same file (e.g., partition_map.json or calibration resource files).
"""

import fcntl
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Tuple

logger = logging.getLogger(__name__)


@contextmanager
def atomic_json_file(
    path: Path,
    create_if_missing: bool = True,
) -> Generator[Tuple[Dict[str, Any], Callable[[Dict[str, Any]], None]], None, None]:
    """
    Context manager for atomic JSON file operations with file locking.
    
    Provides exclusive access to a JSON file, preventing race conditions when
    multiple processes (e.g., SLURM jobs) try to read/write simultaneously.
    
    Usage:
        with atomic_json_file(path) as (data, save):
            data["key"] = "value"
            save(data)  # Atomic write
    
    Args:
        path: Path to the JSON file.
        create_if_missing: If True, create the file with empty dict if it
            doesn't exist. If False, raise FileNotFoundError.
    
    Yields:
        Tuple of (data, save_fn):
            - data: Current JSON contents as dict
            - save_fn: Function to atomically save new data
    
    Raises:
        FileNotFoundError: If file doesn't exist and create_if_missing=False
        json.JSONDecodeError: If file contains invalid JSON
    
    Note:
        Uses fcntl.LOCK_EX for exclusive locking. The lock is held for the
        entire duration of the context, so keep operations short.
        
        File creation uses atomic 'x' mode to prevent race conditions when
        multiple SLURM jobs start simultaneously. Lock acquisition timing
        is logged at DEBUG level for cluster debugging.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Atomic file creation to prevent race conditions between concurrent jobs.
    # Using 'x' mode fails with FileExistsError if file exists, which is safe
    # to ignore since another process already created it.
    if not path.exists() and create_if_missing:
        try:
            with open(path, "x") as f:
                f.write("{}")
            logger.debug(f"Created empty JSON file: {path}")
        except FileExistsError:
            # Another process created the file between our exists() check and
            # open() call - this is fine, we'll use their file
            pass
    elif not path.exists() and not create_if_missing:
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, "r+") as f:
        # Acquire exclusive lock with telemetry for debugging cluster issues
        start_time = time.perf_counter()
        logger.debug(f"Waiting for exclusive lock on {path}...")
        
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        
        elapsed = time.perf_counter() - start_time
        logger.debug(f"Acquired lock on {path} after {elapsed:.4f}s")
        
        try:
            # Read current contents
            content = f.read().strip()
            data: Dict[str, Any] = json.loads(content) if content else {}
            
            def save(new_data: Dict[str, Any]) -> None:
                """Atomically save new data to the file."""
                f.seek(0)
                f.truncate()
                json.dump(new_data, f, indent=2)
                f.flush()
            
            yield data, save
            
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            logger.debug(f"Released lock on {path}")
