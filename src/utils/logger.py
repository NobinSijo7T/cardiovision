"""
CARDIOVISION - Logger Configuration
Provides a configured logger with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "cardiovision",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_dir: str = "outputs"
) -> logging.Logger:
    """
    Create and configure a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Log file name. If None, defaults to '{name}.log'.
        level: Logging level.
        log_dir: Directory for log files.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_file = f"{name}.log"
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "cardiovision") -> logging.Logger:
    """
    Get an existing logger by name, or create a basic one.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
