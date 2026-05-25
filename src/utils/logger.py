import logging
import sys


def setup_logger(
    logger_name: str,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a reusable logger instance.

    Parameters
    ----------
    logger_name : str
        Name of the logger.

    log_level : int
        Logging verbosity level.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers during multiple imports
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # ----------------------------------------
    # Console Handler
    # ----------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # ----------------------------------------
    # Log Format
    # ----------------------------------------
    formatter = logging.Formatter(
        fmt=(
            "[%(asctime)s] "
            "[%(levelname)s] "
            "[%(name)s] "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Prevent logs from propagating to root logger
    logger.propagate = False

    return logger