import logging


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with console handler only.

    Args:
        name: Name of the logger

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create formatter
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger
