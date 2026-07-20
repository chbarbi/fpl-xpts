""" Logging configuration. Call setup_logging() once from the entry point. """

import logging

from fpl_model.config import LOG_PATH


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging: file + console output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode='w'), # Wipes log after each run 
            logging.StreamHandler(),
        ]
    )