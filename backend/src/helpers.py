"""
Helper functions.
"""
import logging


def setup_logging_levels():
    """
    Basic logging setup.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')
