# ==============================================================================
#                                   SECTION: LOGGING SETUP
# ==============================================================================
"""
Logging configuration for the application.
"""

import logging


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
    )
