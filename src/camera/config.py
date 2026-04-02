# ==============================================================================
#                           SECTION: CAMERA CONFIGURATION
# ==============================================================================
"""
Configuration constants for camera/UDP module.
"""

# UDP Camera Server Configuration
UDP_HOST = "0.0.0.0"
UDP_PORT = 7751
UDP_BUFFER_SIZE = 65535  # Maximum UDP packet size
SOCKET_TIMEOUT = 0.5  # Timeout in seconds for UDP socket operations

# Camera Source Configuration
CAMERA_INDEX = 0  # Index for local camera (usually 0 for built-in webcam)
