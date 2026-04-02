# ==============================================================================
#                           SECTION: UDP CAMERA CLIENT (NO CHUNKING)
# ==============================================================================
"""
UDP camera implementation - RAW Packet Mode (One packet = One frame).
"""

import logging
import socket
import time
from typing import Optional
import cv2
import numpy as np

try:
    from . import config
except ImportError:
    class DummyConfig:
        UDP_PORT = 7751
        UDP_HOST = "0.0.0.0"
        UDP_BUFFER_SIZE = 65535 # Nhận tối đa giới hạn của 1 gói UDP
        SOCKET_TIMEOUT = 1.0
    config = DummyConfig()

class UDPCamera:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Tăng bộ đệm nhận của hệ thống lên để tránh mất gói khi xử lý AI nặng
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        
        try:
            self.sock.bind((config.UDP_HOST, config.UDP_PORT))
            logging.info(f"UDP Receiver (No Chunking) started on port {config.UDP_PORT}")
        except Exception as e:
            logging.error(f"Could not bind: {e}")
            
        self.sock.settimeout(config.SOCKET_TIMEOUT)

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Nhận trực tiếp 1 gói UDP và giải mã thành ảnh.
        """
        try:
            # 1. Nhận nguyên 1 gói tin
            data, addr = self.sock.recvfrom(config.UDP_BUFFER_SIZE)
            
            if not data:
                return None

            # 2. Giải mã trực tiếp từ mảng Byte (JPEG RAW)
            frame_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                # Nếu payload là 1456, khả năng cao là ảnh đã bị Router chặt đứt đuôi
                logging.debug(f"Decode failed. Packet size: {len(data)} bytes")
                return None
                
            return frame

        except socket.timeout:
            return None
        except Exception as e:
            logging.error(f"Error: {e}")
            return None

    def release(self):
        if self.sock:
            self.sock.close()