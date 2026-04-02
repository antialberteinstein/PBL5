# ==============================================================================
#                           SECTION: UDP CAMERA CLIENT (FIXED)
# ==============================================================================
"""
UDP camera implementation - Optimized for ESP32-CAM Push Stream.
"""

import logging
import socket
import time
from typing import Optional
import cv2
import numpy as np

# Giả sử ông có file config.py, nếu không hãy tạo hoặc thay bằng biến trực tiếp
try:
    from . import config
except ImportError:
    # Dự phòng nếu không load được config
    class DummyConfig:
        UDP_PORT = 7751
        UDP_HOST = "0.0.0.0" # Nghe trên tất cả các interface
        UDP_BUFFER_SIZE = 8192 # Phải lớn hơn CHUNK_SIZE của ESP32
        SOCKET_TIMEOUT = 1.0
    config = DummyConfig()

class UDPCamera:
    """
    Camera provider implementation using UDP Listener.
    """
    def __init__(self):
        # Khởi tạo socket một lần duy nhất để tránh chiếm dụng tài nguyên
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Ép kiểu cho phép chạy lại cổng cũ nếu crash
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # QUAN TRỌNG: Phải bind cổng thì mới "hứng" được dữ liệu từ ESP32
        try:
            self.sock.bind((config.UDP_HOST, config.UDP_PORT))
            logging.info(f"UDP Camera Listener started on {config.UDP_HOST}:{config.UDP_PORT}")
        except Exception as e:
            logging.error(f"Could not bind to port: {e}")
            
        self.sock.settimeout(config.SOCKET_TIMEOUT)
        self.chunks = {}
        self.total_chunks = None

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Listen and reassemble frame chunks sent by ESP32-CAM.
        """
        print('NNNN1')
        start_time = time.time()
        self.chunks = {} # Reset buffer cho frame mới
        self.total_chunks = None
        print('NNNNN2')
        
        while True:
            print('NNNNN3')
            try:
                print('NNNNNNN4')
                # 1. Nhận dữ liệu từ cổng UDP
                data, addr = self.sock.recvfrom(config.UDP_BUFFER_SIZE)
                
                # 2. Tách Header (Format: chunk_idx|total_chunks|data)
                try:
                    # Tìm 2 dấu gạch đứng đầu tiên
                    first_pipe = data.find(b'|')
                    second_pipe = data.find(b'|', first_pipe + 1)
                    
                    if first_pipe == -1 or second_pipe == -1:
                        continue
                        
                    header_str = data[:second_pipe].decode('utf-8')
                    chunk_data = data[second_pipe + 1:]
                    
                    parts = header_str.split('|')
                    chunk_idx = int(parts[0])
                    curr_total = int(parts[1])
                    
                except Exception as e:
                    logging.warning(f"Header error: {e}")
                    continue

                # 3. Đồng bộ hóa frame
                # Nếu nhận được chunk 0, coi như bắt đầu frame mới, xóa đống cũ đi
                if chunk_idx == 0:
                    self.chunks = {0: chunk_data}
                    self.total_chunks = curr_total
                elif self.total_chunks == curr_total:
                    self.chunks[chunk_idx] = chunk_data
                
                # 4. Kiểm tra xem đã đủ "mảnh ghép" chưa
                if self.total_chunks and len(self.chunks) == self.total_chunks:
                    break
                    
                # 5. Thoát nếu đợi quá lâu (Tránh treo app khi mất gói tin nặng)
                if time.time() - start_time > config.SOCKET_TIMEOUT:
                    logging.warning("Frame assembly timeout - dropping partial frame")
                    return None
                    
            except socket.timeout:
                logging.debug("Socket timeout - No data from ESP32")
                return None
            except Exception as e:
                logging.error(f"Error during recv: {e}")
                return None

        # 6. Ghép các mảnh và giải mã ảnh
        try:
            # Sắp xếp các mảnh theo đúng thứ tự
            full_frame_data = b''.join(self.chunks[i] for i in range(self.total_chunks))
            
            frame_array = np.frombuffer(full_frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                logging.error(f"Decode failed! Payload size: {len(full_frame_data)} bytes")
                return None
                
            return frame
        except Exception as e:
            logging.error(f"Reassembly error: {e}")
            return None

    def release(self):
        """Giải phóng socket"""
        if self.sock:
            self.sock.close()
            logging.info("UDP Camera socket closed.")