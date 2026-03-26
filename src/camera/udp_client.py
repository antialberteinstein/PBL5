# ==============================================================================
#                           SECTION: UDP CAMERA CLIENT
# ==============================================================================
"""
UDP camera implementation.
"""

import logging
import socket
from typing import Optional

import cv2
import numpy as np

from . import config


class UDPCamera:
    """
    Camera provider implementation using UDP client.
    """
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from UDP camera server.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(config.SOCKET_TIMEOUT)
        
        try:
            ### 1. Send GET_FRAME request to server
            sock.sendto(b"GET_FRAME", (config.UDP_HOST, config.UDP_PORT))
            
            ### 2. Receive chunks
            chunks = {}
            total_chunks = None
            
            while True:
                try:
                    data, _ = sock.recvfrom(config.UDP_BUFFER_SIZE)
                    
                    ### 3. Check for error messages
                    if data.startswith(b"ERROR|"):
                        error_msg = data.decode('utf-8', errors='ignore')
                        logging.error(f"Server error: {error_msg}")
                        return None
                    
                    ### 4. Parse chunk header
                    # Format: chunk_index|total_chunks|data
                    try:
                        header_end = data.index(b'|', data.index(b'|') + 1) + 1
                        header = data[:header_end].decode('utf-8')
                        chunk_data = data[header_end:]
                        
                        parts = header.rstrip('|').split('|')
                        if len(parts) != 2:
                            continue
                        
                        chunk_idx = int(parts[0])
                        curr_total_chunks = int(parts[1])
                    except (ValueError, IndexError):
                        logging.warning("Invalid chunk header, skipping")
                        continue
                    
                    ### 5. Initialize total chunks on first chunk
                    if total_chunks is None:
                        total_chunks = curr_total_chunks
                    
                    ### 6. Store chunk
                    chunks[chunk_idx] = chunk_data
                    
                    ### 7. Check if all chunks received
                    if len(chunks) == total_chunks:
                        break
                        
                except socket.timeout:
                    if chunks:
                        logging.warning(f"Timeout: received {len(chunks)}/{total_chunks} chunks")
                    return None
            
            ### 8. Reassemble frame
            frame_data = b''.join(chunks[i] for i in range(total_chunks))
            
            ### 9. Decode image
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            return frame
            
        except Exception as e:
            logging.error(f"Error capturing frame: {e}")
            return None
        finally:
            sock.close()

    def send_result(self, message: str) -> None:
        """
        Send result message to camera server via UDP.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            ### Send message to server with RESULT| prefix
            formatted_message = f"RESULT|{message}"
            sock.sendto(
                formatted_message.encode('utf-8'),
                (config.UDP_HOST, config.UDP_PORT)
            )
            
            ### Wait for acknowledgment (optional)
            sock.settimeout(0.1)
            try:
                ack, _ = sock.recvfrom(1024)
                if ack == b"ACK":
                    logging.debug("Result acknowledged by server")
            except socket.timeout:
                pass  # Server might not send ACK
                
        except Exception as e:
            logging.error(f"Error sending result: {e}")
        finally:
            sock.close()
            
    def release(self) -> None:
        """
        Release any resources held by the camera provider.
        """
        pass
