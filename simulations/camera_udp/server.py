"""
UDP Socket Camera Server
Provides ultra-low latency frame streaming via UDP sockets
"""
import logging
import socket
import threading
import time
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 7751
CHUNK_SIZE = 8192  # 8KB - safe for macOS UDP limits
JPEG_QUALITY = 50  # Lower quality for smaller frames

# Global variables
latest_frame = None
frame_lock = threading.Lock()


class CameraThread(threading.Thread):
    """Background thread to continuously capture frames from webcam"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.camera = None
        
    def run(self):
        """Continuously capture frames from webcam"""
        global latest_frame
        
        logger.info("Starting camera capture thread...")
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            return
        
        # Warm up camera
        logger.info("Warming up camera...")
        time.sleep(1.0)
        for _ in range(5):
            self.camera.read()
            time.sleep(0.1)
        
        logger.info("Camera ready, starting continuous capture")
        
        while self.running:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                with frame_lock:
                    latest_frame = frame.copy()
            time.sleep(0.033)  # ~30 FPS
        
        self.camera.release()
        logger.info("Camera capture thread stopped")
    
    def stop(self):
        """Stop the camera thread"""
        self.running = False


def send_frame_chunks(sock, addr, frame_data):
    """
    Send frame data in chunks via UDP.
    
    Args:
        sock: UDP socket
        addr: Client address
        frame_data: JPEG encoded frame bytes
    """
    total_size = len(frame_data)
    num_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_size)
        chunk = frame_data[start:end]
        
        # Format: chunk_index|total_chunks|data
        header = f"{i}|{num_chunks}|".encode()
        packet = header + chunk
        
        # Ensure packet is under UDP limit (65507 bytes)
        if len(packet) > 65000:
            logger.error(f"Packet too large: {len(packet)} bytes, skipping chunk {i}")
            continue
        
        try:
            sock.sendto(packet, addr)
            time.sleep(0.001)  # Small delay to prevent packet loss
        except OSError as e:
            logger.error(f"Failed to send chunk {i}: {e}")
            continue
    
    logger.debug(f"Sent {num_chunks} chunks ({total_size} bytes) to {addr}")


def handle_client_request(sock, data, addr):
    """
    Handle incoming client requests.
    
    Args:
        sock: UDP socket
        data: Request data
        addr: Client address
    """
    global latest_frame
    
    try:
        message = data.decode('utf-8', errors='ignore')
        
        if message == "GET_FRAME":
            # Client requesting a frame
            with frame_lock:
                if latest_frame is None:
                    # Send error message
                    sock.sendto(b"ERROR|No frame available", addr)
                    return
                
                frame = latest_frame.copy()
            
            
            # Encode frame as JPEG with lower quality for smaller size
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ret:
                sock.sendto(b"ERROR|Encode failed", addr)
                return
            
            frame_data = buffer.tobytes()
            
            # Send frame in chunks
            send_frame_chunks(sock, addr, frame_data)
            
        elif message.startswith("RESULT|"):
            # Client sending verification result
            result_message = message[7:]  # Remove "RESULT|" prefix
            logger.info(f"Verification result: {result_message}")
            
            # Send acknowledgment
            sock.sendto(b"ACK", addr)
            
        else:
            logger.warning(f"Unknown request from {addr}: {message}")
            sock.sendto(b"ERROR|Unknown", addr)
            
    except Exception as e:
        logger.error(f"Error handling request from {addr}: {str(e)}")
        try:
            # Truncate error message to prevent "Message too long" error
            error_msg = str(e)[:50]  # Limit to 50 characters
            sock.sendto(f"ERROR|{error_msg}".encode(), addr)
        except:
            # If even that fails, send generic error
            try:
                sock.sendto(b"ERROR|Server error", addr)
            except:
                pass


def run_udp_server():
    """Run the UDP server"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Allow address reuse to prevent "Address already in use" errors
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    sock.bind((UDP_IP, UDP_PORT))
    
    logger.info(f"UDP server listening on {UDP_IP}:{UDP_PORT}")
    
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            # Handle each request in the main thread (UDP is stateless)
            handle_client_request(sock, data, addr)
            
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        sock.close()


if __name__ == '__main__':
    logger.info("Starting UDP Socket Camera Server...")
    
    # Start camera capture thread
    camera_thread = CameraThread()
    camera_thread.start()
    
    # Give camera time to initialize
    time.sleep(2)
    
    try:
        run_udp_server()
    finally:
        camera_thread.stop()
        logger.info("Server stopped")
