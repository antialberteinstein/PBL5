import cv2
import numpy as np
import urllib.request
import socket

class HTTPCamera:
    def __init__(self):
        # Tăng timeout lên hẳn 20 giây để ESP32 kịp "thở"
        self.url = "http://172.20.10.3:81/stream"
        self.stream = None
        self.byte_buffer = b''

        try:
            print(f"🔄 Đang thử kết nối tới {self.url}...")
            # Vô hiệu hóa Proxy hệ thống để tránh đi đường vòng
            proxy_support = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_support)
            urllib.request.install_opener(opener)
            
            # Mở kết nối với timeout dài
            self.stream = urllib.request.urlopen(self.url, timeout=20)
            print("✅ KẾT NỐI THÀNH CÔNG!")
        except socket.timeout:
            print("❌ LỖI: Timeout - ESP32 không trả lời kịp.")
        except Exception as e:
            print(f"❌ LỖI KẾT NỐI: {e}")

    def capture_frame(self):
        if not self.stream: return None
        try:
            # Đọc từng cụm data nhỏ để tránh tràn buffer
            chunk = self.stream.read(1024)
            if not chunk: return None
            self.byte_buffer += chunk
            
            a = self.byte_buffer.find(b'\xff\xd8')
            b = self.byte_buffer.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg = self.byte_buffer[a:b+2]
                self.byte_buffer = self.byte_buffer[b+2:]
                return cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        except:
            return None
        return None