"""
Simple test client for Flask Camera Server
Captures image and sends message
"""
import requests

SERVER_URL = "http://127.0.0.1:7749"

# 1. Capture image
print("Capturing image...")
response = requests.post(f"{SERVER_URL}/api/cam/capture")
with open("captured_image.jpg", "wb") as f:
    f.write(response.content)
print(f"Image saved: captured_image.jpg ({len(response.content)} bytes)")

# 2. Send message
print("Sending message...")
requests.post(
    f"{SERVER_URL}/api/cam/result",
    json={"message": "OK NHE EM IU"}
)
print("Message sent!")
