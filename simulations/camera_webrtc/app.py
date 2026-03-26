"""
Flask WebRTC Camera Server
Provides real-time video streaming and frame capture API
"""
import logging
import threading
import time
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for frame storage
latest_frame = None
frame_lock = threading.Lock()
latest_result = None
result_lock = threading.Lock()


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


@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')


@app.route('/api/frame', methods=['POST', 'GET'])
def get_frame():
    """
    Get the latest captured frame from webcam.
    
    Returns:
        Response: JPEG image or error message
    """
    global latest_frame
    
    try:
        with frame_lock:
            if latest_frame is None:
                return jsonify({
                    'error': 'No frame available',
                    'message': 'Camera not ready or no frame captured yet'
                }), 503
            
            frame = latest_frame.copy()
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        
        if not ret:
            logger.error("Failed to encode image as JPEG")
            return jsonify({
                'error': 'Encoding failed',
                'message': 'Could not encode captured image'
            }), 500
        
        # Convert to bytes
        image_bytes = buffer.tobytes()
        
        # Return the image as response
        return Response(
            image_bytes,
            mimetype='image/jpeg',
            headers={'Content-Disposition': 'inline; filename=frame.jpg'}
        )
        
    except Exception as e:
        logger.error(f"Error getting frame: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/result', methods=['POST'])
def receive_result():
    """
    Receive verification result from AI server.
    
    Expected JSON payload:
        {
            "message": "Verification result message"
        }
    
    Returns:
        JSON: Success or error response
    """
    global latest_result
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            logger.warning("Received request with no JSON data")
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must be JSON'
            }), 400
        
        # Extract message
        message = data.get('message', '')
        
        if not message:
            logger.warning("Received request with empty message")
            return jsonify({
                'error': 'Invalid request',
                'message': 'Message field is required'
            }), 400
        
        # Store the latest result
        with result_lock:
            latest_result = {
                'message': message,
                'timestamp': time.time()
            }
        
        # Log the message to console
        logger.info(f"Verification result: {message}")
        
        return jsonify({
            'success': True,
            'message': 'Result received successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing result: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/latest_result', methods=['GET'])
def get_latest_result():
    """
    Get the latest verification result.
    
    Returns:
        JSON: Latest result or null if no result available
    """
    global latest_result
    
    with result_lock:
        if latest_result is None:
            return jsonify({'result': None})
        
        # Return result if it's less than 5 seconds old
        if time.time() - latest_result['timestamp'] < 5:
            return jsonify({
                'result': latest_result['message'],
                'timestamp': latest_result['timestamp']
            })
        else:
            return jsonify({'result': None})


@app.route('/video_feed')
def video_feed():
    """
    Video streaming route for web interface.
    """
    def generate():
        global latest_frame
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.1)
                    continue
                frame = latest_frame.copy()
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    logger.info("Starting Flask WebRTC Camera Server...")
    
    # Start camera capture thread
    camera_thread = CameraThread()
    camera_thread.start()
    
    # Give camera time to initialize
    time.sleep(2)
    
    try:
        app.run(host='127.0.0.1', port=7750, debug=False, threaded=True)
    finally:
        camera_thread.stop()
