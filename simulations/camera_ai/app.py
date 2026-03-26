"""
Flask Camera Simulation Server
Provides APIs for camera capture and message logging
"""
import logging
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


@app.route('/api/cam/capture', methods=['POST'])
def capture_image():
    """
    Capture an image from the computer's webcam and return it.
    
    Returns:
        Response: JPEG image or error message
    """
    try:
        logger.info("Received request to capture image from webcam")
        
        # Open the default camera (index 0)
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            logger.error("Failed to open camera")
            return jsonify({
                'error': 'Camera not available',
                'message': 'Could not access the webcam'
            }), 500
        
        # Allow camera to warm up and adjust exposure
        # This fixes the black image issue
        logger.info("Warming up camera...")
        time.sleep(1.0)  # Give camera time to initialize
        
        # Read and discard several frames to let auto-exposure adjust
        for i in range(5):
            camera.read()
            time.sleep(0.1)
        
        # Now capture the actual frame
        ret, frame = camera.read()
        camera.release()
        
        if not ret or frame is None:
            logger.error("Failed to capture frame from camera")
            return jsonify({
                'error': 'Capture failed',
                'message': 'Could not capture image from webcam'
            }), 500
        
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
        
        logger.info(f"Successfully captured image ({len(image_bytes)} bytes)")
        
        # Return the image as response
        return Response(
            image_bytes,
            mimetype='image/jpeg',
            headers={'Content-Disposition': 'inline; filename=capture.jpg'}
        )
        
    except Exception as e:
        logger.error(f"Error during image capture: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/cam/result', methods=['POST'])
def log_message():
    """
    Receive a message and log it to the console.
    
    Expected JSON payload:
        {
            "message": "Your message here"
        }
    
    Returns:
        JSON: Success or error response
    """
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
        
        # Log the message to console
        logger.info(f"Client message: {message}")
        
        return jsonify({
            'success': True,
            'message': 'Message logged successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'endpoints': {
            '/api/cam/capture': 'POST - Capture image from webcam',
            '/api/cam/result': 'POST - Log message to console'
        }
    })


if __name__ == '__main__':
    logger.info("Starting Flask Camera Simulation Server...")
    app.run(host='127.0.0.1', port=7749, debug=True)
