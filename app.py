# app.py
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import cv2
import base64
import os
import time
from slr import SignLanguageProcessor
from threading import Lock

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'slr'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins='*')

thread_lock = Lock()
sign_processor = None
processing_active = False

def init_processor():
    global sign_processor
    with app.app_context():
        sign_processor = SignLanguageProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    global processing_active
    processing_active = True
    with thread_lock:
        eventlet.spawn(generate_frames)

def generate_frames():
    global sign_processor, processing_active
    last_frame_time = time.time()
    frame_interval = 0.03

    while processing_active:
        current_time = time.time()
        if current_time - last_frame_time < frame_interval:
            eventlet.sleep(0)
            continue

        if sign_processor is None or not sign_processor.is_camera_active():
            eventlet.sleep(0.1)
            continue

        try:
            frame, predicted_word = sign_processor.process_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = base64.b64encode(buffer).decode('utf-8')
                
                socketio.emit('video_frame', {
                    'frame': frame_bytes,
                    'predicted_word': predicted_word
                })
                last_frame_time = current_time
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            import traceback
            traceback.print_exc()
            eventlet.sleep(0.1)
            continue

        eventlet.sleep(0)

@socketio.on('control')
def handle_control(data):
    global sign_processor, processing_active
    key = data.get('key')
    
    if key == 'start':
        if sign_processor is None:
            init_processor()
        else:
            sign_processor.start_camera()
        processing_active = True
        with thread_lock:
            eventlet.spawn(generate_frames)
        return {'status': 'success', 'message': 'Camera started'}
    
    elif key == 'backspace':
        sign_processor.remove_last_letter()
    
    if sign_processor is None:
        return {'status': 'error', 'message': 'Processor not initialized'}
    
    try:
        if key == 'r':
            sign_processor.reset_word()
        elif key == 'c':
            sign_processor.complete_word()
        elif key == 's':
            sign_processor.add_space()
        elif key == 't':
            eventlet.spawn(handle_translation)
        elif key == 'q':
            processing_active = False
            sign_processor.stop_camera()
        return {'status': 'success'}
    except Exception as e:
        print(f"Error in handle_control: {e}")
        return {'status': 'error', 'message': str(e)}

def handle_translation():
    try:
        translation = sign_processor.translate_word()
        if translation:
            socketio.emit('translation', {'text': translation})
    except Exception as e:
        print(f"Translation error: {e}")
        socketio.emit('translation_error', {'message': 'Translation failed'})

if __name__ == '__main__':
    init_processor()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)