# slr.py
import cv2
import mediapipe as mp
import numpy as np
import time
from translate import Translator
from tensorflow.keras.models import load_model
from threading import Lock
import asyncio
from concurrent.futures import ThreadPoolExecutor

class SignLanguageProcessor:
    def __init__(self):
        self.model = load_model('asl_mobilenetv2_mediapipe_resized.h5')
        
        # Mediapipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Class Labels
        self.classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # Translation setup
        self.target_language = 'es'
        self.translator = Translator(to_lang=self.target_language)
        self.translation_executor = ThreadPoolExecutor(max_workers=1)
        
        # State variables
        self.spelled_word = []
        self.history = []
        self.translated_word = None
        self.last_prediction_time = time.time()
        self.last_double_time = time.time()
        self.last_prediction = None
        
        # Thread safety
        self.lock = Lock()
        
        # Camera variables
        self.cap = None
        self.camera_active = False
        self.start_camera()
        
        # Constants
        self.debounce_time = 3
        self.double_letter_time = 2
        self.confidence_threshold = 0.7

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = self.initialize_camera()
        self.camera_active = True

    def stop_camera(self):
        if self.cap is not None:
            self.camera_active = False
            self.cap.release()
            self.cap = None

    def is_camera_active(self):
        return self.camera_active and self.cap is not None and self.cap.isOpened()

    def initialize_camera(self):
        for i in range(5):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
            print(f"Retrying to open camera... Attempt {i+1}/5")
            time.sleep(1)
        print("Error: Could not open webcam.")
        return None

    def resize_to_mobilenet_format(self, pseudo_image):
        return cv2.resize(pseudo_image, (32, 32))

    def extract_landmarks_as_image(self, frame):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            pseudo_image = landmarks.reshape(7, 9)
            return self.resize_to_mobilenet_format(pseudo_image), results
        return np.zeros((32, 32)), results

    def translate_word(self):
        if not self.spelled_word:
            return None
            
        word = ''.join(self.spelled_word).lower()
        try:
            future = self.translation_executor.submit(self.translator.translate, word)
            self.translated_word = future.result(timeout=5)  # 5 second timeout
            return self.translated_word
        except Exception as e:
            print(f"Translation Error: {e}")
            return None

    def process_frame(self):
        if not self.is_camera_active():
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Reduce frame size for processing
        frame = cv2.resize(frame, (640, 480))
        
        pseudo_image, results = self.extract_landmarks_as_image(frame)
        
        if np.any(pseudo_image):
            pseudo_image = pseudo_image.reshape(1, 32, 32, 1)
            prediction = self.model.predict(pseudo_image, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]
            predicted_label = self.classes[predicted_class]

            current_time = time.time()
            with self.lock:
                if confidence > self.confidence_threshold:
                    if (not self.spelled_word or self.spelled_word[-1] != predicted_label) and \
                            (current_time - self.last_prediction_time > self.debounce_time):
                        self.spelled_word.append(predicted_label)
                        self.last_prediction_time = current_time
                        self.last_prediction = predicted_label
                        self.last_double_time = current_time
                    elif self.spelled_word and self.spelled_word[-1] == predicted_label and \
                            (current_time - self.last_double_time > self.double_letter_time):
                        self.spelled_word.append(predicted_label)
                        self.last_double_time = current_time

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Add text overlays
        if self.last_prediction:
            cv2.putText(frame, f'Predicted: {self.last_prediction}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        frame_height, frame_width, _ = frame.shape
        cv2.rectangle(frame, (0, frame_height - 60), (frame_width, frame_height), (0, 0, 0), -1)
        
        with self.lock:
            cv2.putText(frame, 'Word: ' + ''.join(self.spelled_word), (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, ''.join(self.spelled_word)

    def reset_word(self):
        with self.lock:
            self.history.append(''.join(self.spelled_word))
            self.spelled_word = []
            self.translated_word = None

    def complete_word(self):
        with self.lock:
            if self.spelled_word:
                self.history.append(''.join(self.spelled_word))
                self.spelled_word = []

    def add_space(self):
        with self.lock:
            if self.spelled_word:
                self.spelled_word.append(" ")

    def remove_last_letter(self):
        with self.lock:
            if self.spelled_word:
                self.spelled_word.pop()

    def get_history(self):
        return self.history.copy()