# SignBridge 🤝  
*Real-Time Sign Language Recognition and Translation*

## 📌 Overview
**SignBridge** is a real-time Sign Language Recognition and Translation system designed to bridge the communication gap between the Deaf/Hard-of-Hearing and the hearing community. It uses computer vision and deep learning to recognize ASL gestures and convert them into readable English text in real time.

## 🎯 Features
- 🔠 Real-time recognition of ASL alphabets (A–Z)
- 📷 Live webcam input and gesture detection
- 🧠 Lightweight MobileNetV2 model with high accuracy
- ✋ Hand landmark tracking using MediaPipe
- 💬 Dynamic translation display and reset functionality
- 🕒 Translation history tracking

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras (MobileNetV2)
- **Computer Vision**: OpenCV, MediaPipe

## 💻 Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/SignBridge.git
cd SignBridge
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Unzip the trained model
- The trained model is included in the repository as a .7z file.
- Download and extract asl_mobilenetv2_mediapipe_resized.7z using 7-Zip or any compatible tool.
- Place the extracted asl_mobilenetv2_mediapipe_resized.h5 file in the project root directory.

### 5. Run the application 
```bash
python app.py
```

