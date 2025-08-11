# American Sign Language Converter 🤟

An intelligent ASL (American Sign Language) recognition tool that uses your webcam to detect hand signs in real time and convert them into English letters using a trained machine learning model.

<img width="1440" height="900" alt="ASLthumnail" src="https://github.com/user-attachments/assets/76d34e5d-cf4e-42c2-9e80-fb1dcdd3657d" />


---

## 📸 How It Works

This ASL Converter uses **computer vision** and **machine learning** to recognize hand gestures from a webcam feed and convert them into text in real-time.

### ⚙️ Technical Overview

- **Camera Input**: Uses your device’s webcam to capture live hand gestures
- **Hand Detection**: Powered by [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) for accurate hand landmark tracking
- **Model Prediction**: A trained machine learning model (Random Forest) processes the landmark data to predict the corresponding ASL letter
- **Live Output**: The predicted letter is displayed on screen as you sign

---

## 🚀 Features

- 🖐️ Real-time hand gesture recognition
- 🔤 Translates signs into English letters instantly
- 📷 Webcam-based — no special hardware needed
- 🤖 Built with Python, OpenCV, MediaPipe, and Scikit-learn

---



