# Stickman: Real-Time Pose Detection and Visualization

## 🚀 Project Overview

Stickman is a Python application that uses computer vision to detect human body poses and creates a real-time stick figure visualization. The project leverages MediaPipe for pose detection and OpenCV for video processing, creating a side-by-side comparison of the original video feed and a stylized stick figure representation.

## ✨ Features

- **Real-time pose detection** using MediaPipe's Pose model
- **Stick figure visualization** that mirrors the user's movements
- **Custom face overlay** on the stick figure head
- **Movement detection** with classification of common poses:
  - Arms raised
  - Hands up
  - Hip tilt
  - Arms down
  - Neutral position
- **Full body landmark tracking** with 33 pose landmarks
- **Side-by-side display** of original video and stick figure visualization

## 🛠️ Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- NumPy

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stickman.git
cd stickman
```

2. Install the required packages:
```bash
pip install opencv-python mediapipe numpy
```

3. (Optional) Add a custom face image:
   - Replace the URL in the code with a local path to your image
   - Recommended size: 30x30 pixels
   - Supported formats: PNG (with transparency), JPG

## 🎮 Usage

1. Run the script:
```bash
python stickman.py
```

2. Position yourself in front of the camera
3. The application will display:
   - Left panel: Original video feed with pose landmarks
   - Right panel: Stick figure visualization
   - Movement status text at the top of the video feed

4. Press 'q' to quit the application

## 🧠 How It Works

1. **Pose Detection**:
   - MediaPipe's Pose model detects 33 body landmarks in real-time
   - Landmarks include head, shoulders, elbows, wrists, hips, knees, ankles, and more

2. **Stick Figure Creation**:
   - A blank canvas is created for the stick figure
   - Connections between landmarks are drawn with rose-colored lines
   - All landmarks are marked with small black circles
   - A custom face image is overlaid on the head position

3. **Movement Detection**:
   - Calculates angles between key body points (shoulders, elbows, wrists)
   - Analyzes positions of body parts relative to each other
   - Classifies the current pose based on predefined rules

4. **Visualization**:
   - Original video feed is displayed on the left
   - Stick figure is displayed on the right
   - Movement status is shown as text overlay

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe for the pose detection model
- OpenCV for computer vision capabilities
- The image used for the face overlay is from [Pinterest](https://i.pinimg.com/736x/70/1f/e8/701fe8ac0ec65f3d20e20ff89f78767c.jpg)

---

*Move in front of the camera and watch as your movements are transformed into a stick figure in real-time!* 🕺💃
