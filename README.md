# Real-Time Object Detection and Tracking with YOLOv8

This project implements a real-time object detection and tracking system using YOLOv8 and background subtraction. The application uses a graphical user interface (GUI) built with Tkinter to display the video feed and control the detection process.

## Features

- **Real-time Object Detection**: Detects and tracks objects in live video streams using YOLOv8.
- **Background Subtraction**: Applies background subtraction to enhance detection accuracy.
- **GUI Controls**: Provides start and stop buttons for easy control of the detection process.
- **Class-Based Detection**: Differentiates between human, hand, and other object detections.
- **Top Detections Highlighted**: Highlights the top detections with bounding boxes and labels.

## Requirements

- Python 3.7+
- `opencv-python`
- `numpy`
- `Pillow`
- `ultralytics` (for YOLOv8)
- `tkinter` (usually included with Python standard library)

You can install the necessary Python packages using pip:

```bash
pip install opencv-python numpy Pillow ultralytics
```

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```

2. **Download YOLOv8 Model Weights:**

   Make sure you have the YOLOv8 model weights file `yolov8n.pt`. You can download it from the [Ultralytics YOLOv8 GitHub page](https://github.com/ultralytics/yolov5/releases).

3. **Run the Application:**

   Navigate to the project directory and run the main script:

   ```bash
   python main.py
   ```

## Code Overview

- **`main.py`**: The main script that initializes the YOLO model, captures video frames, applies background subtraction, performs object detection, and updates the Tkinter GUI.

### Key Components

- **YOLO Model Initialization**: Loads the YOLOv8 model for object detection.
- **Video Capture**: Captures live video from the webcam.
- **Background Subtraction**: Applies background subtraction to focus on moving objects.
- **Detection and Tracking**: Processes detection results and updates the GUI with bounding boxes and labels.
- **Tkinter GUI**: Provides an interface to start and stop detection and displays the video feed.

## Usage

- **Start Detection**: Click the "Start" button to begin object detection and tracking.
- **Stop Detection**: Click the "Stop" button to stop the detection process and close the video feed.

## Troubleshooting

- **Model File Not Found**: Ensure `yolov8n.pt` is located in the same directory as `main.py`.
- **Camera Access Issues**: Check that your webcam is properly connected and accessible.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements or find bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
