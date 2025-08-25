# 🤟 Indian Sign Language Recognition System

A comprehensive AI-powered system for real-time Indian Sign Language (ISL) recognition and translation. This project includes data collection tools, model training notebooks, and a real-time translation application with speech synthesis.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Data Collection](#data-collection)
- [Technical Details](#technical-details)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Real-time Hand Detection**: Uses MediaPipe for accurate hand landmark detection
- **Multi-class Recognition**: Supports 35 classes (A-Z, 0-9, space, delete)
- **Live Translation**: Real-time sign language to text conversion
- **Speech Synthesis**: Text-to-speech functionality for accessibility
- **Modern GUI**: Clean, responsive interface with dual camera views
- **Data Collection Tools**: Automated dataset creation with hand tracking
- **Multiple Model Architectures**: Support for ResNet and MobileNetV2

## 📁 Project Structure

```
project/
├── collect_imgs.py                    # Data collection script
├── Script.py                         # Main real-time translation app
├── Indian_sign_language_Mobilenetv2.ipynb  # MobileNetV2 training
├── Indian sign language resnet.ipynb  # ResNet training
├── indian-sign-language-classification.ipynb  # Classification notebook
├── README.md                         # This file
└── data_images/                      # Collected image dataset (generated)
    ├── A/
    ├── B/
    └── ...
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam for real-time detection
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd indian-sign-language-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional packages**
   ```bash
   pip install opencv-python mediapipe tensorflow pillow pyttsx3
   ```

## 📖 Usage

### Real-time Translation

Run the main application for live sign language translation:

```bash
python Script.py
```

**Controls:**
- **🔊 Speak Text**: Convert detected text to speech (Ctrl+S)
- **⏹️ Stop Speaking**: Stop current speech (Esc)
- **🗑️ Clear Text**: Clear the text area (Ctrl+C)
- **⏸️ Pause Detection**: Toggle hand detection (Ctrl+P)
- **❌ Quit**: Exit the application

**Keyboard Shortcuts:**
- `Ctrl+S`: Speak detected text
- `Ctrl+C`: Clear text area
- `Ctrl+P`: Pause/resume detection
- `Esc`: Stop speech

### Data Collection

Collect training data for new signs:

```bash
python collect_imgs.py
```

**Instructions:**
1. Run the script
2. For each class (A-Z, 0-9, space, delete):
   - Press 'Q' when ready to start capturing
   - Show the sign to the camera
   - The script will automatically capture 500 images per class
   - Press 'ESC' to quit early

## 🧠 Model Training

### MobileNetV2 Training

Open `Indian_sign_language_Mobilenetv2.ipynb` for training with MobileNetV2 architecture.

**Features:**
- Transfer learning with pre-trained MobileNetV2
- Data augmentation for better generalization
- Early stopping and learning rate scheduling
- Three-way data split (train/validation/test)
- Regularization techniques to prevent overfitting

### ResNet Training

Use `Indian sign language resnet.ipynb` for ResNet-based training.

### Training Process

1. **Data Preparation**:
   - Images are resized to 224x224 pixels
   - RGB color space conversion
   - Data augmentation (rotation, zoom, flip)

2. **Model Architecture**:
   - Base model: MobileNetV2/ResNet (pre-trained)
   - Global Average Pooling
   - Dropout layers for regularization
   - Dense output layer (35 classes)

3. **Training Configuration**:
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Callbacks: Early stopping, learning rate reduction
   - Batch size: 32
   - Epochs: 50 (with early stopping)

## 📊 Data Collection

The `collect_imgs.py` script provides automated data collection:

**Features:**
- Real-time hand detection using MediaPipe
- Automatic bounding box calculation
- Image cropping and resizing
- Keypoint extraction and storage
- Progress tracking and status display

**Output:**
- `data_images/`: Cropped and resized images (224x224)
- `data_keypoints/`: Hand landmark coordinates (.npy files)

## 🔧 Technical Details

### Hand Detection
- **Library**: MediaPipe Hands
- **Detection**: Real-time hand landmark detection
- **Landmarks**: 21 3D points per hand
- **Confidence**: Configurable detection thresholds

### Model Architecture
- **Input**: 224x224x3 RGB images
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Output**: 35-class classification
- **Activation**: Softmax for multi-class prediction

### Performance Optimizations
- **GPU Acceleration**: TensorFlow GPU support
- **Batch Processing**: Efficient inference
- **Memory Management**: Proper resource cleanup
- **Threading**: Non-blocking speech synthesis

## 📋 Requirements

### Core Dependencies
```
tensorflow>=2.8.0
opencv-python>=4.5.0
mediapipe>=0.8.0
numpy>=1.21.0
pillow>=8.0.0
pyttsx3>=2.90
tkinter (built-in)
```

### Optional Dependencies
```
matplotlib>=3.5.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

### System Requirements
- **OS**: Windows 10/11, macOS, Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space
- **Camera**: USB webcam or built-in camera
- **GPU**: NVIDIA GPU with CUDA support (optional)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Areas
- Model improvements and new architectures
- Data augmentation techniques
- UI/UX enhancements
- Performance optimizations
- Documentation improvements
- Bug fixes and error handling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Hand detection and landmark extraction
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision library
- **Indian Sign Language Community**: For inspiration and feedback

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Include system specifications and error messages

## 🔮 Future Enhancements

- [ ] Support for dynamic signs (words/phrases)
- [ ] Multi-hand gesture recognition
- [ ] Real-time video translation
- [ ] Mobile app development
- [ ] Cloud-based processing
- [ ] Integration with communication platforms

---

**Made with ❤️ for the Indian Sign Language community**
