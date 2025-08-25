# 🤟 Indian Sign Language Recognition System

An **AI-powered real-time Indian Sign Language (ISL) recognition and translation system**.  
Includes **data collection tools, training notebooks, and a real-time translation app with speech synthesis**.

---

## 📑 Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Real-time Translation](#real-time-translation)  
  - [Data Collection](#data-collection)  
- [Model Training](#model-training)  
- [Technical Details](#technical-details)  
- [Requirements](#requirements)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  
- [Future Enhancements](#future-enhancements)  
- [Support](#support)  

---

## 🔎 Overview
This project enables **real-time ISL-to-text and speech translation**.  
It supports **35 sign classes** (A–Z, 0–9, space, delete) and comes with tools for **dataset collection** and **deep learning model training** using **MobileNetV2 and ResNet**.

---

## ✨ Features
- 🖐️ **Hand Detection** – MediaPipe for accurate landmark detection  
- 🧾 **Multi-class Recognition** – 35 classes supported  
- 🗣️ **Live Translation** – Sign-to-text with speech synthesis  
- 💻 **Modern GUI** – Clean, dual camera view interface  
- 📊 **Data Collection Tools** – Automated dataset generation  
- 🧠 **Multiple Models** – MobileNetV2 & ResNet architectures  
- ⚡ **Performance Optimized** – GPU acceleration, efficient inference  

---

## 📁 Project Structure
```
project/
├── Script.py                           # Real-time translation app
├── collect_imgs.py                     # Data collection script
├── Indian_sign_language_Mobilenetv2.ipynb   # MobileNetV2 training
├── Indian sign language resnet.ipynb        # ResNet training
├── indian-sign-language-classification.ipynb # Classification notebook
├── data_images/                        # Image dataset (generated)
└── README.md                           # Documentation
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+  
- Webcam  
- CUDA-compatible GPU (optional, for faster training)  

### Setup
```bash
git clone <repository-url>
cd indian-sign-language-recognition
pip install -r requirements.txt
pip install opencv-python mediapipe tensorflow pillow pyttsx3
```

---

## 📖 Usage

### Real-time Translation
```bash
python Script.py
```
**Controls:**  
- `Ctrl+S` → Speak detected text  
- `Ctrl+C` → Clear text area  
- `Ctrl+P` → Pause/resume detection  
- `Esc` → Stop speech  

### Data Collection
```bash
python collect_imgs.py
```
**Steps:**  
1. Run the script  
2. Select class (A–Z, 0–9, space, delete)  
3. Press `Q` to start capturing (500 images/class)  
4. Press `Esc` to quit early  

---

## 🧠 Model Training
- **MobileNetV2** → Transfer learning, augmentation, early stopping  
- **ResNet** → Pre-trained architecture with fine-tuning  

**Training Highlights:**  
- Input: 224×224 RGB images  
- Optimizer: Adam, Loss: Categorical Crossentropy  
- Callbacks: Early stopping, LR scheduler  
- Epochs: 50 (with early stopping)  

---

## 🔧 Technical Details
- **Hand Detection**: MediaPipe (21 landmarks/hand)  
- **Model Input**: 224×224×3 images  
- **Output**: 35-class softmax prediction  
- **Optimizations**: GPU acceleration, memory management, threaded speech synthesis  

---

## 📋 Requirements
**Core:**  
```
tensorflow>=2.8.0
opencv-python>=4.5.0
mediapipe>=0.8.0
numpy>=1.21.0
pillow>=8.0.0
pyttsx3>=2.90
tkinter (built-in)
```
**Optional:**  
```
matplotlib>=3.5.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

**System:**  
- OS: Windows / macOS / Linux  
- RAM: 8GB+ (16GB recommended)  
- GPU: NVIDIA CUDA (optional)  

---

## 🤝 Contributing
1. Fork repo  
2. Create branch (`feature/xyz`)  
3. Commit changes  
4. Push & open Pull Request  

---

## 📄 License
MIT License – see [LICENSE](LICENSE).  

---

## 🙏 Acknowledgments
- MediaPipe – hand detection  
- TensorFlow – deep learning  
- OpenCV – computer vision  
- Indian Sign Language community – inspiration & feedback  

---

## 🔮 Future Enhancements
- [ ] Dynamic sign (words/phrases) support  
- [ ] Multi-hand gesture recognition  
- [ ] Real-time video translation  
- [ ] Mobile app version  
- [ ] Cloud-based processing  
- [ ] Communication platform integration  

---

## 📞 Support
- Check [Issues](https://github.com/your-repo/issues)  
- Open a new issue with details (system specs + error logs)  

---

✨ *Made with ❤️ for the Indian Sign Language community*  
