# 🎯 Vanishing Points Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Detect vanishing points in images using Computer Vision techniques**

[Features](#-features) • 
[Installation](#-installation) • 
[Usage](#-usage) • 
[How It Works](#-how-it-works) • 
[Screenshots](#-screenshots)

</div>

---

## 📌 Overview

Vanishing Points Detector is a powerful computer vision application that automatically identifies vanishing points in perspective images. Using advanced algorithms like Hough Transform, DBSCAN clustering, and RANSAC refinement, it accurately detects where parallel lines converge in architectural photos, road scenes, corridors, and any image with strong perspective geometry.

### 🎯 What are Vanishing Points?
Vanishing points are points in an image where parallel lines appear to meet. They are fundamental concepts in perspective drawing and computer vision, enabling:
- 3D reconstruction from 2D images
- Camera calibration and pose estimation
- Perspective correction and image rectification
- Scene understanding for autonomous vehicles

---

## ✨ Features

### Core Capabilities
- ✅ **Automatic Line Detection** - Uses Canny edge detection and Hough Transform
- ✅ **Multiple Vanishing Points** - Supports 1-point and 2-point perspective
- ✅ **RANSAC Refinement** - Improves accuracy by removing outliers
- ✅ **DBSCAN Clustering** - Groups intersection points intelligently
- ✅ **Real-time Visualization** - Interactive display with color-coded results

### User Interface
- 🎨 **Modern Web UI** - Built with Streamlit for easy interaction
- ⚙️ **Adjustable Parameters** - Fine-tune detection for different image types
- 📊 **Detailed Metrics** - Confidence scores, line counts, perspective type
- 💾 **Export Results** - Download annotated images with detection results

### Technical Features
- 🔄 **Adaptive Thresholding** - Automatic parameter adjustment based on image size
- 📐 **Angle Filtering** - Ignores horizontal/vertical lines to reduce noise
- 🧩 **Smart Sampling** - Optimized intersection calculation for large line sets
- 🎲 **Reproducible Results** - Fixed random seed for consistent outputs

---

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository

  git clone https://github.com/MohammedElshora2005/vanishing-points-detector.git
  cd vanishing-points-detector

### Step 2: Install Requirements
  pip install -r requirements.txt

### Step 3: Run the App
  streamlit run perception.py


📖 How to Use
1- Upload your image (JPG, PNG, BMP)
2- Click "Start Analysis"
3- View the vanishing points
4- Download the result

📁 Project Files
perception.py - Main application
requirements.txt - Dependencies
README.md - This file

📝 License
MIT License - feel free to use and modify

👨‍💻 Author
@MohammedElshora2005
