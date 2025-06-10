# ♻️ EcoSnap  : Intelligent Waste Classification & Recycling Guide

🌍 Overview
EcoSnap is a deep learning-powered application that uses MobileNet to classify waste and provide tailored recycling suggestions. With a user-friendly interface built using Streamlit, the app promotes sustainable waste management by helping users quickly identify how to properly dispose of various types of waste.

🚀 Features
📷 Real-time Waste Classification
Upload a waste image and get an instant prediction of its category using MobileNet.

🔁 Recycling Suggestions
Receive actionable tips on how to properly recycle or dispose of each type of waste.

💻 Streamlit Web App
A simple and interactive UI for users to classify waste from any device.

🗂️ Dataset
Source: TrashNet Dataset
Link: https://www.kaggle.com/datasets/feyzazkefe/trashnet/code 

Classes:
Plastic
Metal
Glass
Paper
Cardboard
Trash

Preprocessing:
Image resizing to match MobileNet input requirements
Normalization and augmentation to improve model robustness

🧠 Model Architecture
Component	Description
Base Model	: MobileNet (pretrained on ImageNet)
Fine-tuning :	Last few layers retrained on TrashNet
Optimizer :	Adam
Loss Function	: Categorical Cross-Entropy
Metric : Accuracy

📊 Results
✅ Training Accuracy: ~92%
🧪 Validation Accuracy: ~88%
📉 Loss: Optimized to minimize classification errors

🔧 Customization
📐 Image Size Adjustment
Ensures compatibility and visibility of uploaded images

📚 Expanded Recycling Info
Provides detailed recycling tips for each category

🧪 How to Use
🔗 Open the Streamlit web app (see Live Demo below)
📤 Upload an image of waste
📈 View the predicted category
♻️ Read recycling suggestions and dispose responsibly

🌐 Live Demo
👉 

🧠 Future Improvements
🗺️ GPS-based local recycling info
📱 Mobile-friendly PWA version
🧠 Integration with YOLO 

🙌 Acknowledgements
TrashNet Dataset
MobileNet (Keras)


