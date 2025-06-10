import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io

# Load your trained model (adjust the path if needed)
@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model('best_model.keras')
    return model

model = load_model()

# Define the target image size (same as used during training)
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Define your class mapping (update if necessary)
class_indices = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
idx2label = {v: k for k, v in class_indices.items()}

def load_and_preprocess_image(image_data):
    """Load an image from file bytes and preprocess it."""
    # Open the image using PIL
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    # Resize the image
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    # Convert image to numpy array
    img_array = image.img_to_array(img_resized)
    # Expand dimensions to match model input (1, IMG_HEIGHT, IMG_WIDTH, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image using MobileNetV2's preprocessing
    img_array = preprocess_input(img_array)
    return img_array, img

# Streamlit App Layout
st.title("EcoSnap♻️ ")
st.write("Upload an image of waste, and the model will predict its class along with detailed recycling instructions.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image_data = uploaded_file.read()
    # Preprocess the image and get original image for display
    img_array, original_img = load_and_preprocess_image(image_data)
    
    # Make prediction with the model
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    predicted_label = idx2label[predicted_class]
    confidence = predictions[0][predicted_class]
    
    # Display results in Streamlit with a reduced image width (e.g., 300 pixels)
    st.image(original_img, caption="Uploaded Image", width=300)
    st.write(f"**Prediction:** {predicted_label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
    
    # Expanded recycling suggestions with more detailed information
    recycling_suggestions = {
        'cardboard': (
            "Flatten cardboard boxes and keep them dry to prevent contamination. "
            "Place them in the designated cardboard recycling bin. "
            "Remove any tape, labels, or food residue before recycling."
        ),
        'glass': (
            "Rinse glass containers and sort them by color if required locally. "
            "Place clean glass in the appropriate recycling container. "
            "Do not include ceramics, mirrors, or broken glass unless specified."
        ),
        'metal': (
            "Rinse aluminum and steel items to remove food residue. "
            "Recycle them in the metal bin or contact a scrap metal collector for larger items. "
            "Avoid mixing with non-metal materials."
        ),
        'paper': (
            "Ensure paper is clean, dry, and free from grease or food. "
            "Flatten and stack or bundle paper before placing it in the recycling bin. "
            "Avoid glossy, laminated, or heavily dyed paper unless accepted locally."
        ),
        'plastic': (
            "Rinse plastics to remove food and check for the recycling symbol and number. "
            "Sort by type if required and place in the designated plastic recycling bin. "
            "Avoid recycling soft plastics or contaminated items unless your facility accepts them."
        ),
        'trash': (
            "Dispose of non-recyclable items in general waste. "
            "Before discarding, consider reusing, repairing, or donating usable items. "
            "Check local guidelines for proper disposal of hazardous, electronic, or bulky waste."
        )
    }
    
    suggestion = recycling_suggestions.get(predicted_label, "No suggestion available.")
    st.write(f"**Recycling Suggestion:** {suggestion}")
