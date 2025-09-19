import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Class names (modify if your classes are different)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI scan image to detect the type of brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# Reply Generator Function
def generate_reply(predicted_class):
    replies = {
        "Glioma": "⚠️ The scan indicates a possible *Glioma*, a tumor affecting the brain's supportive tissues. Please consult a neurologist immediately for further diagnosis.",
        "Meningioma": "⚠️ This scan suggests *Meningioma*, a typically benign tumor originating from the brain's membranes. Early medical consultation is advised.",
        "Pituitary": "⚠️ A potential *Pituitary Tumor* is detected, often affecting hormone levels. Endocrinologist consultation is recommended.",
        "No Tumor": "✅ No signs of tumor were detected in the MRI scan. However, always follow up with a medical professional if symptoms persist."
    }
    return replies.get(predicted_class, "Unable to interpret the scan result.")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess the image
    img = image.convert("RGB").resize((150, 150))  # force 3 channels
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 150, 150, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Generate and display reply
    reply = generate_reply(predicted_class)
    st.markdown(f"### 🧪 Prediction: `{predicted_class}`")
    st.markdown(f"### 🔬 Confidence: `{confidence * 100:.2f}%`")
    st.markdown(f"### 🤖 Medical Assistant Suggestion:\n{reply}")
