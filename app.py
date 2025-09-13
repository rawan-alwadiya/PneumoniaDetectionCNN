import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(
    repo_id="RawanAlwadeya/PneumoniaDetectionCNN", 
    filename="chest_xray_cnn.h5"  
)

model = tf.keras.models.load_model(model_path)


# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model("best_model.h5")
#     return model

# model = load_model()


def preprocess_image(image):
    IMG_SIZE = (224, 224)

    image = image.convert("L")  
    image = image.resize(IMG_SIZE)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  
    img_array = np.expand_dims(img_array, axis=0)   

    return img_array


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Detection"])


if page == "Home":
    st.markdown("<h1 style='text-align: center;'>🩺 Pneumonia Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Deep Learning Powered Chest X-Ray Classification</h3>", unsafe_allow_html=True)

    st.write(
        """
        Pneumonia is an infection that inflames the air sacs in one or both lungs.  
        It can be caused by bacteria, viruses, or fungi, and is a serious condition 
        that requires medical attention.  
        
        This app demonstrates how Convolutional Neural Networks (CNNs) can assist 
        in detecting Pneumonia from chest X-Ray images.  
        """
    )

    st.image("Pneumonia.jpg", caption="Chest X-Ray: Pneumonia vs Normal", use_container_width=True)

    st.info("👉 Go to the **Detection** page from the left sidebar to upload your chest X-Ray image and get predictions.")


elif page == "Detection":
    
    st.markdown("<h1 style='text-align: center;'>🩺 Pneumonia Detection App</h1>", unsafe_allow_html=True)
    st.write("Upload a chest X-Ray image below, and the model will predict whether it shows **Pneumonia** or is **Normal**.")

    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0][0]

    
        if prediction >= 0.5:
            st.error("⚠️ The model predicts that this X-Ray **likely shows Pneumonia**. Please consult a medical professional for confirmation.")
        else:
            st.success("✅ The model predicts that this X-Ray **likely appears Normal**. For medical certainty, always consult a professional.")
