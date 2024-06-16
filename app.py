import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model('ResNet50_TransferLearning/resnet_model.h5')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def predict_class(img):
    if img is None:
        st.error("No image provided.")
        return None

    img = img.resize((48, 48)).convert('L')
    img = np.array(img)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    return emotions[predicted_class[0]]

st.markdown(
    """
    <style>
    h2{
        text-align: center;
        font-family: 'poppins', sans-serif;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 2px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #000;
        margin-top: 2rem;
    }
    .copyright {
        text-align: center;
        font-size: 0.8rem;
        color: #000;
        margin-top: 1rem;
    }
    .footer a{
        font-family: 'poppins', sans-serif;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<title>Human Emotion Classifier</title> <div class="main"> <h2>Emotion Classifier</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    st.image(image, caption='Uploaded Image.', use_column_width=False)

    if st.button("Predict"):
        emotion = predict_class(image)
        if emotion:
            st.success(f"Predicted Emotion: {emotion}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('''
    <div class="footer">
        Developed by Prabhat Kumar Raj
        <br>
        <a href="https://github.com/Prabhat-2101/Emotion_Detection_Using_DeepLearning" target="_blank">View on GitHub</a>
    </div>
    <div class="copyright">
        &copy; 2024 All rights reserved @moodscannerai
    </div>
    ''', unsafe_allow_html=True)