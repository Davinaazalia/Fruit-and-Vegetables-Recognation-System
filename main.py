import streamlit as st 
import tensorflow as tf
import numpy as np

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('trained_model.h5')

model = load_my_model()

def model_prediction(input_arr):
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    st.image("Image.jpg")

elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("Dataset ini berisi gambar tentang variasi buah dan sayuran seperti :")
    st.code("Buah : Pisang, Apel, Pear, Anggur, Jeruk, Kiwi, Melon, Delima, Nanas, Mangga")
    st.code("Sayuran : Mentimun, Wortel, Capsicum, Bawang, Kentang, Lemon, Tomat, Lobak, Bit, Kubis, Selada, Bayam, Kedelai, Kembang Kol, Paprika, Cabai, Lobak, Jagung, Jagung, Ubi Jalar, Paprika, Jalapeño, Jahe, Bawang Putih, Kacang Polong, Terong")
    st.subheader("Content")
    st.text("The dataset is organized into three main folders:")
    st.text("1. Train: Contains 100 images per category")
    st.text("2. Test: Contains 10 images per category")
    st.text("3. Validation: Contains 10 images per category")

elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Pilih Gambar", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, width=250)  

        if st.button("Predict"):
            image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])
            
            result_index = model_prediction(input_arr)

            with open("labels.txt") as file:
                content = file.read().splitlines()
                if result_index < len(content):
                    st.success(f"Model memprediksi bahwa ini adalah {content[result_index]}")
                else:
                    st.error("Index hasil prediksi di luar jangkauan, cek model dan labels.txt!")

            st.balloons()
