import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import instaloader
import tempfile
import os
import shutil


st.set_page_config(
    page_title="Human vs AI Image Detector",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

@st.cache_resource
def load_keras_model(model_path):
    """Memuat model Keras dari path yang diberikan."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# --- Fungsi untuk Preprocessing Gambar ---
def preprocess_image(image, target_size=(512, 512)):
    """
    Mengubah ukuran dan menormalisasi gambar agar sesuai
    dengan input model.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize gambar
    img = image.resize(target_size)
    
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 
    
    return img_array


def main():
    st.title("🤖 Detektor Gambar: AI vs Manusia")
    st.markdown("""
        Aplikasi ini bisa mendeteksi apakah gambar dibuat oleh **AI** atau oleh **Manusia** dengan menggunakan deep learning.
    """)
    st.divider()
    
    model = load_keras_model('./models/MobileNetV2-AIvsHumanGenImages-Final.keras')
    if model is None:
        st.warning("Model tidak dapat dimuat. Pastikan file model ada di direktori yang sama.")
        return

    uploaded_file = st.file_uploader(
        "Pilih sebuah gambar...",
         type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG"
    )
        
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
            
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
            
        with st.spinner('Menganalisis gambar...'):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
                
            prob_ai = float(prediction[0][0])
            prob_human = 1 - prob_ai
            hasil_prediksi = "Manusia" if prob_human > prob_ai else "AI"
            confidence = max(prob_human, prob_ai)

        with col2:
            st.subheader("Hasil Prediksi:")
            if hasil_prediksi == "Manusia":
                st.success("✅ Gambar ini sepertinya dibuat oleh **Manusia**.")
            else:
                st.error("🤖 Gambar ini sepertinya dibuat oleh **AI**.")
                
            st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2%}")
            st.write("Detail Probabilitas:")
            st.write(f"- Manusia: `{prob_human:.2%}`")
            st.write(f"- AI: `{prob_ai:.2%}`")


if __name__ == "__main__":
    main()