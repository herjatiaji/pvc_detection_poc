import streamlit as st
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw
import os

# --- 1. Konfigurasi Halaman Web ---
st.set_page_config(page_title="Sistem Deteksi Pipa", layout="centered")
st.title("Pipe Counting System 🔍")
st.write("Upload foto tumpukan pipa untuk mendapatkan ID dan total jumlahnya.")

# --- 2. Komponen Upload Foto ---
uploaded_file = st.file_uploader("Pilih foto pipa (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_image.jpg"
    
    # Membaca gambar menggunakan Pillow (PIL) dan menyimpannya secara fisik
    # Ini 100% mencegah error 400 Bad Request dari Roboflow!
    image = Image.open(uploaded_file).convert("RGB")
    image.save(temp_path)

    with st.spinner('Sedang menghitung dan menganalisis pipa...'):
        try:
            # --- 3. Inisialisasi Client Roboflow ---
            CLIENT = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key="wsDVlZyPMCoMTu7Z9los" # Ganti dengan API Key Anda
            )

            # Konfigurasi NMS & Confidence
            custom_config = InferenceConfiguration(iou_threshold=0.8, confidence_threshold=0.6)

            # Melakukan Inferensi dengan membaca FILE FISIK
            with CLIENT.use_configuration(custom_config):
                result = CLIENT.infer(temp_path, model_id="pipe-counting-3/1")

            # --- 4. Menggambar ID dan Menghitung dengan Pillow ---
            if "predictions" in result:
                total_pipa = len(result["predictions"])
                
                # Siapkan "kanvas" untuk menggambar di atas gambar
                draw = ImageDraw.Draw(image)
                
                for index, pred in enumerate(result["predictions"]):
                    pipe_id = f"ID-{index + 1}"
                    
                    x_center = pred['x']
                    y_center = pred['y']
                    width = pred['width']
                    height = pred['height']
                    
                    # Hitung koordinat sudut
                    x_min = x_center - (width / 2)
                    y_min = y_center - (height / 2)
                    x_max = x_center + (width / 2)
                    y_max = y_center + (height / 2)
                    
                    # Menggambar Bounding Box (warna hijau, tebal 3)
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="#00FF00", width=3)
                    
                    # Menambahkan Teks ID Pipa (warna merah)
                    draw.text((x_min, y_min - 15), pipe_id, fill="#FF0000")

                # --- 5. Tampilkan Hasil Akhir ---
                st.success(f"✅ Deteksi Selesai! Total pipa terhitung: {total_pipa}")
                
                # Menampilkan gambar langsung ke Streamlit
                st.image(image, caption="Hasil Deteksi Pipa PVC", use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses API: {e}")
            
        finally:
            # --- PENGHAPUSAN FILE AMAN ---
            if os.path.exists(temp_path):
                os.remove(temp_path)