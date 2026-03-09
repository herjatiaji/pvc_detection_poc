import cv2
import streamlit as st
import numpy as np
from inference_sdk import InferenceHTTPClient
import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
# --- 1. Konfigurasi Halaman Web ---
st.set_page_config(page_title="Sistem Deteksi Pipa", layout="centered")
st.title("Pipe Counting System")
st.write("Upload foto tumpukan pipa untuk mendapatkan ID dan total jumlahnya.")

# --- 2. Komponen Upload Foto ---
uploaded_file = st.file_uploader("Pilih foto pipa (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca file gambar yang diupload menjadi format OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Menyimpan sementara untuk diproses Roboflow

    with st.spinner('Sedang menghitung dan menganalisis pipa...'):
        try:
            # --- 3. Inisialisasi Client Roboflow ---
            CLIENT = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key="wsDVlZyPMCoMTu7Z9los"# Ganti dengan API Key Anda
            )

            # Melakukan Inferensi
            # --- 3. Inisialisasi Client Roboflow ---
            CLIENT = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key="wsDVlZyPMCoMTu7Z9los" # Ganti dengan API Key Anda
            )

            # --- KONFIGURASI NMS & CONFIDENCE ---
            # iou_threshold = 0.8 (Semakin tinggi, semakin mengizinkan kotak tumpang tindih)
            # confidence_threshold = 0.4 (Menurunkan batas keyakinan agar pipa samar ikut terdeteksi)
            custom_config = InferenceConfiguration(iou_threshold=0.8, confidence_threshold=0.6)

            # Melakukan Inferensi dengan konfigurasi custom
            with CLIENT.use_configuration(custom_config):
                result = CLIENT.infer(img, model_id="pipe-counting-3/1")

            # --- 4. Menggambar ID dan Menghitung ---
            if "predictions" in result:
                total_pipa = len(result["predictions"])
                
                for index, pred in enumerate(result["predictions"]):
                    pipe_id = f"ID-{index + 1}"
                    
                    x_center = int(pred['x'])
                    y_center = int(pred['y'])
                    width = int(pred['width'])
                    height = int(pred['height'])
                    
                    # Konversi koordinat tengah ke sudut OpenCV
                    x_min = int(x_center - (width / 2))
                    y_min = int(y_center - (height / 2))
                    x_max = int(x_center + (width / 2))
                    y_max = int(y_center + (height / 2))
                    
                    # Menggambar Bounding Box
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Menambahkan Teks ID Pipa
                    cv2.putText(img, pipe_id, (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Konversi warna BGR (OpenCV) ke RGB (agar warna tidak terbalik di web)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # --- 5. Tampilkan Hasil Akhir ---
                st.success(f"✅ Deteksi Selesai! Total pipa terhitung: {total_pipa}")
                
                # Menampilkan gambar dengan bounding box dan ID
                st.image(img_rgb, caption=f"Hasil Deteksi Pipa PVC", use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        
        finally:
            # Hapus file sementara agar storage tidak penuh
            if os.path.exists(temp_path):
                os.remove(temp_path)