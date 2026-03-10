import streamlit as st
from PIL import Image, ImageDraw
import requests
import base64
import io

# --- 1. Konfigurasi Halaman Web ---
st.set_page_config(page_title="Sistem Deteksi Pipa", layout="centered")
st.title("Pipe Counting System")
st.write("Upload foto tumpukan pipa untuk mendapatkan ID dan total jumlahnya.")

# --- 2. Komponen Upload Foto ---
uploaded_file = st.file_uploader("Pilih foto pipa (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar menggunakan Pillow (PIL)
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner('Sedang mengirim data ke server AI...'):
        try:
            # --- 3. Mengubah Gambar menjadi Teks (Base64) ---
            # Ini membuat kita tidak perlu menyimpan file temp_image.jpg di server!
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("ascii")

            # --- 4. Memanggil REST API Roboflow secara Langsung ---
            # Endpoint standar deteksi Roboflow (tanpa inference-sdk)
            api_url = "https://detect.roboflow.com/pipe-counting-3/1"
            
            # Parameter API (API Key, Confidence 60%, Overlap NMS 80%)
            params = {
                "api_key": "wsDVlZyPMCoMTu7Z9los", # API Key Anda
                "confidence": 60,
                "overlap": 80
            }
            
            # Melakukan HTTP POST Request
            response = requests.post(
                api_url, 
                params=params, 
                data=img_str, 
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            # Membaca hasil balasan dari Roboflow
            result = response.json()

            # --- 5. Menggambar ID dan Menghitung ---
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

                # --- 6. Tampilkan Hasil Akhir ---
                st.success(f"✅ Deteksi Selesai! Total pipa terhitung: {total_pipa}")
                st.image(image, caption="Hasil Deteksi Pipa PVC", use_container_width=True)
            else:
                st.warning("Tidak ada pipa yang terdeteksi atau terjadi kesalahan format API.")
                st.write("Respons Server:", result) # Untuk keperluan debugging jika kosong

        except Exception as e:
            st.error(f"Terjadi kesalahan komunikasi API: {e}")