import streamlit as st 
import numpy as np
import pickle

# ================================
#     STREAMLIT PAGE CONFIG
# ================================

st.set_page_config(
    page_title="Ames Housing Prediction",
    page_icon="🏘️ ",
    layout="centered"
)

st.markdown("""
   <h2 style='text-align:center; color:#4A90E2;'>
       рџЏ  Ames Housing Price Prediction
   </h2>
   <p style='text-align:center; color:gray; font-size:17px;'>
       Input manual fitur → Prediksi harga rumah (USD & IDR)
   </p>
   <br>
""", unsafe_allow_html=True)

# ================================
#        LOAD MODEL PACKAGE
# ================================

try:
    package = pickle.load(open("RFREG_model.pkl", "rb"))
except Exception:
    st.error("⚠️ File RFREG_model.pkl tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py.")
    st.stop()

# model & error metrics from notebook
model = package["model"]
MODEL_MAE_TRAIN = package["mae_train"]
MODEL_MAE_TEST = package["mae_test"]
MODEL_R2_TRAIN = package["r2_train"]
MODEL_R2_TEST = package["r2_test"]

# ================================
#  INPUT FITUR (VERTIKAL)
# ================================

st.header("📝 Input Fitur Rumah")
overall_qual = st.slider("Overall Quality (1 - 10)", 1, 10, 5)
overall_cond = st.slider("Overall Condition (1 - 9)", 1, 9, 5)
central_air = st.selectbox("Central Air", ["Yes", "No"])
gr_liv_area = st.number_input("Gr Liv Area (sqft)", 200, 7000, 1500)
total_bsmt_sf = st.number_input("Total Basement SF", 0, 3000, 800)

central_air_val = 1 if central_air == "Yes" else 0

# final feature vector
features = np.array([[overall_qual,
                      overall_cond,
                      gr_liv_area,
                      central_air_val,
                      total_bsmt_sf]])

# ================================
#           PREDIKSI
# ================================

st.markdown("---")
st.subheader(" Prediksi Harga")

predict_btn = st.button(" Prediksi Sekarang", use_container_width=True)

if predict_btn:

    price = model.predict(features)[0]

    # konversi USD в†’ IDR
    usd_to_idr = 16000
    price_idr = price * usd_to_idr

    st.success("Prediksi berhasil!")
    
    st.markdown(f"""
       <h3 style='text-align:center; color:#4A90E2;'>
            Perkiraan Harga (USD): <b>${price:,.0f}</b><br>
            Perkiraan Harga (IDR): <b>Rp {price_idr:,.0f}</b>
       </h3>
   """, unsafe_allow_html=True)

# ================================
#     PENJELASAN PREDIKSI
# ================================
st.markdown("---")
st.subheader(" Penjelasan Prediksi")
st.markdown(f"""
            **Interpretasi Berdasarkan Input:**
            -   **Overall Quality = {overall_qual}**  
                Level ini {"tinggi" if overall_qual >= 7 else "sedang" if overall_qual >= 5 else "rendah"} dan sangat memengaruhi harga.
                
            -   **Overall Condition = {overall_cond}**  
                Menjelaskan kondisi struktural & pemeliharaan rumah.
            
            -   **Gr Liv Area = {gr_liv_area:,} sqft**
                Rumah seluas ini termasuk kategori {"besar" if gr_liv_area > 2000 else "standar"}.
                
            -   **Central Air = {central_air}**  
                Kehadiran AC sentral meningkatkan nilai rumah.
            
            -   **Total Basement SF = {total_bsmt_sf:,} sqft**  
                Basement luas menambah area fungsional.
                
            --
            ###  Akurasi Model (dibaca otomatis dari model)
            - **MAE Train:** В± **${MODEL_MAE_TRAIN:,.0f}**
            - **MAE Test:** В± **${MODEL_MAE_TEST:,.0f}**
            - **RВІ Train:** **{MODEL_R2_TRAIN:.2f}**
            - **RВІ Test:** **{MODEL_R2_TEST:.2f}**
            
            **Apa artinya?**
            - Prediksi model biasanya meleset sekitar **${MODEL_MAE_TEST:,.0f}** dari harga asli.  
            - Model menjelaskan sekitar **{MODEL_R2_TEST*100:.0f}%** variasi harga rumah.  - Prediksi adalah **estimasi**, bukan harga pasti вЂ” ada margin error yang wajar.
            
            --
            
            **Catatan tentang model:**  
            Random Forest mempelajari pola non-linear dari banyak pohon keputusan sehingga mampu memprediksi harga berdasarkan kombinasi fitur secara lebih fleksibel.""")
