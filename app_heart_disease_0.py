import streamlit as st
import pandas as pd
import joblib # Ganti pickle dengan joblib karena kita menggunakannya di skrip pelatihan
import time
from PIL import Image
from sklearn.preprocessing import StandardScaler # Pastikan StandardScaler diimpor

st.set_page_config(
    page_title="CVD_MODEL_PREDICTION",
    page_icon = "🫀",
    layout = 'wide'
)

st.write("""
# Heart CVD Model Prediction
""")

# --- Load Model, Scaler, dan Feature Names di awal aplikasi ---
# Ini penting agar tidak loading setiap kali button predict ditekan
try:
    # Ganti nama file sesuai dengan yang kamu simpan
    loaded_model = joblib.load('generate_heart_disease.pkl')
    scaler = joblib.load('scaler.pkl') # Perhatikan: scaler_jantung_fix.pkl diubah jadi scaler.pkl
    expected_cols = joblib.load('features.pkl') # Perhatikan: feature_names_for_model.pkl diubah jadi features.pkl
    st.success("Model, Scaler, dan Daftar Fitur berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: Pastikan 'generate_heart_disease.pkl', 'scaler.pkl', dan 'features.pkl' ada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika file tidak ditemukan
except Exception as e:
    st.error(f"Error saat memuat file: {e}")
    st.stop()

# --- Definisi fungsi user_input_features() ---
def user_input_features():
    st.sidebar.header('Manual Input')

    # Perhatikan: Sesuaikan pilihan nilai selectbox cp agar sesuai dengan dataset asli (0,1,2,3)
    # Jika dataset aslimu menggunakan 1,2,3,4 seperti yang kamu tulis, pastikan mappingnya benar.
    # Dari head dataset, terlihat cp menggunakan 0, 1, 2, 3.
    cp_options_numeric = [0, 1, 2, 3] # Menggunakan 0, 1, 2, 3 sesuai head dataset
    def format_cp_display(option):
        if option == 0:
            return "0 - Typical Angina"
        elif option == 1:
            return "1 - Atypical Angina"
        elif option == 2:
            return "2 - Non-Anginal Pain"
        elif option == 3:
            return "3 - Asymptomatic"
        return str(option)

    # UI untuk Chest Pain (cp)
    if 'show_cp_info' not in st.session_state:
        st.session_state.show_cp_info = False
    
    with st.sidebar:
        col1_cp, col2_cp = st.columns([0.8, 0.2])

        with col1_cp:
            cp = st.selectbox(
                'Chest Pain (cp)',
                options=cp_options_numeric,
                format_func=format_cp_display,
                index=0, # Default ke Typical Angina (0)
                key='chest_pain_selectbox'
            )

        with col2_cp:
            st.write("") 
            st.write("") 
            if st.button('❔', key='cp_info_button'):
                st.session_state.show_cp_info = not st.session_state.show_cp_info

        if st.session_state.show_cp_info:
            with st.expander("Detail Chest Pain (CP)", expanded=True):
                st.write("""
                **0 - Typical Angina:** Nyeri dada yang khas, biasanya terkait aktivitas fisik dan mereda dengan istirahat. \n
                **1 - Atypical Angina:** Nyeri dada yang kurang khas. \n
                **2 - Non-Anginal Pain:** Nyeri dada yang tidak berasal dari masalah jantung. \n
                **3 - Asymptomatic:** Tidak ada nyeri dada yang dilaporkan. \n """)
    
    # UI untuk Maximum HR (thalach)
    if 'show_thalach_info' not in st.session_state:
        st.session_state.show_thalach_info = False
        
    with st.sidebar:
        col1_thalach, col2_thalach = st.columns([0.8, 0.2])

        with col1_thalach:
            thalach = st.slider("Maximum HR (thalach)", 71, 202, 150, key = 'thalach_slider') # Default disesuaikan dengan range umum

        with col2_thalach:
            st.write("")
            st.write("")
            if st.button('❔', key='thalach_info_button'):
                st.session_state.show_thalach_info = not st.session_state.show_thalach_info

        if st.session_state.show_thalach_info:
            with st.expander("Detail Maximum HR (Thalach)", expanded=True):
                st.write("""
                **Detak Jantung Maksimal yang Dicapai (Thalach):** \n 
                Ini adalah detak jantung tertinggi yang dicatat selama tes stres.\n
                Detak jantung maksimum yang lebih tinggi selama berolahraga umumnya menunjukkan kebugaran kardiovaskular yang lebih baik. \n
                Namun, dalam konteks prediksi penyakit jantung, nilai tersebut dinilai bersamaan dengan faktor-faktor lain. \n 
                Detak jantung maksimum normal bervariasi berdasarkan usia dan tingkat kebugaran individu.
                """)

    # --- Sisa Input UI ---
    # Sesuaikan range slider berdasarkan df.describe() atau pengetahuan domain
    # Slope: range 0-2
    slope = st.sidebar.slider("Slope Segment ST on EKG (slope)", 0, 2, 1) 
    # Oldpeak: range 0.0-6.2 (terlihat dari df.describe() aslimu)
    oldpeak = st.sidebar.slider("Depression Segment ST when Peak Activity (oldpeak)", 0.0, 6.2, 1.0, step=0.1) 
    # Exang: 0 atau 1
    exang = st.sidebar.selectbox("Exercise-Induced Angina (exang)", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya") 
    # Ca: 0-3
    ca = st.sidebar.slider("Number of Major Vessels (ca)", 0, 3, 0) # Default 0 agar sesuai dengan nilai umum
    # Thal: 0-3 (sesuaikan jika ada 0, atau 1-3 jika 0 tidak valid)
    # Dari data aslimu, thal memiliki nilai 0, 2, 3. Sesuaikan opsi.
    thal = st.sidebar.selectbox("Thallium Stress Test (thal)", [0, 1, 2, 3], index=2, # Default ke 2 (fixed defect)
                                format_func=lambda x: {0: "(Unknown/Invalid)", 1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}.get(x, str(x)))
    
    # Sex: 0 atau 1
    sex_option = st.sidebar.selectbox("Sex", ('Female', 'Male'))
    sex = 0 if sex_option == "Female" else 1 
    
    # Age: range 29-77 (dari df.describe() aslimu)
    age = st.sidebar.slider("Age", 29, 77, 50) # Default 50 sebagai nilai tengah

    # Tambahkan input untuk kolom lain yang mungkin ada di dataset aslimu dan digunakan model
    # (misal: trestbps, chol, fbs, restecg) jika kamu menambahkannya di model training script
    # Saya akan mengasumsikan kamu hanya menggunakan fitur yang terlihat di df.head() untuk kesederhanaan.

    data = {'sex': sex,
            'age': age,
            'cp': cp,
            'thalach': thalach,
            'slope': slope,
            'exang': exang,
            'ca': ca,
            'thal': thal,
            'oldpeak': oldpeak}
    
    # Buat DataFrame dari input pengguna
    features = pd.DataFrame([data])
    return features

# --- Fungsi utama aplikasi Streamlit ---
def heart_prediction_app():
    st.write("""
    This app predicts the **Heart Disease**
    
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    """)
    st.sidebar.header('List Data Input:')

    uploaded_file = st.sidebar.file_uploader("Upload your data here (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.sidebar.write("Data CSV berhasil diunggah.")
    else:
        input_df = user_input_features()
        st.sidebar.write("Menggunakan input manual.")

    img = Image.open("heart-disease.jpg")
    st.image(img, width=500)

    if st.sidebar.button('Predict!'):
        st.subheader("Data Input Sebelum Pra-pemrosesan:")
        st.write(input_df)

        # --- Pra-pemrosesan Data Input (HARUS KONSISTEN DENGAN TRAINING) ---
        # 1. Identifikasi kolom numerik dan kategorikal dari input_df
        # Sesuai dengan yang kamu identifikasi di skrip pelatihan
        NUMERICAL_COLS_APP = ['age', 'thalach', 'oldpeak'] 
        CATEGORICAL_COLS_APP = ['sex', 'cp', 'exang', 'slope', 'ca', 'thal'] # Kolom yang akan di-one-hot encode

        # Filter agar hanya kolom yang benar-benar ada di input_df yang masuk list
        NUMERICAL_COLS_APP = [col for col in NUMERICAL_COLS_APP if col in input_df.columns]
        CATEGORICAL_COLS_APP = [col for col in CATEGORICAL_COLS_APP if col in input_df.columns]

        # Buat salinan input_df untuk diolah
        processed_input_df = input_df.copy()

        # 2. Lakukan One-Hot Encoding pada kolom kategorikal
        # Perhatikan: get_dummies akan membuat kolom baru seperti 'sex_1', 'cp_1', 'cp_2', dll.
        # Jika ada nilai kategorikal yang tidak muncul di data training (misal cp=4 di training tapi input cp=5),
        # maka one-hot encoding akan menghasilkan kolom yang berbeda.
        # Ini perlu ditangani dengan membuat semua kolom dummy yang mungkin ada di training.
        
        # Buat semua kolom dummy yang mungkin, berdasarkan expected_cols yang dimuat
        # Ambil hanya nama kolom dummy dari expected_cols
        dummy_cols_from_expected = [col for col in expected_cols if '_' in col]
        
        # Lakukan one-hot encoding pada input_df
        # Jangan pakai drop_first=True di sini karena kita ingin semua kolom dummy yang potensial
        # akan diisi 0 jika nilainya tidak ada.
        processed_input_df = pd.get_dummies(processed_input_df, columns=CATEGORICAL_COLS_APP, dtype=int)

        # Tambahkan kolom dummy yang mungkin hilang (jika input tidak mencakup semua kategori)
        # dan pastikan urutan kolom sesuai dengan expected_cols
        final_input_for_model = pd.DataFrame(columns=expected_cols)

        for col in expected_cols:
            if col in processed_input_df.columns:
                final_input_for_model[col] = processed_input_df[col]
            else:
                # Jika kolom tidak ada di input_df (misal 'sex_0' jika hanya 'sex_1' yg ada)
                # atau jika kategori tertentu tidak ada di input saat ini
                final_input_for_model[col] = 0 # Isi dengan 0 (karena ini one-hot encoding)
        
        # Pastikan indeks cocok sebelum operasi scaling dan finalisasi
        final_input_for_model = final_input_for_model.fillna(0) # Mengisi NaN yang mungkin muncul dari penggabungan
        
        st.write("### Data Input Setelah One-Hot Encoding dan Penyesuaian Kolom:")
        st.write(final_input_for_model.head())
        st.write(f"Jumlah kolom: {final_input_for_model.shape[1]}, Expected: {len(expected_cols)}")


        # 3. Scaling Fitur Numerik
        # Buat salinan df yang akan diskala
        df_scaled = final_input_for_model.copy() 
        
        # Pastikan hanya kolom numerik yang ada di daftar NUMERICAL_COLS_APP yang diskala
        # dan kolom tersebut ada di df_scaled
        cols_to_scale_exist = [col for col in NUMERICAL_COLS_APP if col in df_scaled.columns]

        if cols_to_scale_exist:
            # Menggunakan scaler yang sudah dimuat
            df_scaled[cols_to_scale_exist] = scaler.transform(df_scaled[cols_to_scale_exist])
            st.write("### Data Input Setelah Scaling Fitur Numerik:")
            st.write(df_scaled.head())
        else:
            st.warning("Tidak ada kolom numerik yang ditemukan untuk scaling berdasarkan daftar NUMERICAL_COLS_APP.")
            df_scaled = final_input_for_model # Gunakan yang belum diskala jika tidak ada kolom numerik
            
        
        # Prediksi
        with st.spinner('Melakukan Prediksi...'):
            time.sleep(1) # Jeda singkat untuk UX
            prediction_proba = loaded_model.predict_proba(df_scaled)
            # Karena ini klasifikasi biner, prediction_proba akan memiliki 2 kolom.
            # Kolom [0] adalah probabilitas kelas 0 (Tidak Penyakit)
            # Kolom [1] adalah probabilitas kelas 1 (Penyakit)
            
            # Jika target positif adalah 1, ambil probabilitas di kolom 1
            score = prediction_proba[0][1] 
            prediction = loaded_model.predict(df_scaled)

        # Tampilkan hasil prediksi
        st.subheader('Hasil Prediksi:')
        if prediction[0] == 0:
            st.markdown(
                f"<h2 style='color: green;'>NEGATIF HEART DISEASE</h2>"
                f"<h4 style='color: gray;'>Tingkat Kepercayaan Model (Tidak Penyakit): {prediction_proba[0][0]:.2f}</h4>"
                f"<h4 style='color: gray;'>Tingkat Kepercayaan Model (Penyakit): {score:.2f}</h4>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='color: red;'>POSITIF HEART DISEASE</h2>"
                f"<h4 style='color: gray;'>Tingkat Kepercayaan Model (Tidak Penyakit): {prediction_proba[0][0]:.2f}</h4>"
                f"<h4 style='color: gray;'>Tingkat Kepercayaan Model (Penyakit): {score:.2f}</h4>",
                unsafe_allow_html=True
            )
        
        # Tambahkan visualisasi untuk debugging
        st.write("---")
        st.write("Debugging Info:")
        st.write(f"Kolom yang diharapkan model (dari 'features.pkl'): {expected_cols}")
        st.write(f"Kolom setelah Pra-pemrosesan di Streamlit: {df_scaled.columns.tolist()}")
        st.write(f"Jumlah kolom: {df_scaled.shape[1]} (Expected: {len(expected_cols)})")

# Panggil fungsi heart_prediction_app untuk menjalankan aplikasi
heart_prediction_app()
