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
# Bagian ini bertujuan untuk memuat model Machine Learning, scaler (untuk normalisasi data),
# dan daftar nama fitur yang diharapkan oleh model.
# Ini penting agar objek-objek ini hanya dimuat sekali saat aplikasi dimulai,
# sehingga tidak perlu memuat ulang setiap kali tombol 'Predict!' ditekan,
# yang akan meningkatkan kinerja aplikasi.
try:
    # Ganti nama file sesuai dengan yang kamu simpan
    # Memuat objek model yang sudah dilatih dari file 'generate_heart_disease.pkl'.
    loaded_model = joblib.load('generate_heart_disease.pkl')
    # Memuat objek scaler (StandardScaler) yang sudah dilatih dari file 'scaler.pkl'.
    # Perhatikan: sebelumnya nama file mungkin 'scaler_jantung_fix.pkl', tapi sudah diubah menjadi 'scaler.pkl' di sini.
    scaler = joblib.load('scaler.pkl') 
    # Memuat daftar nama kolom fitur yang diharapkan oleh model dari file 'features.pkl'.
    # Ini sangat penting untuk memastikan urutan dan keberadaan kolom yang konsisten saat prediksi.
    # Perhatikan: sebelumnya nama file mungkin 'feature_names_for_model.pkl', tapi sudah diubah menjadi 'features.pkl' di sini.
    expected_cols = joblib.load('features.pkl') 
    st.success("Model, Scaler, dan List Features Successfully Loaded!")
except FileNotFoundError:
    # Jika salah satu file .pkl tidak ditemukan, tampilkan pesan error dan hentikan aplikasi.
    st.error("Error: Pastikan 'generate_heart_disease.pkl', 'scaler.pkl', dan 'features.pkl' ada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika file tidak ditemukan
except Exception as e:
    # Tangani error lain yang mungkin terjadi saat memuat file dan tampilkan pesan error.
    st.error(f"Error when loading file: {e}")
    st.stop()

# --- Definisi fungsi user_input_features() ---
# Fungsi ini bertanggung jawab untuk membuat antarmuka pengguna (UI) di sidebar Streamlit
# agar pengguna dapat memasukkan nilai fitur secara manual.
def user_input_features():
    st.sidebar.header('Manual Input')

    # Perhatikan: Sesuaikan pilihan nilai selectbox cp agar sesuai dengan dataset asli (0,1,2,3)
    # Pilihan numerik untuk jenis nyeri dada. Ini harus sesuai dengan nilai yang ada di data training.
    # Dari head dataset, terlihat cp menggunakan 0, 1, 2, 3.
    cp_options_numeric = [0, 1, 2, 3] # Menggunakan 0, 1, 2, 3 sesuai head dataset
    def format_cp_display(option):
        # Fungsi pembantu untuk menampilkan teks yang lebih deskriptif
        # di selectbox berdasarkan nilai numerik 'cp'.
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
    # Menginisialisasi state sesi untuk mengontrol visibilitas informasi detail cp.
    # Jika 'show_cp_info' belum ada di session_state, setel ke False.
    if 'show_cp_info' not in st.session_state:
        st.session_state.show_cp_info = False
    
    with st.sidebar:
        # Menggunakan kolom untuk menempatkan selectbox dan tombol '❔' secara berdampingan di sidebar.
        col1_cp, col2_cp = st.columns([0.8, 0.2])

        with col1_cp:
            # Membuat selectbox untuk input 'Chest Pain (cp)'.
            cp = st.selectbox(
                'Chest Pain (cp)',
                options=cp_options_numeric,
                format_func=format_cp_display,
                index=0, # Default ke Typical Angina (0)
                key='chest_pain_selectbox' # Memberikan kunci unik untuk selectbox.
            )

        with col2_cp:
            # Baris kosong untuk penataan agar tombol sejajar dengan selectbox.
            st.write("") 
            st.write("") 
            # Tombol '❔' untuk menampilkan/menyembunyikan detail informasi 'cp'.
            if st.button('❔', key='cp_info_button'):
                st.session_state.show_cp_info = not st.session_state.show_cp_info # Mengganti status visibilitas.

        if st.session_state.show_cp_info:
            # Menampilkan detail 'Chest Pain (CP)' dalam expander jika 'show_cp_info' True.
            with st.expander("Details of Chest Pain (CP)", expanded=True):
                st.write("""
                **0 - Typical Angina:** Chest pain that is characteristic, usually associated with physical activity and relieved by rest or nitroglycerin. \n
                **1 - Atypical Angina:** Chest pain that is less characteristic. \n
                **2 - Non-Anginal Pain:** Chest pain that does not originate from heart problems. \n
                **3 - Asymptomatic:** No chest pain reported. \n """)
    
    # UI untuk Maximum HR (thalach)
    # Menginisialisasi state sesi untuk mengontrol visibilitas informasi detail thalach.
    if 'show_thalach_info' not in st.session_state:
        st.session_state.show_thalach_info = False
        
    with st.sidebar:
        # Menggunakan kolom untuk menempatkan slider dan tombol '❔' secara berdampingan di sidebar.
        col1_thalach, col2_thalach = st.columns([0.8, 0.2])

        with col1_thalach:
            # Membuat slider untuk input 'Maximum HR (thalach)'.
            # Rentang (71, 202) dan nilai default (150) disesuaikan dengan data umum.
            thalach = st.slider("Maximum HR (thalach)", 71, 202, 150, key = 'thalach_slider') 

        with col2_thalach:
            # Baris kosong untuk penataan.
            st.write("")
            st.write("")
            # Tombol '❔' untuk menampilkan/menyembunyikan detail informasi 'thalach'.
            if st.button('❔', key='thalach_info_button'):
                st.session_state.show_thalach_info = not st.session_state.show_thalach_info

        if st.session_state.show_thalach_info:
            # Menampilkan detail 'Maximum HR (Thalach)' dalam expander jika 'show_thalach_info' True.
            with st.expander("Details of Maximum HR (Thalach)", expanded=True):
                st.write("""
                **Maximum Heart Rate Achieved (Thalach):** \n 
                This is the highest heart rate recorded during a stress test. \n
                A higher maximum heart rate during exercise generally indicates better cardiovascular fitness. \n
                However, in the context of heart disease prediction, the value is assessed alongside other factors. \n 
                Normal maximum heart rates vary by age and individual fitness levels.
                """)

    # --- Sisa Input UI ---
    # Sesuaikan range slider berdasarkan df.describe() atau pengetahuan domain
    # Membuat slider untuk input 'Slope Segment ST on EKG (slope)'. Rentang (0, 2).
    slope = st.sidebar.slider("Slope Segment ST on EKG (slope)", 0, 2, 1) 
    # Membuat slider untuk input 'ST Depression Induced by Exercise Relative to Rest (oldpeak)'.
    # Rentang (0.0, 6.2) dan langkah (step=0.1) disesuaikan dengan data asli.
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest (oldpeak)", 0.0, 6.2, 1.0, step=0.1) 
    # Membuat selectbox untuk input 'Exercise-Induced Angina (exang)'.
    # Opsi (0, 1) dengan format tampilan 'No' atau 'Yes'.
    exang = st.sidebar.selectbox("Exercise-Induced Angina (exang)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes") 
    # Membuat slider untuk input 'Number of Major Vessels (ca)'. Rentang (0, 3).
    ca = st.sidebar.slider("Number of Major Vessels (ca)", 0, 3, 0) # Default 0 agar sesuai dengan nilai umum
    # Membuat selectbox untuk input 'Thallium Stress Test (thal)'.
    # Opsi (0, 1, 2, 3) dengan format tampilan yang deskriptif.
    # Dari data aslimu, thal memiliki nilai 0, 2, 3. Sesuaikan opsi.
    thal = st.sidebar.selectbox("Thallium Stress Test (thal)", [0, 1, 2, 3], index=2, # Default ke 2 (fixed defect)
                                 format_func=lambda x: {0: "(Unknown/Invalid)", 1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}.get(x, str(x)))
    
    # Membuat selectbox untuk input 'Sex'. Opsi ('Female', 'Male').
    sex_option = st.sidebar.selectbox("Sex", ('Female', 'Male'))
    # Mengubah input teks 'Sex' menjadi nilai numerik (0 untuk Female, 1 untuk Male)
    # agar sesuai dengan format yang diharapkan oleh model.
    sex = 0 if sex_option == "Female" else 1 
    
    # Membuat slider untuk input 'Age'. Rentang (29, 77) dan nilai default (50) disesuaikan.
    age = st.sidebar.slider("Age", 29, 77, 50) # Default 50 sebagai nilai tengah

    # Tambahkan input untuk kolom lain yang mungkin ada di dataset aslimu dan digunakan model
    # (misal: trestbps, chol, fbs, restecg) jika kamu menambahkannya di model training script
    # Baris ini adalah catatan untuk pengembang agar menambah fitur lain jika diperlukan.
    # Saya akan mengasumsikan kamu hanya menggunakan fitur yang terlihat di df.head() untuk kesederhanaan.
    # Asumsi ini menyatakan bahwa kode saat ini hanya menangani fitur yang sudah terlihat di awal diskusi.

    # Membuat kamus (dictionary) dari semua input fitur yang telah dikumpulkan.
    data = {'sex': sex,
            'age': age,
            'cp': cp,
            'thalach': thalach,
            'slope': slope,
            'exang': exang,
            'ca': ca,
            'thal': thal,
            'oldpeak': oldpeak}
    
    # Mengonversi kamus 'data' menjadi objek Pandas DataFrame.
    # 'index=[0]' menunjukkan bahwa ini adalah satu baris data.
    features = pd.DataFrame([data])
    return features

# --- Fungsi utama aplikasi Streamlit ---
# Fungsi ini mengelola alur utama aplikasi prediksi penyakit jantung.
def heart_prediction_app():
    st.write("""
    This app predicts the **Heart Disease** status based on various physiological parameters.
    
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    """)
    st.sidebar.header('Data Input:')

    # Widget untuk mengunggah file CSV di sidebar.
    uploaded_file = st.sidebar.file_uploader("Upload your data here (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Jika file CSV diunggah, baca file tersebut ke dalam DataFrame.
        input_df = pd.read_csv(uploaded_file)
        st.sidebar.write("CSV data uploaded successfully.")
    else:
        # Jika tidak ada file diunggah, gunakan input manual dari fungsi user_input_features().
        input_df = user_input_features()
        st.sidebar.write("Using manual input.")

    # Memuat dan menampilkan gambar 'heart-disease.jpg'.
    img = Image.open("heart-disease.jpg")
    st.image(img, width=500)

    # Ketika tombol 'Predict!' di sidebar ditekan, proses prediksi akan dimulai.
    if st.sidebar.button('Predict!'):
        st.subheader("Input Data Before Preprocessing:")
        st.write(input_df)

        # --- Data Preprocessing (MUST BE CONSISTENT WITH TRAINING) ---
        # Bagian ini sangat krusial untuk memastikan data input diproses dengan cara yang sama
        # seperti data yang digunakan saat model dilatih.
        # 1. Identifikasi kolom numerik dan kategorikal dari input_df
        # Daftar kolom numerik yang akan diskalakan. Ini harus sama dengan skrip pelatihan.
        NUMERICAL_COLS_APP = ['age', 'thalach', 'oldpeak'] 
        
        # Daftar kolom kategorikal yang akan di-one-hot encode. Ini juga harus sama.
        CATEGORICAL_COLS_APP = ['sex', 'cp', 'exang', 'slope', 'ca', 'thal'] 

        # Filter agar hanya kolom yang benar-benar ada di input_df yang masuk list
        # Memastikan bahwa hanya kolom yang benar-benar ada di DataFrame input saat ini
        # yang dimasukkan ke dalam daftar kolom numerik/kategorikal untuk diproses.
        NUMERICAL_COLS_APP = [col for col in NUMERICAL_COLS_APP if col in input_df.columns]
        CATEGORICAL_COLS_APP = [col for col in CATEGORICAL_COLS_APP if col in input_df.columns]

        # Buat salinan input_df untuk diolah
        # Membuat salinan DataFrame input untuk menghindari modifikasi pada DataFrame asli.
        processed_input_df = input_df.copy()

        # 2. Lakukan One-Hot Encoding pada kolom kategorikal
        # Perhatikan: get_dummies akan membuat kolom baru seperti 'sex_1', 'cp_1', 'cp_2', dll.
        # Jika ada nilai kategorikal yang tidak muncul di data training (misal cp=4 di training tapi input cp=5),
        # maka one-hot encoding akan menghasilkan kolom yang berbeda.
        # Ini perlu ditangani dengan membuat semua kolom dummy yang mungkin ada di training.
        
        # Buat semua kolom dummy yang mungkin, berdasarkan expected_cols yang dimuat
        # (Baris ini sekarang menjadi catatan karena logika pengisian kolom di bawah sudah mencakup ini.)
        # Extract only the dummy column names from expected_cols
        # No need for this, as we iterate through all expected_cols below.
        
        # Perform one-hot encoding on processed_input_df
        # Melakukan One-Hot Encoding pada kolom kategorikal yang telah diidentifikasi.
        # 'dtype=int' memastikan kolom baru berisi 0 atau 1 (integer).
        # Penting untuk TIDAK menggunakan 'drop_first=True' di sini, agar semua kolom dummy yang mungkin ada
        # (misalnya, cp_0, cp_1, cp_2, cp_3) akan dibuat. Kita akan mengelola keberadaan kolom-kolom ini
        # agar sesuai dengan 'expected_cols' di langkah berikutnya.
        processed_input_df = pd.get_dummies(processed_input_df, columns=CATEGORICAL_COLS_APP, dtype=int)

        # Initialize final_input_for_model with all expected columns, filled with zeros
        # Membuat DataFrame baru bernama 'final_input_for_model' yang berisi SEMUA kolom
        # yang diharapkan oleh model ('expected_cols').
        # Semua nilai diinisialisasi dengan 0. Ini memastikan bahwa jika ada kolom dummy
        # yang tidak dihasilkan dari input saat ini (karena kategorinya tidak ada),
        # kolom tersebut tetap ada dan bernilai 0, seperti yang diharapkan model.
        final_input_for_model = pd.DataFrame(0, index=processed_input_df.index, columns=expected_cols)

        # Populate final_input_for_model with values from processed_input_df
        # Mengisi 'final_input_for_model' dengan nilai-nilai dari 'processed_input_df'
        # untuk kolom-kolom yang ada di keduanya.
        # Ini menyalin nilai-nilai dari fitur asli dan fitur dummy yang baru dibuat.
        for col in expected_cols:
            if col in processed_input_df.columns:
                final_input_for_model[col] = processed_input_df[col]
        
        st.write("### Input Data After One-Hot Encoding and Column Alignment:")
        st.write(final_input_for_model.head())
        st.write(f"Number of columns: {final_input_for_model.shape[1]}, Expected: {len(expected_cols)}")


        # 3. Scale Numerical Features
        # Membuat salinan dari DataFrame yang sudah di-encode dan disesuaikan kolomnya.
        df_scaled = final_input_for_model.copy() 
        
        # Pastikan hanya kolom numerik yang ada di daftar NUMERICAL_COLS_APP yang diskala
        # dan kolom tersebut ada di df_scaled
        # Memfilter daftar kolom numerik untuk memastikan hanya kolom yang benar-benar ada
        # di DataFrame saat ini yang akan diskalakan.
        cols_to_scale_exist = [col for col in NUMERICAL_COLS_APP if col in df_scaled.columns]

        if cols_to_scale_exist:
            # Menggunakan scaler yang sudah dimuat
            # Menerapkan transformasi scaling pada kolom numerik yang teridentifikasi
            # menggunakan 'scaler' yang sudah dilatih dari fase training.
            # INI PENTING: HANYA TRANSFORM, BUKAN FIT_TRANSFORM.
            df_scaled[cols_to_scale_exist] = scaler.transform(df_scaled[cols_to_scale_exist])
            st.write("### Input Data After Numerical Feature Scaling:")
            st.write(df_scaled.head())
        else:
            st.warning("No numerical columns found for scaling based on NUMERICAL_COLS_APP list.")
            df_scaled = final_input_for_model # Gunakan yang belum diskala jika tidak ada kolom numerik
            
        
        # Prediksi
        # Menampilkan indikator 'spinner' saat prediksi sedang berjalan untuk pengalaman pengguna yang lebih baik.
        with st.spinner('Performing Prediction...'):
            time.sleep(1) # Jeda singkat untuk UX (User Experience).
            # Melakukan prediksi probabilitas menggunakan model yang sudah dimuat.
            # Ini akan menghasilkan probabilitas untuk setiap kelas (0 dan 1).
            prediction_proba = loaded_model.predict_proba(df_scaled)
            # Karena ini klasifikasi biner, prediction_proba akan memiliki 2 kolom.
            # Kolom [0] adalah probabilitas kelas 0 (Tidak Penyakit)
            # Kolom [1] adalah probabilitas kelas 1 (Penyakit)
            
            # Jika target positif adalah 1, ambil probabilitas di kolom 1
            # Mengambil probabilitas untuk kelas positif (penyakit jantung, yaitu kelas 1).
            score = prediction_proba[0][1] 
            # Melakukan prediksi kelas (0 atau 1) menggunakan model.
            prediction = loaded_model.predict(df_scaled)

        # Tampilkan hasil prediksi
        st.subheader('Prediction Result:')
        # Memeriksa hasil prediksi dan menampilkan pesan yang sesuai.
        if prediction[0] == 0:
            # Jika prediksi adalah 0 (Tidak Penyakit), tampilkan teks hijau.
            st.markdown(
                f"<h2 style='color: green;'>NEGATIVE FOR HEART DISEASE</h2>"
                f"<h4 style='color: gray;'>Model Confidence (Disease): {score:.2f}</h4>",
                unsafe_allow_html=True
            )
        else:
            # Jika prediksi adalah 1 (Penyakit), tampilkan teks merah.
            st.markdown(
                f"<h2 style='color: red;'>POSITIVE FOR HEART DISEASE</h2>"
                f"<h4 style='color: gray;'>Model Confidence (Disease): {score:.2f}</h4>",
                unsafe_allow_html=True
            )
        
        # Tambahkan visualisasi untuk debugging
        st.write("---")
        st.write("Debugging Info:")
        # Menampilkan daftar kolom yang diharapkan oleh model (dari 'features.pkl') untuk debugging.
        st.write(f"Expected columns for the model (from 'features.pkl'): {expected_cols}")
        # Menampilkan daftar kolom yang sebenarnya digunakan untuk prediksi setelah pra-pemrosesan di Streamlit.
        st.write(f"Columns after preprocessing in Streamlit: {df_scaled.columns.tolist()}")
        # Menampilkan jumlah kolom yang diproses dibandingkan dengan jumlah kolom yang diharapkan.
        st.write(f"Number of columns: {df_scaled.shape[1]} (Expected: {len(expected_cols)})")

# Panggil fungsi heart_prediction_app untuk menjalankan aplikasi
heart_prediction_app()
