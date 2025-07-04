import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image
# Tambahkan import StandardScaler jika modelmu menggunakan itu

st.set_page_config(
    page_title="CVD_MODEL_PREDICTION",
    page_icon = "🫀",
    layout = 'wide'
)

st.write("""
# Heart CVD Model Prediction
""")

# --- Definisi fungsi user_input_features() ---
def user_input_features():
    st.sidebar.header('Manual Input')
    cp_options_numeric = [1, 2, 3, 4]
    def format_cp_display(option):
        if option == 1:
            return "1 - Typical Angina"
        elif option == 2:
            return "2 - Atypical Angina"
        elif option == 3:
            return "3 - Non-Anginal Pain"
        elif option == 4:
            return "4 - Asymptomatic"
        return str(option)
    cp = st.sidebar.selectbox(
        'Chest Pain (cp)',
        options=cp_options_numeric,
        format_func=format_cp_display,
        index=1)
    
    thalach = st.sidebar.slider("Maximum HR (thalach)", 71, 202, 80)
    slope = st.sidebar.slider("Slope Segment ST on EKG (slope)", 0, 2, 1)
    oldpeak = st.sidebar.slider("Depression Segment ST when Peak Activity (peak)", 0.0, 6.2, 1.0)
    exang = st.sidebar.slider("Exercise-Induced Angina (exang)", 0, 1, 1)
    ca = st.sidebar.slider("Number of Major Vessels (ca)", 0, 3, 1)
    thal = st.sidebar.slider("Thallium Stress Test (thal)", 1, 3, 1)
    
    sex_option = st.sidebar.selectbox("Sex", ('Female', 'Male'))
    sex = 0 if sex_option == "Female" else 1 
    
    age = st.sidebar.slider("Age", 29, 77, 29)
    
    data = {'cp': cp,
            'thalach': thalach,
            'slope': slope,
            'oldpeak': oldpeak,
            'exang': exang,
            'ca': ca,
            'thal': thal,
            'sex': sex,
            'age': age}
    features = pd.DataFrame(data, index=[0])
    return features

def heart():
    st.write("""
    This app predicts the **Heart Disease**
    
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    """)
    st.sidebar.header('List Data Input by Document:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your data here (CSV)", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        # Panggil fungsi user_input_features yang sudah didefinisikan di luar
        input_df = user_input_features() # Menggunakan fungsi user_input_features yang pertama
    
    img = Image.open("heart-disease.jpg")
    st.image(img, width=500)
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        with open("generate_heart_disease.pkl", 'rb') as file: 
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)        
        
        st.subheader('Prediction: ')
        with st.spinner('Wait for it...'):
            time.sleep(3)
            if prediction == 0:
                # Jika NO HEART DISEASE, tampilkan dengan warna hijau
                st.markdown(f"<h2 style='color: green;'>NEGATIF HEART DISEASE</h2>", unsafe_allow_html=True)
            else:
                # Jika POSITIF HEART DISEASE, tampilkan dengan warna merah
                st.markdown(f"<h2 style='color: red;'>POSITIF HEART DISEASE</h2>", unsafe_allow_html=True)
# Panggil fungsi heart untuk menjalankan aplikasi
heart()
