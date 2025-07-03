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
# Heart Disease CVD Model Prediction
""")

# --- Definisi fungsi user_input_features() ---
def user_input_features():
    st.sidebar.header('Manual Input')
    cp = st.sidebar.slider('Chest Pain Type', 1, 4, 2) 
    
    wcp_text = ""
    if cp == 1:
        wcp_text = "Chest pain type angina (Tipikal Angina)"
    elif cp == 2:
        wcp_text = "Chest pain type unstable (Atypical Angina)"
    elif cp == 3:
        wcp_text = "Chest pain type unstable and severe (Non-Anginal Pain)"
    elif cp == 4:
        wcp_text = "Chest pain type non heart disease (Asymptomatic)"
    st.sidebar.write("The type of chest pain felt by the patient:", wcp_text)
    
    thalach = st.sidebar.slider("Maximum HR Achived", 71, 202, 80)
    slope = st.sidebar.slider("Slope Segment ST on Elektrokardiogram (EKG)", 0, 2, 1)
    oldpeak = st.sidebar.slider("Depression Segment ST when Peak Activity", 0.0, 6.2, 1.0)
    exang = st.sidebar.slider("Exercise-Induced Angina", 0, 1, 1)
    ca = st.sidebar.slider("Number of Major Vessels", 0, 3, 1)
    thal = st.sidebar.slider("Thallium Stress Test", 1, 3, 1)
    
    sex_option = st.sidebar.selectbox("Sex", ('Female', 'Male'))
    sex = 0 if sex_option == "Female" else 1 
    
    age = st.sidebar.slider("Age", 29, 77, 30)
    
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
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
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
            time.sleep(4)
            if prediction == 0:
                # Jika NO HEART DISEASE, tampilkan dengan warna hijau
                st.markdown(f"<h2 style='color: green;'>NO HEART DISEASE</h2>", unsafe_allow_html=True)
            else:
                # Jika POSITIF HEART DISEASE, tampilkan dengan warna merah
                st.markdown(f"<h2 style='color: red;'>POSITIF HEART DISEASE</h2>", unsafe_allow_html=True)
# Panggil fungsi heart untuk menjalankan aplikasi
heart()
