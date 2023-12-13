import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def label_encode_data(data, categorical_features):
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        if feature in data.columns:
            data[feature] = label_encoder.fit_transform(data[feature])
    return data

with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "PreProcessing Data", "Modelling", "Klasifikasi"],
        icons=['house', 'table', 'boxes','check2-circle'],
        menu_icon="cast",
        default_index=1,
        orientation='vertical')

with st.sidebar:
    upload_file = st.sidebar.file_uploader("Masukkan file csv disini", key=1)

if selected == 'Home':
    st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Kelayakan Keluarga Penerima Bantuan Langsung Tunai Dana Desa </h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: left;"> Bantuan Langsung Tunai Dana Desa (BLT-DD) </h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: left;"> View Data </h1>', unsafe_allow_html=True)
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.write("Dataset yang digunakan untuk seleksi kelayakan penerima BLT-DD adalah data yang didapatkan dari Balai Desa Bandung Kecamatan Konang Kabupaten Bangkalan pada bulan April Tahun 2020 dan berjumlah sebanyak 623 data. Dataset tersebut memiliki 2 kelas yaitu memenuhi syarat sebagai penerima BLT-DD dan belum memenuhi syarat sebagai penerima BLT-DD dengan beberapa kriteria yang digunakan.")
        st.dataframe(df)

elif selected == 'PreProcessing Data':
    st.markdown('<h3 style="text-align: left;"> Data Asli </h1>', unsafe_allow_html=True)
    st.write("Berikut merupakan data asli yang didapat dari Balai Desa Bandung Kecamatan Konang Kabupaten Bangkalan.")
    
    encoded_data = pd.DataFrame()  # Define encoded_data outside the 'One-Hot Encoding' block
    
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.dataframe(df)
        st.markdown('<h3 style="text-align: left;"> Melakukan Transformation Data </h1>', unsafe_allow_html=True)
        # Specify the categorical features for one-hot encoding
        categorical_features = ['jenis kelamin', 'penerima jps', 'belum menerima jps', 'target']
        # One-hot encoding
        if st.button("Label Encoding"):
            encoded_data = label_encode_data(st.session_state.cleaned_data, categorical_features)
            st.write("Label encoding completed.")
            st.dataframe(encoded_data)
        st.markdown('<h3 style="text-align: left;"> Melakukan Normalisasi Data </h1>', unsafe_allow_html=True)
        # Min-Max scaling for all features
                    
elif selected == 'Modelling':
    st.write("You are at Klasifikasi Datamining")

elif selected == 'Evaluasi':
    st.write("You are at Uji Coba")
