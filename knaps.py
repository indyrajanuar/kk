import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import keras
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(data): 
    # Replace commas with dots and convert numerical columns to floats
    numerical_columns = ['IMT']
    data[numerical_columns] = data[numerical_columns].replace({',': '.'}, regex=True).astype(float)
    # One-hot encoding for 'Jenis Kelamin'
    one_hot_encoder = OneHotEncoder()
    encoded_gender = one_hot_encoder.fit_transform(data[['Jenis Kelamin']].values.reshape(-1, 1))
    encoded_gender = pd.DataFrame(encoded_gender.toarray(), columns=one_hot_encoder.get_feature_names_out(['Jenis Kelamin']))    
    # Transform 'Diagnosa' feature to binary values
    data['Diagnosa'] = data['Diagnosa'].map({'YA': 1, 'TIDAK': 0})
    # Drop the original 'Jenis Kelamin' feature
    data = data.drop('Jenis Kelamin', axis=1)    
    # Concatenate encoded 'Jenis Kelamin' and transformed 'Diagnosa' with original data
    data = pd.concat([data, encoded_gender], axis=1)
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return normalized_data

def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "PreProcessing Data", "Klasifikasi ERNN", "ERNN + Bagging", "Uji Coba"],
            icons=['house', 'table', 'boxes', 'boxes', 'check2-circle'],
            menu_icon="cast",
            default_index=1,
            orientation='vertical')
    
        upload_file = st.sidebar.file_uploader("Masukkan file csv disini", key=1)
    
    if selected == 'Home':
        st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: left;"> Hipertensi </h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: left;"> View Data </h1>', unsafe_allow_html=True)
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.write("Data yang digunakan yaitu data Penyakit Hipertensi dari UPT Puskesmas Modopuro Mojokerto.")
            st.dataframe(df)
    
    elif selected == 'PreProcessing Data':
        st.markdown('<h3 style="text-align: left;"> Data Asli </h1>', unsafe_allow_html=True)
        st.write("Berikut merupakan data asli yang didapat dari UPT Puskesmas Modopuro Mojokerto.")
    
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.dataframe(df)
            st.markdown('<h3 style="text-align: left;"> Melakukan Transformation Data </h1>', unsafe_allow_html=True)
            if st.button("Transformation Data"):  # Check if button is clicked
                preprocessed_data = preprocess_data(df)
                st.write("Transformation completed.")
                st.dataframe(preprocessed_data)
                st.session_state.preprocessed_data = preprocessed_data  # Store preprocessed data in session state
    
            st.markdown('<h3 style="text-align: left;"> Melakukan Normalisasi Data </h1>', unsafe_allow_html=True)
            if 'preprocessed_data' in st.session_state:  # Check if preprocessed_data exists in session state
                if st.button("Normalize Data"):
                    normalized_data = normalize_data(st.session_state.preprocessed_data.copy())
                    st.write("Normalization completed.")
                    st.dataframe(normalized_data)
    
    elif selected == 'Klasifikasi ERNN':
        st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan Elman Recurrent Neural Network (ERNN)")
    
    elif selected == 'ERNN + Bagging':
        st.write("You are at Klasifikasi ERNN + Bagging")
        
    elif selected == 'Uji Coba':
        st.title("Uji Coba")

if __name__ == "__main__":
    main()
