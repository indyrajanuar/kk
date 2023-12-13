import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

def label_encode_data(data, categorical_features):
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        if feature in data.columns:
            data[feature] = label_encoder.fit_transform(data[feature])
    return data

# Define session state
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = pd.DataFrame()

# Load the pre-trained model files
nb_model = joblib.load('naive_bayes_model.joblib')  # Replace 'naive_bayes_model.joblib' with the actual filename
knn_model = joblib.load('knn_model.joblib')  # Replace 'knn_model.joblib' with the actual filename
c45_model = joblib.load('c45_model.joblib')  # Replace 'c45_model.joblib' with the actual filename

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
        # Label encoding
        if st.button("Label Encoding"):
            st.session_state.label_encoder = label_encode_data(df, categorical_features)
            st.write("Label encoding completed.")
            st.dataframe(st.session_state.label_encoder)

        st.markdown('<h3 style="text-align: left;"> Melakukan Normalisasi Data </h1>', unsafe_allow_html=True)
        # Min-Max scaling for all features
        if not st.session_state.label_encoder.empty:
            if st.button("Min-Max Scaling"):
                scaler = MinMaxScaler()
                normalized_data = pd.DataFrame(scaler.fit_transform(st.session_state.label_encoder), columns=st.session_state.label_encoder.columns)
                st.write("Min-Max scaling completed.")
                st.dataframe(normalized_data)
            else:
                st.warning("No numeric columns found for Min-Max Scaling.")
        else:
            st.warning("Encoded data is empty. Please perform label encoding first.")

                    
elif selected == 'Modelling':
    st.write("You are at Klasifikasi Datamining")
    # Load data for modeling
    if upload_file is not None:
        data_for_modeling = pd.read_csv(upload_file)

        # Specify the categorical features for label encoding
        categorical_features = ['jenis kelamin', 'penerima jps', 'belum menerima jps', 'target']

        # Perform label encoding
        data_for_modeling_encoded = label_encode_data(data_for_modeling, categorical_features)

        # Split data into features (X) and labels (y)
        x = data_for_modeling_encoded.drop('target', axis=1)
        y = data_for_modeling_encoded['target']

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Naive Bayes model
        nb_model.fit(x_train, y_train)
        nb_pred = nb_model.predict(x_test)
        nb_accuracy = metrics.accuracy_score(y_test, nb_pred)
        st.write("Naive Bayes Model Accuracy:", nb_accuracy)

        # Display confusion matrix for Naive Bayes
        nb_cm = metrics.confusion_matrix(y_test, nb_pred)
        st.pyplot(plot_confusion_matrix(nb_cm, title="Confusion Matrix for Naive Bayes"))

        # k-Nearest Neighbors (KNN) model
        knn_model.fit(x_train, y_train)
        knn_pred = knn_model.predict(x_test)
        knn_accuracy = metrics.accuracy_score(y_test, knn_pred)
        st.write("KNN Model Accuracy:", knn_accuracy)

        # Display confusion matrix for KNN
        knn_cm = metrics.confusion_matrix(y_test, knn_pred)
        st.pyplot(plot_confusion_matrix(knn_cm, title="Confusion Matrix for KNN"))

        # C4.5 decision tree model
        c45_model.fit(x_train, y_train)
        c45_pred = c45_model.predict(x_test)
        c45_accuracy = metrics.accuracy_score(y_test, c45_pred)
        st.write("C4.5 Decision Tree Model Accuracy:", c45_accuracy)
        
        # Display confusion matrix for C4.5 Decision Tree
        c45_cm = metrics.confusion_matrix(y_test, c45_pred)
        st.pyplot(plot_confusion_matrix(c45_cm, title="Confusion Matrix for C4.5 Decision Tree"))

    else:
        st.warning("Please upload a CSV file for modeling.")

elif selected == 'Evaluasi':
    st.write("You are at Uji Coba")
