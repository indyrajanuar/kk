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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def clean_data(data):
    # Cleaning data
    data = data[data['Umur Tahun'].notnull()]
    data = data[data['Sistole'].notnull()]
    data = data[data['Diastole'].notnull()]
    data = data[data['Nafas'].notnull()]
    data = data[data['Detak Nadi'].notnull()]
    data['Umur Tahun'] = data['Umur Tahun'].apply(lambda x: int(x.split(' ')[0]))
    data['Sistole'] = data['Sistole'].apply(lambda x: int(x.split(' ')[0]))
    data['Diastole'] = data['Diastole'].apply(lambda x: int(x.split(' ')[0]))
    data['Nafas'] = data['Nafas'].apply(lambda x: int(x.split(' ')[0]))
    data['Detak Nadi'] = data['Detak Nadi'].apply(lambda x: int(x.split(' ')[0]))
    return data
    
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
    
def split_data(data):
    # split data fitur, target
    x = data.drop('Diagnosa', axis=1)
    y = data['Diagnosa']
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 
    return x_train, x_test, y_train, y_test, None

def load_model():
    # Load pre-trained ERNN model
    model = keras.models.load_model('model_fold_1.h5')
    return model

def ernn(data, model):
    if data is None:
        return None, None, "Data is not available"
    # Apply Threshold
    y_pred = model.predict(data)
    y_pred = (y_pred > 0.5).astype(int)
    return y_pred

def ernn_classification(normalized_input, model):
    # Atur diagnosis berdasarkan prediksi
    if prediction > 0.5:
        diagnosis = "Ya Hipertensi"
    else:
        diagnosis = "Tidak Hipertensi"
    
    return diagnosis
    
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
            
            st.markdown('<h3 style="text-align: left;"> Melakukan Cleaning Data </h1>', unsafe_allow_html=True)
            if st.button("Clean Data"):
                df_cleaned = clean_data(df)
                st.write("Data cleaning completed.")
                st.dataframe(df_cleaned)
                st.session_state.df_cleaned = df_cleaned
                
            st.markdown('<h3 style="text-align: left;"> Melakukan Transformation Data </h1>', unsafe_allow_html=True)
            if st.button("Transformation Data"):  # Check if button is clicked
                if 'df_cleaned' in st.session_state:  # Check if cleaned data exists in session state
                    preprocessed_data = preprocess_data(st.session_state.df_cleaned.copy())
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
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            
            # Data preprocessing
            df_cleaned = clean_data(df)
            preprocessed_data = preprocess_data(df_cleaned)
            normalized_data = normalize_data(preprocessed_data)

            # Splitting the data
            x_train, x_test, y_train, y_test, _ = split_data(normalized_data)
            
            # Load the model
            model = load_model()
            # Compile the model with appropriate metrics
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Predict using the model
            y_pred = ernn(x_test, model)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Kelas Prediksi')                
            plt.ylabel('Kelas Aktual')
            plt.title('Confusion Matrix')
            st.pyplot(plt.gcf())  # Pass the current figure to st.pyplot()        
            # Clear the current plot to avoid displaying it multiple times
            plt.clf()  

            # Generate classification report
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress division by zero warning
                report = classification_report(y_test, y_pred, zero_division=0)
            # Display the metrics
            html_code = f"""
            <table style="margin: auto;">
                <tr>
                    <td style="text-align: center;"><h5>Accuracy</h5></td>
                    <td style="text-align: center;"><h5>Precision</h5></td>
                    <td style="text-align: center;"><h5>Recall</h5></td>
                    <td style="text-align: center;"><h5>F1- Score</h5></td>
                </tr>
                <tr>
                    <td style="text-align: center;">{accuracy * 100:.2f}%</td>
                    <td style="text-align: center;">{precision * 100:.2f}%</td>
                    <td style="text-align: center;">{recall * 100:.2f}%</td>
                    <td style="text-align: center;">{f1 * 100:.2f}%</td>
                </tr>
            </table>
            """
                
            st.markdown(html_code, unsafe_allow_html=True)
            
    elif selected == 'ERNN + Bagging':
        st.write("You are at Klasifikasi ERNN + Bagging")
        st.image('bagging plotting.png', caption='')
        
    elif selected == 'Uji Coba':
        st.title("Uji Coba")
        st.write("Masukkan nilai untuk pengujian:")
    
        # Input fields
        Umur_Tahun = st.number_input("Umur", min_value=0, max_value=150, step=1)
        IMT = st.number_input("IMT", min_value=0.0, max_value=100.0, step=0.1)
        Sistole = st.number_input("Sistole", min_value=0, max_value=300, step=1)
        Diastole = st.number_input("Diastole", min_value=0, max_value=200, step=1)
        Nafas = st.number_input("Nafas", min_value=0, max_value=100, step=1)
        Detak_Nadi = st.number_input("Detak Nadi", min_value=0, max_value=300, step=1)
        Jenis_Kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        
        # Convert gender to binary
        gender_binary = 1 if Jenis_Kelamin == "Laki-laki" else 0
        submit = st.button('Uji Coba')      
         
        # Button for testing
        if submit:
            # Input data
            data_input = {
                'Umur Tahun': [Umur_Tahun],
                'IMT': [IMT],
                'Sistole': [Sistole],
                'Diastole': [Diastole],
                'Nafas': [Nafas],
                'Detak Nadi': [Detak_Nadi],
                'Jenis Kelamin': [gender_binary],
                'Diagnosa': [1]
            }
        
            # Convert input data into DataFrame
            data_input_df = pd.DataFrame(data_input)
            preprocess_input = preprocess_data(data_input_df)
            normalized_input = normalize_data(preprocess_input)
        
            # Load the pre-trained model
            model = load_model()
        
            # Perform classification
            prediction = ernn_classification(normalized_input, model)

            # Display the prediction result
            st.write(f"Hasil klasifikasi: {prediction}")

if __name__ == "__main__":
    main()
