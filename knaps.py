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
import joblib

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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) 
    return x_train, x_test, y_train, y_test, None

def load_model():
    # Load pre-trained ERNN model
    model = keras.models.load_model('model_fold_4 (1).h5')
    return model

def ernn(data, model):
    if data is None:
        return None, "Data is not available"
    # Apply Threshold
    y_pred = model.predict(data)
    y_pred = (y_pred > 0.5).astype(int)
    return y_pred

def load_keras_model(model_path):
    # Load a pre-trained Keras model
    model = keras.models.load_model(model_path)
    return model

# Fungsi untuk melakukan prediksi dengan Model1 (ERNN)
def predict_with_model1(data):
    # Memuat model
    model1 = load_keras_model('model_fold_4 (1).h5')
    # Melakukan prediksi
    predictions1 = model1.predict(datanorm1)
    y_pred1 = (predictions1 > 0.5).astype("int32")
    return y_pred1

# Fungsi untuk melakukan prediksi dengan Model2 (ERNN + Bagging)
def predict_with_model2(data):
    # Memuat model
    model2 = load_keras_model('model-final.h5')
    # Melakukan prediksi
    predictions2 = model2.predict(datanorm2)
    y_pred2 = (predictions2 > 0.5).astype("int32")
    return y_pred2    
    
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
        st.write('Selamat Datang di Website Data Penyakit Hipertensi menggunakan Metode Elman Recurrent Neural Network')
        st.write('Hipertensi (Tekanan Darah Tinggi) merupakan kondisi dimana terjadinya peningkatan tekanan darah sistolik ≥ 140 mmHg atau diastolik ≥ 90 mmHg. Hipertensi biasa disebut dengan “the silent killer”, dikarenakan kebanyakan penderitanya tidak sadar dirinya mengidap hipertensi, dan baru menyadari ketika telah terjadinya komplikasi. Dan diketahui bahwa dari penderita hipertensi, hanya sepertiga atau 36.8% dari penderita hipertensi yang terdiagnosa oleh tenaga medis dan sekitar hanya 0.7% yang meminum obat.')
        st.write('Elman Recurrent Neural Network adalah jenis Jaringan Syaraf Tiruan yang memperkenalkan lapisan tambahan yang dikenal sebagai lapisan konteks. Neuron konteks berperan dalam menyimpan informasi dari neuron hidden, yang akan digunakan bersamaan dengan data input dalam perhitungan fungsi pembelajaran')
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
            
            st.markdown('<h3 style="text-align: left;"> Menghapus atribut yang tidak diinginkan </h1>', unsafe_allow_html=True)
            st.write('Pada bagian ini melakukan pembersihan dataset yang tidak memiliki relevansi terhadap faktor risiko pada penyakit hipertensi, seperti menghapus satuan yang tidak diperlukan.')
            if st.button("Submit"):
                df_cleaned = clean_data(df)
                st.write("Data completed.")
                st.dataframe(df_cleaned)
                st.session_state.df_cleaned = df_cleaned
                
            st.markdown('<h3 style="text-align: left;"> Melakukan Transformation Data </h1>', unsafe_allow_html=True)
            st.write('Dibagian ini terjadi proses perubahan pada data ke dalam bentuk atau format yang akan diproses oleh sistem, dengan maksud memudahkan dalam pengelolaan data tersebut.')
            if st.button("Transformation Data"):  # Check if button is clicked
                if 'df_cleaned' in st.session_state:  # Check if cleaned data exists in session state
                    preprocessed_data = preprocess_data(st.session_state.df_cleaned.copy())
                    st.write("Transformation completed.")
                    st.dataframe(preprocessed_data)
                    st.session_state.preprocessed_data = preprocessed_data  # Store preprocessed data in session state
    
            st.markdown('<h3 style="text-align: left;"> Melakukan Normalisasi Data </h1>', unsafe_allow_html=True)
            st.write('Dalam penelitian ini digunakan metode normalisasi min-max normalization, metode ini mengubah data numerik menjadi range nol sampai satu [0-1].')
            if 'preprocessed_data' in st.session_state:  # Check if preprocessed_data exists in session state
                if st.button("Normalize Data"):
                    normalized_data = normalize_data(st.session_state.preprocessed_data.copy())
                    st.write("Normalization completed.")
                    st.dataframe(normalized_data)
    
    elif selected == 'Klasifikasi ERNN':
        st.write("<h5 style='text-align: center;'>Konfigurasi Elman Recurrent Neural Network</h5>", unsafe_allow_html=True)
        st.write("""
            <table style="margin: auto;">
                <tr>
                    <td style="text-align: center;"><b>Parameter</b></td>
                    <td style="text-align: center;"><b>Nilai</b></td>
                </tr>
                <tr>
                    <td style="text-align: center;">Neuron Input</td>
                    <td style="text-align: center;">7</td>
                </tr>
                <tr>
                    <td style="text-align: center;">Neuron Hidden</td>
                    <td style="text-align: center;">6</td>
                </tr>
                <tr>
                    <td style="text-align: center;">Neuron Context</td>
                    <td style="text-align: center;">6</td>
                </tr>
                <tr>
                    <td style="text-align: center;">Neuron Output</td>
                    <td style="text-align: center;">1</td>
                </tr>
                <tr>
                    <td style="text-align: center;">Fungsi Aktivasi</td>
                    <td style="text-align: center;">Sigmoid</td>
                </tr>
                <tr>
                    <td style="text-align: center;">Optimizer</td>
                    <td style="text-align: center;">Adam</td>
                </tr>
                <tr>
                    <td style="text-align: center;">Max Error</td>
                    <td style="text-align: center;">0.0001</td>
                </tr>
                <tr>
                    <td style="text-align: center;">Pembagian Data</td>
                    <td style="text-align: center;">70%:30%</td>
                </tr>
                <tr>
                    <td style="text-align: center;">Epoch</td>
                    <td style="text-align: center;">500</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
        
        st.write("<br><br>", unsafe_allow_html=True)
        st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan Elman Recurrent Neural Network (ERNN)")
        st.write("<br>", unsafe_allow_html=True)

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
            <br>
            <table style="margin: auto;">
                <tr>
                    <td style="text-align: center; border: none;"><h5>Accuracy</h5></td>
                </tr>
                <tr>
                    <td style="text-align: center; border: none;">{accuracy * 100:.2f}%</td>
                </tr>
            </table>
            """
                
            st.markdown(html_code, unsafe_allow_html=True)
            
    elif selected == 'ERNN + Bagging':
        st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan Elman Recurrent Neural Network (ERNN) dengan teknik Bagging")
        st.image('bagging.png', caption='')
        # Display the metrics
        html_code = f"""
        <br>
        <table style="margin: auto;">
            <tr>
                <td style="text-align: center; border: none;"><h5>5 Iterations</h5></td>
                <td style="text-align: center; border: none;"><h5>10 Iterations</h5></td>
                <td style="text-align: center; border: none;"><h5>15 Iterations</h5></td>
                <td style="text-align: center; border: none;"><h5>20 Iterations</h5></td>        
            </tr>
            <tr>
                <td style="text-align: center; border: none;">94.63%</td>
                <td style="text-align: center; border: none;">94.43%</td>
                <td style="text-align: center; border: none;">94.04%</td>
                <td style="text-align: center; border: none;">94.63%</td>
            </tr>
        </table>
        """                
        st.markdown(html_code, unsafe_allow_html=True)
        
    elif selected == 'Uji Coba':
        st.title("Uji Coba")
        st.write("Masukkan nilai untuk pengujian:")
        
        # Input fields
        with st.form("my_form"):
            with st.container():
                col1, col2 = st.columns(2)  # Split the layout into two columns
                with col1:
                    age = st.number_input("Umur", min_value=0, max_value=150, step=1, value=30)
                    bmi = st.number_input("IMT", min_value=0.0, max_value=100.0, step=0.1, value=25.0)
                    systole = st.number_input("Sistole", min_value=0, max_value=300, step=1, value=120)
                    gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
                with col2:
                    diastole = st.number_input("Diastole", min_value=0, max_value=200, step=1, value=80)
                    breaths = st.number_input("Nafas", min_value=0, max_value=100, step=1, value=16)
                    heart_rate = st.number_input("Detak Nadi", min_value=0, max_value=300, step=1, value=70)

            model_choice = st.selectbox("Pilih Model", ["Elman Recurrent Neural Network", "ERNN+Bagging"])
        
            submit_button = st.form_submit_button(label='Submit')
        
        # Convert gender to binary
        # gender_binary = 1 if gender == "Perempuan" else 0
            
        # Proses data setelah form disubmit
        if submit_button:
            # Prepare input data for testing
            data = pd.DataFrame({
                "Umur Tahun": [age],
                "IMT": [bmi],
                "Sistole": [systole],
                "Diastole": [diastole],
                "Nafas": [breaths],
                "Detak Nadi": [heart_rate],
                "Jenis Kelamin_L" : [0 if gender.lower() == 'perempuan' else 1],
                "Jenis Kelamin_P" : [1 if gender.lower() == 'perempuan' else 0]
            })
            
            new_data = pd.DataFrame(data)
            datatest = pd.read_csv('x_test2.csv')  
            datatest = pd.concat([datatest, new_data], ignore_index=True)
            #st.write(datatest)
            # Muat objek normalisasi
            normalizer = joblib.load('normalized_data1 (1).pkl')
            # Terapkan transformasi pada data pengujian
            datanorm = normalizer.fit_transform(datatest)
            #st.write(datanorm)
            
            if model_choice == "Elman Recurrent Neural Network":
                y_pred = predict_with_model1(data)
            elif model_choice == "ERNN + Bagging":
                y_pred = predict_with_model2(data)

            predictions = model.predict(datanorm)
            y_pred = (predictions > 0.5).astype("int32")
            
            if y_pred[-1] == 1:
                st.write("Hasil klasifikasi:")
                st.write("Data termasuk dalam kategori 'Diagnosa': YA")
            else:
                st.write("Hasil klasifikasi:")
                st.write("Data termasuk dalam kategori 'Diagnosa': TIDAK")
                
                
if __name__ == "__main__":
    main()
