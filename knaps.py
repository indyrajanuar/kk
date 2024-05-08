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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) 
    return x_train, x_test, y_train, y_test, None

def load_model():
    # Load pre-trained ERNN model
    model = keras.models.load_model('model-final (10).h5')
    return model

def ernn(data, model):
    if data is None:
        return None, "Data is not available"
    # Apply Threshold
    y_pred = model.predict(data)
    y_pred = (y_pred > 0.5).astype(int)
    return y_pred
    
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
            
            st.markdown('<h3 style="text-align: left;"> Melakukan Cleaning Data </h1>', unsafe_allow_html=True)
            st.write('Pada bagian ini melakukan pembersihan dataset yang tidak memiliki relevansi terhadap faktor risiko pada penyakit hipertensi, seperti menghapus satuan yang tidak diperlukan.')
            if st.button("Clean Data"):
                df_cleaned = clean_data(df)
                st.write("Data cleaning completed.")
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
        st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan Elman Recurrent Neural Network (ERNN) dengan teknik Bagging")
        st.image('bagging.png', caption='')
        
    elif selected == 'Uji Coba':
        st.title("Uji Coba")
        st.write("Masukkan nilai untuk pengujian:")
    
        # Input fields
        age = st.number_input("Umur", min_value=0, max_value=150, step=1, value=30)
        bmi = st.number_input("IMT", min_value=0.0, max_value=100.0, step=0.1, value=25.0)
        systole = st.number_input("Sistole", min_value=0, max_value=300, step=1, value=120)
        diastole = st.number_input("Diastole", min_value=0, max_value=200, step=1, value=80)
        breaths = st.number_input("Nafas", min_value=0, max_value=100, step=1, value=16)
        heart_rate = st.number_input("Detak Nadi", min_value=0, max_value=300, step=1, value=70)
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    
        # Convert gender to binary
        gender_binary = 1 if gender == "Perempuan" else 0
        
        # Button for testing
        if st.button("Hasil Uji Coba"):
            # Prepare input data for testing
            input_data = pd.DataFrame({
                "Umur": [age],
                "IMT": [bmi],
                "Sistole": [systole],
                "Diastole": [diastole],
                "Nafas": [breaths],
                "Detak Nadi": [heart_rate],
                "Jenis Kelamin": [gender_binary]
            })

            new_data = pd.DataFrame(datafix)
            datatest = pd.read_csv('transformed_data.csv')  
            datatest = pd.concat([datatest, new_data], ignore_index=True)
            datanorm = joblib.load('normalized_data.pkl').fit_transform(datatest)
            datapredict = keras.models.load_model('model-final (10).h5').predict(datanorm)
        
            # Perform classification
            y_pred = ernn(datapredict)
            
            # Display result
            if y_pred is None:
                st.write("Insufficient data for classification")
            else:
                if y_pred[0] == 1:
                    st.write("Hasil klasifikasi:")
                    st.write("Data termasuk dalam kategori 'Diagnosa': YA")
                else:
                    st.write("Hasil klasifikasi:")
                    st.write("Data termasuk dalam kategori 'Diagnosa': TIDAK")
            

if __name__ == "__main__":
    main()
