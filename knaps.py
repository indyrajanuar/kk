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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False) 
    return x_train, x_test, y_train, y_test, None

def load_model():
    # Load pre-trained ERNN model
    model = keras.models.load_model('model-final.h5')
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

            # Convert y_test and y_pred to numpy arrays
            y_test = y_test.to_numpy()
            y_pred = y_pred.flatten()

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
                     <td style="text-align: center; border: none;"><h5>Precision</h5></td>
                     <td style="text-align: center; border: none;"><h5>Recall</h5></td>
                     <td style="text-align: center; border: none;"><h5>F1- Score</h5></td>
                 </tr>
                 <tr>
                     <td style="text-align: center; border: none;">{accuracy * 100:.2f}%</td>
                     <td style="text-align: center; border: none;">{precision * 100:.2f}%</td>
                     <td style="text-align: center; border: none;">{recall * 100:.2f}%</td>
                     <td style="text-align: center; border: none;">{f1 * 100:.2f}%</td>
                 </tr>
             </table>
             """
                
            st.markdown(html_code, unsafe_allow_html=True)

            # # Membuat DataFrame untuk menampilkan x_test, prediksi vs aktual
            # comparison_df = x_test.copy()
            # comparison_df['Actual'] = y_test
            # comparison_df['Predicted'] = y_pred
            # st.write("<br><br>", unsafe_allow_html=True)
            # # Menampilkan DataFrame perbandingan hasil prediksi dan label aktual
            # st.write("DataFrame Perbandingan Hasil Prediksi dan Label Aktual")
            # st.dataframe(comparison_df)

            # # Menentukan TP, TN, FP, FN
            # tp_index = comparison_df[(comparison_df['Actual'] == 1) & (comparison_df['Predicted'] == 1)].index
            # tn_index = comparison_df[(comparison_df['Actual'] == 0) & (comparison_df['Predicted'] == 0)].index
            # fp_index = comparison_df[(comparison_df['Actual'] == 0) & (comparison_df['Predicted'] == 1)].index
            # fn_index = comparison_df[(comparison_df['Actual'] == 1) & (comparison_df['Predicted'] == 0)].index
            
            # # Menampilkan hasil TP, TN, FP, FN
            # st.write("True Positives (TP):")
            # st.dataframe(comparison_df.loc[tp_index])
            
            # st.write("True Negatives (TN):")
            # st.dataframe(comparison_df.loc[tn_index])
            
            # st.write("False Positives (FP):")
            # st.dataframe(comparison_df.loc[fp_index])
            
            # st.write("False Negatives (FN):")
            # st.dataframe(comparison_df.loc[fn_index])
            
    elif selected == 'ERNN + Bagging':
        st.write("Berikut merupakan hasil klasifikasi yang didapat dari pemodelan Elman Recurrent Neural Network (ERNN) dengan teknik Bagging")
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.image("bagging.png")
            st.write("<br><br>", unsafe_allow_html=True)
            st.write("Di bawah ini adalah matriks kebingungan dari iterasi bagging dengan tingkat akurasi tertinggi.")
            
            # Data preprocessing
            df_cleaned = clean_data(df)
            preprocessed_data = preprocess_data(df_cleaned)
            normalized_data = normalize_data(preprocessed_data)
    
            # Splitting the data
            x_train, x_test, y_train, y_test, _ = split_data(normalized_data)
            
            # Initialize list to hold predictions from each model
            num_models = 5  # Define the number of models
            predictions = []
    
            # Load the models and make predictions
            for i in range(num_models):
                model = keras.models.load_model(f'model_iteration_5_model_{i + 1}.h5')
                y_pred = model.predict(x_test)
                predictions.append(y_pred)
            
            # Aggregate predictions through voting
            voted_predictions = np.mean(predictions, axis=0) >= 0.5  # Voting threshold of 0.5 for binary classification
            
            # Convert boolean array to integers for comparison with ground truth
            voted_predictions_int = voted_predictions.astype(int)
    
            # Calculate accuracy
            accuracy = accuracy_score(y_test, voted_predictions_int)
            precision = precision_score(y_test, voted_predictions_int)
            recall = recall_score(y_test, voted_predictions_int)
            f1 = f1_score(y_test, voted_predictions_int)
    
            # Generate confusion matrix
            cm = confusion_matrix(y_test, voted_predictions_int)
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
                report = classification_report(y_test, voted_predictions_int, zero_division=0)
            # Display the metrics
            html_code = f"""
            <br>
            <table style="margin: auto;">
                 <tr>
                     <td style="text-align: center; border: none;"><h5>Accuracy</h5></td>
                     <td style="text-align: center; border: none;"><h5>Precision</h5></td>
                     <td style="text-align: center; border: none;"><h5>Recall</h5></td>
                     <td style="text-align: center; border: none;"><h5>F1- Score</h5></td>
                 </tr>
                 <tr>
                     <td style="text-align: center; border: none;">{accuracy * 100:.2f}%</td>
                     <td style="text-align: center; border: none;">{precision * 100:.2f}%</td>
                     <td style="text-align: center; border: none;">{recall * 100:.2f}%</td>
                     <td style="text-align: center; border: none;">{f1 * 100:.2f}%</td>
                 </tr>
             </table>
            """
            
            st.markdown(html_code, unsafe_allow_html=True)

            # Mengonversi y_test menjadi numpy array jika perlu
            if isinstance(y_test, pd.Series):
                y_test = y_test.to_numpy()
        
            # # Membuat DataFrame untuk menampilkan x_test, prediksi vs aktual
            # comparison_df = x_test.copy()
            # comparison_df['Actual'] = y_test
            # comparison_df['Predicted'] = voted_predictions_int
        
            # st.write("<br><br>", unsafe_allow_html=True)
            # # Menampilkan DataFrame perbandingan hasil prediksi dan label aktual
            # st.write("DataFrame Perbandingan Hasil Prediksi dan Label Aktual")
            # st.dataframe(comparison_df)

            # # Menentukan TP, TN, FP, FN
            # tp_index = comparison_df[(comparison_df['Actual'] == 1) & (comparison_df['Predicted'] == 1)].index
            # tn_index = comparison_df[(comparison_df['Actual'] == 0) & (comparison_df['Predicted'] == 0)].index
            # fp_index = comparison_df[(comparison_df['Actual'] == 0) & (comparison_df['Predicted'] == 1)].index
            # fn_index = comparison_df[(comparison_df['Actual'] == 1) & (comparison_df['Predicted'] == 0)].index
            
            # # Menampilkan hasil TP, TN, FP, FN
            # st.write("True Positives (TP):")
            # st.dataframe(comparison_df.loc[tp_index])
            
            # st.write("True Negatives (TN):")
            # st.dataframe(comparison_df.loc[tn_index])
            
            # st.write("False Positives (FP):")
            # st.dataframe(comparison_df.loc[fp_index])
            
            # st.write("False Negatives (FN):")
            # st.dataframe(comparison_df.loc[fn_index])
        
    elif selected == 'Uji Coba':
        st.title("Uji Coba")
        st.write("Hipertensi dipengaruhi oleh beberapa faktor:")
        st.write('1. Jenis Kelamin: Jenis Kelamin pasien. P= Perempuan, L= Laki-Laki')
        st.write('2. Usia: Usia dari pasien')
        st.write('3. IMT: Indeks Massa Tubuh Pasien. Hitung IMT Menggunakan rumus IMT= Berat Badan(kg)/Tinggi badan(m)x Tinggi badan(m)')
        st.write('4. Sistolik: Tekanan darah sistolik Pasien (mmHg). Secara umum, tekanan darah manusia normal adalah 120 mmHg – 140 mmHg, namun pada individu yang mengalami hipertensi, tekanan darah sistoliknya melebihi 140 mmHg')
        st.write('5. Diastolik: Tekanan darah diastolik pasien (mmHg). Tekanan darah diastolik adalah tekanan darah saat jantung berelaksasi (jantung tidak sedang memompa darah) sebelum kembali memompa darah, tekanan darah diastolik meningkat melebihi 90 mmHg')
        st.write('6. Nafas: Nafas pasien yang dihitung /menit. Secara umum frekuensi nafas pada orang dewasa (19-59 tahun) adalah 12-20 nafas/menit')
        st.write('7. Detak Nadi: Detak nadi pasien. Pada orang normal dewasa detak nadi berkisar 60-100 kali/menit.')

        st.write("Masukkan nilai untuk pengujian:")        
        # Input fields
        with st.form("my_form"):
            with st.container():
                col1, col2 = st.columns(2)  # Split the layout into two columns
                with col1:
                    age = st.number_input("Umur (tahun)", min_value=0, max_value=150, step=1, value=30)
                    bmi = st.number_input("IMT", min_value=0.0, max_value=100.0, step=0.1, value=25.0)
                    systole = st.number_input("Sistole (mm/Hg)", min_value=0, max_value=300, step=1, value=120)
                    gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
                with col2:
                    diastole = st.number_input("Diastole (mm/Hg)", min_value=0, max_value=200, step=1, value=80)
                    breaths = st.number_input("Nafas (/menit)", min_value=0, max_value=100, step=1, value=16)
                    heart_rate = st.number_input("Detak Nadi (/menit)", min_value=0, max_value=300, step=1, value=70)

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
            
            # Load the selected model and make predictions
            if model_choice == "Elman Recurrent Neural Network":
                model = keras.models.load_model('model-final.h5')
                predictions = model.predict(datanorm)
                final_prediction = predictions[-1]
            else:
                num_models = 5  # Adjust this to the number of bagging models you have
                predictions = []
        
                for i in range(num_models):
                    model = keras.models.load_model(f'model_iteration_5_model_{i + 1}.h5')
                    y_pred = model.predict(datanorm)
                    predictions.append(y_pred)
        
                # Aggregate predictions through voting
                voted_predictions = np.mean(predictions, axis=0) >= 0.5  # Voting threshold of 0.5 for binary classification
                final_prediction = voted_predictions[-1]  # Take the prediction for the latest input data

        
            # Perform classification
            y_pred = (final_prediction > 0.5).astype("int32")
        
            # Display result
            st.write("Hasil klasifikasi:")
            if y_pred == 1:
                st.write("Data termasuk dalam kategori 'Diagnosa': YA")
            else:
                st.write("Data termasuk dalam kategori 'Diagnosa': TIDAK")
                
if __name__ == "__main__":
    main()
