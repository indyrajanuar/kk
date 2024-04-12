import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
    
def split_data(data):
    # split data fitur, target
    x = data.drop('Diagnosa', axis=1)
    y = data['Diagnosa']
    # Check if the dataset has sufficient samples for splitting
    if len(data) < 2:
        return None, None, "Insufficient data for classification" 
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
    
def load_bagging_model(iteration):
    # Load Bagging models based on the specified iteration
    bagging_models = []
    if iteration == 3:
        for i in range(1, 4):
            model_path = f'model_3_{i}.h5'
            bagging_model = keras.models.load_model(model_path)
            bagging_models.append(bagging_model)
    elif iteration == 5:
        for i in range(1, 6):
            model_path = f'model_5_{i}.h5'
            bagging_model = keras.models.load_model(model_path)
            bagging_models.append(bagging_model)
    elif iteration == 7:
        for i in range(1, 8):
            model_path = f'model_7_{i}.h5'
            bagging_model = keras.models.load_model(model_path)
            bagging_models.append(bagging_model)
    elif iteration == 9:
        for i in range(1, 10):
            model_path = f'model_9_{i}.h5'
            bagging_model = keras.models.load_model(model_path)
            bagging_models.append(bagging_model)
    else:
        raise ValueError(f"Invalid iteration specified: {iteration}. Please choose from [3, 5, 7, 9].")
    
    if not bagging_models:
        raise ValueError(f"No models were loaded for iteration {iteration}.")    
    return bagging_models

def apply_threshold(predictions, threshold):
    return (predictions > threshold).astype(int)

def classification_process(x_train, y_train, bagging_iterations):
    models = load_bagging_model(bagging_iterations)
    accuracies_all_iterations = []
    
    for iteration in bagging_iterations:
        accuracies = []

        for model in models:
            y_pred_prob = model.predict(x_test)
            y_pred = (y_pred_prob > 0.5).astype(int)  # Apply threshold if needed
            accuracy = np.mean(y_pred == y_test)
            accuracies.append(accuracy)

        average_accuracy = np.mean(accuracies)
        accuracies_all_iterations.append(average_accuracy)        
    return accuracies_all_iterations
    
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
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            if 'preprocessed_data' in st.session_state:  # Check if preprocessed_data exists in session state
                x_train, x_test, y_train, y_test, _ = split_data(st.session_state.preprocessed_data.copy())
                normalized_test_data = normalize_data(x_test)
                model = load_model()
                y_pred = ernn(normalized_test_data, model)
    
                # Generate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
        
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                st.pyplot(plt.gcf())  # Pass the current figure to st.pyplot()
        
                # Clear the current plot to avoid displaying it multiple times
                plt.clf()
        
                # Generate classification report
                with np.errstate(divide='ignore', invalid='ignore'):  # Suppress division by zero warning
                    report = classification_report(y_test, y_pred, zero_division=0)
        
                # Extract metrics from the classification report
                lines = report.split('\n')
                accuracy = float(lines[5].split()[1]) * 100
                precision = float(lines[2].split()[1]) * 100
                recall = float(lines[3].split()[1]) * 100
        
                # Display the metrics
                html_code = f"""
                <table style="margin: auto;">
                    <tr>
                        <td style="text-align: center;"><h5>Accuracy</h5></td>
                        <td style="text-align: center;"><h5>Precision</h5></td>
                        <td style="text-align: center;"><h5>Recall</h5></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;">{accuracy:.2f}%</td>
                        <td style="text-align: center;">{precision:.2f}%</td>
                        <td style="text-align: center;">{recall:.2f}%</td>
                    </tr>
                </table>
                """
                
                st.markdown(html_code, unsafe_allow_html=True)
                
    elif selected == 'ERNN + Bagging':
        st.write("You are at Klasifikasi ERNN + Bagging")
        bagging_iterations = [3, 5, 7, 9]  # Define your bagging iterations
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            if 'preprocessed_data' in st.session_state:  # Check if preprocessed_data exists in session state
                x_train, x_test, y_train, y_test, _ = split_data(st.session_state.preprocessed_data.copy())
                normalized_data = normalize_data(st.session_state.preprocessed_data.copy())
        
                # Perform ERNN + Bagging classification for each iteration
                accuracies_all_iterations = []
                for iteration in bagging_iterations:
                    models = load_bagging_model(iteration)
                    accuracies_all_iterations.append(classification_process(x_train, y_train, normalized_data, models, iteration))


    elif selected == 'Uji Coba':
        st.title("Uji Coba")

if __name__ == "__main__":
    main()
