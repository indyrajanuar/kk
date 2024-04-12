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
    bagging_models = []

    if iteration in [3, 5, 7, 9]:
        for i in range(1, iteration + 1):
            model_path = f'model_{iteration}_{i}.h5'  
            bagging_model = keras.models.load_model(model_path)
            bagging_models.append(bagging_model)
    else:
        raise ValueError("Invalid iteration specified. Please choose from [3, 5, 7, 9].")

    return bagging_models

def run_ernn_bagging(data):
    x = data.drop('Diagnosa', axis=1)
    y = data['Diagnosa']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    y_train = np.array(y_train).reshape(-1,)
    y_test = np.array(y_test).reshape(-1,)
    
    bagging_iterations = load_bagging_model(iteration=3)  
    
    models = []

    accuracies_all_iterations = []  
    for iteration_models in bagging_iterations:
        accuracies_per_iteration = []  

        for model in iteration_models:  # Fix the iteration here
            indices = np.random.choice(len(x_train), len(x_train), replace=True)
            x_bag = x_train.iloc[indices]
            y_bag = y_train[indices]

            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.Adam(learning_rate=0.1),
                          metrics=[keras.metrics.BinaryAccuracy()])

            history = model.fit(x_bag, y_bag, batch_size=32, epochs=200, verbose=0)
            models.append(model)

            accuracy = model.evaluate(x_test, y_test, verbose=0)[1]  
            accuracies_per_iteration.append(accuracy)
            
        avg_accuracy = np.mean(accuracies_per_iteration)
        accuracies_all_iterations.append(avg_accuracy)

    y_preds = []
    for model in models:
        y_pred = model.predict(x_test)
        y_pred = (y_pred > 0.5).astype(int)
        y_preds.append(y_pred)
    
    y_pred_avg = np.mean(y_preds, axis=0)  

    return y_test, y_pred_avg
    
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
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            normalized_data = normalize_data(st.session_state.preprocessed_data.copy())  
            y_test, y_pred, bagging_iterations, accuracies_all_iterations = run_ernn_bagging(normalized_data)
            
            print("Average accuracies for each bagging iteration:")
            for iteration, accuracy in zip(bagging_iterations, accuracies_all_iterations):
                print(f"Iteration {iteration}: {accuracy:.2f}%")
                
    elif selected == 'Uji Coba':
        st.title("Uji Coba")

if __name__ == "__main__":
    main()
