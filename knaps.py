import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import streamlit as st

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import altair as alt

from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("By: Indyra Januar - 200411100022")
st.write("Grade: Penambangan Data C")
upload_data, preporcessing, modeling, implementation = st.tabs(["Upload Data", "Prepocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan pada percobaan ini adalah data penyakit jantung yang di dapat dari UCI (Univercity of California Irvine)")
    st.write("link dataset : https://archive.ics.uci.edu/ml/datasets/Heart+Disease")
    st.write("Terdiri dari 270 dataset terdapat 13 atribut dan 2 kelas.")
    st.write("Heart Attack (Serangan Jantung) adalah kondisi medis darurat ketika darah yang menuju ke jantung terhambat.")
    
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)


with preporcessing:
    st.write("""# Preprocessing""")
    
    "### There's no need for categorical encoding"
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    X,y

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    st.write("Hasil Preprocesing : ", scaled)

    "### Splitting the dataset into training and testing data"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)
    st.write("Shape for training data", X_train.shape, y_train.shape)
    st.write("Shape for testing data", X_test.shape, y_test.shape)

    "### Feature Scaling"
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train,X_test
    

with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('SVM')
    mod = st.button("Modeling")

    # NB
    model = GaussianNB()
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    akurasi_nb = round(accuracy_score(y_test, predicted)*100)

    #KNN
    model = KNeighborsClassifier(n_neighbors = 1)  
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    
    akurasi_knn = round(accuracy_score(y_test, predicted.round())*100)

    #SVM
    model = SVC()
    model.fit(X_train, y_train)
    
    predicted = model.predict(X_test)
    akurasi_svm = round(accuracy_score(y_test, predicted)*100)

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi_nb))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(akurasi_knn))
    if des :
        if mod :
            st.write("Model SVM score : {0:0.2f}" . format(akurasi_svm))

    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi_nb,akurasi_knn,akurasi_svm],
            'Nama Model' : ['Naive Bayes','KNN','SVM']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

with implementation:
    st.write("# Implementation")

    #age
    age = st.number_input('Umur Pasien')

    #sex
    sex = st.radio("Jenis Kelamin",('Laki-Laki', 'Perempuan'))
    if sex == "Laki-Laki":
        sex_Female = 0
        sex_Male = 1
    elif sex == "Perempuan":
        sex_Female = 1
        sex_Male = 0
    
    #blood pressure
    trtbps = st.number_input('Tekanan Darah (mm Hg)')

    #cholestoral
    chol = st.number_input('Kolesterol (mg/dl)')

    #fasting blood sugar
    fbs = st.radio("Gula Darah Puasa > 120 mg/dl",('No', 'Yes'))
    if fbs == "Yes":
        fbs_y = 1
        fbs_n = 0
    elif fbs == "No":
        fbs_y = 0
        fbs_n = 1
    
    #Maximum heart rate achieved
    thalachh = st.number_input('Detak jantung maksimum')

    #Exercise induced angina
    exang = st.radio("Nyeri Dada",('Ya', 'Tidak'))
    if exang == "Ya":
        exang_y = 1
        exang_n = 0
    elif exang == "Tidak":
        exang_y = 0
        exang_n = 1

    #old peak
    oldpeak = st.number_input('ST depression induced by exercise relative to rest')

    def submit():
        # input
        inputs = np.array([[
            age,
            sex_Female, sex_Male,
            trtbps,
            chol,
            fbs_y, fbs_n,
            thalachh,
            exang_y, exang_n,
            oldpeak
            ]])

        le = joblib.load("le.save")

        if akurasi_nb > akurasi_knn and akurasi_svm:
            model = joblib.load("nb.joblib")

        elif akurasi_knn > akurasi_nb and akurasi_svm:
            model = joblib.load("knn.joblib")

        elif akurasi_svm > akurasi_knn and akurasi_nb:
            model = joblib.load("svm.joblib")
    
        y_pred3 = model.predict(inputs)
        st.write(f"Berdasarkan data yang di masukkan, maka pasien termasuk : {le.inverse_transform(y_pred3)[0]}")
        st.write("0 = Tidak menderita penyakit jantung")
        st.write("1 = menderita penyakit jantung")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()
