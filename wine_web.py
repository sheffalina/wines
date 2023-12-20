# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

import streamlit as st 
import pandas as pd 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

df = pd.read_csv('wine_5.csv')
#df = df.drop(['Unnamed: 0'], axis=1)
st.title('Классификация вина')
st.write(df.head())
st.subheader('Информация о датасете')
st.write(df.describe())

st.subheader('Визуализация')
st.bar_chart(df)

x = df.drop(['quality'], axis=1)  # axis=1 - столбец
y = df.iloc[:, -1]  # выбираем последний столбец

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

def userreport():
    # ввод pregnancies, glucose с помощью ползунков
    fixed_acidity = st.sidebar.slider('Фиксированная кислотность', 4.6, 15.9, 5.0)
    volatile_acidity = st.sidebar.slider('Летучая кислотность', 0.12, 1.58, 0.12)
    citric_acid	= st.sidebar.slider('Лимонная кислота', 0.0, 1.0, 0.0)
    residual_sugar = st.sidebar.slider('Остаточный сахар', 0.9, 15.5, 0.0)
    chlorides = st.sidebar.slider('Хлориды', 0.012, 0.611, 0.0)
    free_sulfur_dioxide = st.sidebar.slider('Свободный диоксид серы', 1.0, 72.0, 0.0)
    total_sulfur_dioxide = st.sidebar.slider('Cуммарный диоксид серы', 6.0, 289.0, 0.0)
    density = st.sidebar.slider('Плотность', 0.99, 1.004, 0.0)
    pH	= st.sidebar.slider('pH', 2.74, 4.01, 0.0)
    sulphates = st.sidebar.slider('Сульфаты', 0.33, 2.0, 0.0)
    alcohol = st.sidebar.slider('Спирт', 8.4, 14.9, 0.0)

    # сбор введённых значений в словарь
    report = {
        'fixed acidity' : fixed_acidity,
        'volatile acidity' : volatile_acidity,
        'citric acid' : citric_acid,
        'residual sugar' : residual_sugar,
        'chlorides' : chlorides,
        'free sulfur dioxide' : free_sulfur_dioxide,
        'total sulfur dioxide' : total_sulfur_dioxide,
        'density' : density,
        'pH' : pH,
        'sulphates' : sulphates,
        'alcohol' : alcohol}  


    report = pd.DataFrame(report, index=[0])  # DataFrame будет иметь 1 строку с индексом 0
    return report    


userdata = userreport()

rf = RandomForestClassifier()
# lr = LogisticRegression()
rf.fit(xtrain, ytrain)
# lr.fit(xtrain, ytrain)

st.subheader('Точность: ')
# сравниваем полученные диагнозы с реальными
st.write(str(accuracy_score(ytest, rf.predict(xtest)) * 100) + '%')
# st.write(str(accuracy_score(ytest, lr.predict(xtest)) * 100) + '%')


userresult = rf.predict(userdata)  # передаём параметры, определяем наличие диабета
st.subheader('Тип вина: ')
if userresult[0] == 0:
    output = 'Плохое'
else:
    output = 'Хорошее'


st.write(output)



