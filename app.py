import streamlit as st 
import pandas as pd 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_score, recall_score
# from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
import base64
import img

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")
img_2 = get_img_as_base64("t4.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
# background-image: url("https://media.istockphoto.com/id/499889196/ru/%D1%84%D0%BE%D1%82%D0%BE/%D1%88%D1%82%D0%BE%D0%BF%D0%BE%D1%80-%D0%B8-%D0%B1%D1%83%D1%82%D1%8B%D0%BB%D0%BA%D0%B0-%D0%B2%D0%B8%D0%BD%D0%B0.webp?b=1&s=170667a&w=0&k=20&c=hfjMPZc8MQa9kSFcvA4jP4stvidgrt_iqxX0A-zkbcA=");
# background-size: 100%;
# background-position: top left;
# background-repeat: no-repeat;
# background-attachment: local;
background-image: url("data:image/png;base64,{img_2}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

# [data-testid="stSidebar"] > div:first-child {{
# background-image: url("data:image/png;base64,{img}");
# background-position: center; 
# background-repeat: no-repeat;
# background-attachment: fixed;
# }}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html= True)

df = pd.read_csv('wine_5.csv')
df = df.drop(['fixed acidity'], axis=1) 
df = df.drop(['free sulfur dioxide'], axis=1)

st.title('Классификация красного вина')

st.write("""
    
    Цель проекта состоит в том, чтобы разработать классификационную модель, позволяющую определить качество красного вина (хорошее или плохое).
    """)

st.markdown("""
            Данные для исследования - это открытый датасет по оценкам красного вина и его химического состава от производителя Vinho verde.
             В данных 11 химических признаков и оценка.
            """)
st.write(df.head())

st.markdown("""
    ### Описание полей
    
       
    """)

#st.markdown("- Fixed acidity - фиксированная кислотность", help="Участвует в сбалансированности вкуса вина, привносит свежесть вкусу")
st.markdown("- Volatile acidity - летучая кислотность", help="Обусловлена наличием летучих кислот в вине, например, таких как уксусная кислота")
st.markdown("- Citric acid - лимонная кислота", help="Придает вину более яркую и свежую нотку, делая его более освежающим и приятным на вкус. Она также помогает бороться с излишней сладостью, добавляя нотку кислинки, которая балансирует вкус напитка. Влияет на структуру вина, делая его более стабильным и сохраняющим свежесть на протяжении длительного времени.")
st.markdown("- Residual sugar - остаточный сахар", help="Показывает количество сахара, который не был превращен в спирт в процессе ферментации вина. Участвует в сладости вкуса вина")
st.markdown("- Chlorides - хлориды", help="Количество соли, присутствующей в вине")
#st.markdown("- Free sulfur dioxide - свободный диоксид серы", help="Они же сульфиты, используются в виноделии в качестве безопасного антисептика. Сульфиты не дают вину скисать и потерять свои вкусовые качества. Присутствуют в вине в свободном виде (газообразном) исвязанном виде (соединившись с водой)")
st.markdown("- Total sulfur dioxide - суммарный диоксид серы", help="Они же сульфиты, используются в виноделии в качестве безопасного антисептика. Сульфиты не дают вину скисать и потерять свои вкусовые качества. Присутствуют в вине в свободном виде (газообразном) исвязанном виде (соединившись с водой)")
st.markdown("- Density - плотность", help="Показывает, сколько массы (граммы) содержится в 1 миллилитре вина. Плотность вина напрямую зависит от количества сахара, алкоголя и кислоты в составе напитка.")
st.markdown("- pH", help="Выступает характеристикой цвета вина. Вина с высоким pH темнее и имеют фиолетовый оттенок цвета. Вина с низким pH светлее и имеют ярко-розовый и ярко-красный оттенок цвета")
st.markdown("- Sulphates - сульфаты", help="Предотвращают окисление и сохранять качество продукта на протяжении длительного времени. Серные соединения, добавленные в вино, помогают защитить его от воздействия кислорода, которое может привести к недостатку свежести, окислению и порче вкусовых качеств")
st.markdown("- Alcohol - спирт", help="Характеризует крепость вина")

x = df.drop(['quality'], axis=1)  # axis=1 - столбец
y = df.iloc[:, -1]  # выбираем последний столбец

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

def userreport():
    #number_input = st.sidebar.number_input('Фиксированная кислотность', value=None, placeholder="Type a number...")
    #number = st.sidebar.slider("", 4.6, 15.9, value=number_input)
    
    acid = st.sidebar.number_input('Летучая кислотность', value=None, placeholder="Type a number...")
    number_2 = st.sidebar.slider("", 0.12, 1.58, value=acid)
    
    citric_acid_2 = st.sidebar.number_input('Лимонная кислота', value=None, placeholder="Type a number...")
    citric_acid	= st.sidebar.slider("", 0.0, 1.0, value=citric_acid_2)
    
    residual_sugar_2 = st.sidebar.number_input('Остаточный сахар', value=None, placeholder="Type a number...")
    residual_sugar = st.sidebar.slider("", 0.9, 15.5, value=residual_sugar_2)

    chlorides_2 = st.sidebar.number_input('Хлориды', value=None, placeholder="Type a number...")    
    chlorides = st.sidebar.slider("", 0.012, 0.611, value=chlorides_2)

    #free_sulfur_dioxide_2 = st.sidebar.number_input('Свободный диоксид серы', value=None, placeholder="Type a number...")
    #free_sulfur_dioxide = st.sidebar.slider("", 1.0, 72.0, value=free_sulfur_dioxide_2)

    total_sulfur_dioxide_2 = st.sidebar.number_input('Cуммарный диоксид серы', value=None, placeholder="Type a number...")
    total_sulfur_dioxide = st.sidebar.slider("", 6.0, 289.0, value=total_sulfur_dioxide_2)

    density_2 = st.sidebar.number_input('Плотность', value=None, placeholder="Type a number...")
    density = st.sidebar.slider("", 0.990, 1.004, value=density_2)

    pH_2 = st.sidebar.number_input('pH', value=None, placeholder="Type a number...")
    pH	= st.sidebar.slider("", 2.74, 4.01, value=pH_2)

    sulphates_2 = st.sidebar.number_input('Сульфаты', value=None, placeholder="Type a number...")
    sulphates = st.sidebar.slider("", 0.33, 2.0, value=sulphates_2)

    alcohol_2 = st.sidebar.number_input('Спирт', value=None, placeholder="Type a number...")
    alcohol = st.sidebar.slider("", 8.4, 14.9, value=alcohol_2)

    # сбор введённых значений в словарь
    report = {
        #'fixed acidity' : number,
        'volatile acidity' : number_2,
        'citric acid' : citric_acid,
        'residual sugar' : residual_sugar,
        'chlorides' : chlorides,
        #'free sulfur dioxide' : free_sulfur_dioxide,
        'total sulfur dioxide' : total_sulfur_dioxide,
        'density' : density,
        'pH' : pH,
        'sulphates' : sulphates,
        'alcohol' : alcohol}  


    report = pd.DataFrame(report, index=[0])  # DataFrame будет иметь 1 строку с индексом 0
    return report    


userdata = userreport()

rf = RandomForestClassifier()
rf.fit(xtrain, ytrain)

#st.subheader('Точность оценки (accuracy): ', help="Метрика Accuracy показывает, как часто модель правильно определяет класс")
st.markdown("<h5>Точность оценки (accuracy):</h5>", unsafe_allow_html=True, help="Метрика Accuracy показывает, как часто модель правильно определяет класс")
# сравниваем полученные результаты с реальными
st.write(str(accuracy_score(ytest, rf.predict(xtest)) * 100) + '%')




precision = precision_score(ytest, rf.predict(xtest)) * 100
recall = recall_score(ytest, rf.predict(xtest)) * 100
st.markdown("<h5>Точность оценки (precision):</h5>", unsafe_allow_html=True, help="Метрика Precision определяет, как много из обнаруженных моделью положительных результатов действительно являются положительными")
#st.subheader('Точность оценки (precision): ', help="Метрика Precision определяет, как много из обнаруженных моделью положительных результатов действительно являются положительными")
st.write(format(precision, '.3f') + '%')
#st.subheader('Точность оценки (recall): ', help="Метрика Recall определяет, как много положительных примеров были обнаружены моделью")
st.markdown("<h5>Точность оценки (recall):</h5>", unsafe_allow_html=True, help="Метрика Recall определяет, как много положительных примеров были обнаружены моделью")
st.write(format(recall, '.3f') + '%')



userresult = rf.predict(userdata)  # передаём параметры
st.markdown("\n>")
st.markdown("<h5>Тип вина:</h5>", unsafe_allow_html=True)
#st.subheader('Тип вина: ')
if userresult[0] == 0:
    st.markdown("<h1>Плохое</h1>", unsafe_allow_html=True)
    #output = 'Плохое'
else:
    st.markdown("<h1>Хорошее</h1>", unsafe_allow_html=True)
    #output = 'Хорошее'


#st.write(output)

if userresult[0] == 0:
    #smiley = ('😔')
    #st.image('./sad cat.png', width=300)
    st.image('./glass3.png', width=300)
else:
    st.image('./glasswine.png', width=300)
    #smiley = ('😍🥂')




my_list = ["Antinori Tignanello Toscana IGT 2019 - красное, сухое","Marques de Caceres Crianza 2017 - красное, сухое", 
           "Alamos Malbec 2021 - красное, сухое", "770 Miles Zinfandel - красное, сухое", "Felix Solis Mucho Mas - красное, сухое",
           "Duca di Saragnano Alchymia Primitivo - красное, полусухое"]

if userresult[0] == 1:
    st.title('Рекомендации вина: ')
    for item in my_list:
        st.write(item)
else:
    st.write('Не расстраивайтесь, выберите другой вариант :wink:')
    
# st.subheader('Визуализация')

# #st.bar_chart(df.drop(['quality'], axis=1), title="Это заголовок графика")
# #st.caption("Это подпись к графику")

# df_1 = df.drop(['quality'], axis=1)

# st.bar_chart(df)

# fig, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
# plt.title('Корреляция Пирсона\n', fontsize=20)
# st.write(fig)


# def send_prompt_to_bot_hub(prompt):
#   url = "https://www.phind.com/" # URL-адрес API Bot Hub
#   headers = {"Content-Type": "application/json"}
#   data = {"prompt": prompt}
#   response = requests.post(url, headers=headers, json=data)
#   return response.json()

# def display_response_on_streamlit(response):
#   st.write(response)

# def main():
#   prompt = "посоветуй красное вино"
#   response = send_prompt_to_bot_hub(prompt)
#   display_response_on_streamlit(response)

# if __name__ == "__main__":
#   main()

# importance = rf.feature_importances_
# st.bar_chart(importance)

# feature_importances = rf.feature_importances_
# st.bar_chart(x.columns, feature_importances)
# # plt.show()

# Список закусок, которые хорошо сочетаются с винами
appetizers = ['Салат', 'Сырный бутерброд', 'Гарнир', 'Паста', 'Бекон', 'Ветчина', 'Буженина', 
             'Оливки', 'Груша', 'Персик', 'Манго', 'Мясо кролика/птицы', 'Креветки', 'Мидии', 'Жирные виды рыб', 'Мороженое', 'Сладкая выпечка']

def get_random_appetizers(n):
  return random.choices(appetizers, k=n)

st.title('Рекомендация закусок:')

n = st.number_input('Выберите количество рекомендаций', min_value=1, max_value=len(appetizers), value=1, step=1)

if st.button('Получить рекомендации'):
  st.write('Рекомендуемые закуски для вина: ' + ', '.join(get_random_appetizers(n)))


# wines = ["Chardonnay", "Pinot Noir", "Merlot", "Sauvignon Blanc", "Riesling"]

# # Преобразование названий вин в векторы
# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(wines)

# # Вычисление матрицы косинусного расстояния
# cos_dist_matrix = 1 - cosine_similarity(vectors)

# # Нахождение пар вин с наименьшим расстоянием
# min_dist_index = np.unravel_index(np.argmin(cos_dist_matrix, axis=None), cos_dist_matrix.shape)

# # Вывод названий вин с наименьшим расстоянием
# print(f"Вина с наименьшим расстоянием: {wines[min_dist_index[0]]} и {wines[min_dist_index[1]]}")
  


# wines = ["Chardonnay", "Pinot Noir", "Merlot", "Sauvignon Blanc", "Riesling"]

# # Преобразование названий вин в векторы
# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(wines)

# # Вычисление матрицы косинусного расстояния
# cos_dist_matrix = 1 - cosine_similarity(vectors)

# # Нахождение пар вин с наименьшим расстоянием
# min_dist_index = np.unravel_index(np.argmin(cos_dist_matrix, axis=None), cos_dist_matrix.shape)

# # Вывод названий вин с наименьшим расстоянием
# st.write(f"Вина с наименьшим расстоянием: {wines[min_dist_index[0]]} и {wines[min_dist_index[1]]}")
