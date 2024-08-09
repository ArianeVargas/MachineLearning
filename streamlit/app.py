import streamlit as st
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#titulo de la aplicacion
st.title('Analisis de machine learning sobre el Dataset Iris')

#1. Carga del dataset
st.header('1. Carga del dataset')
st.write('Cargamos el famoso dataset Iris que contiene datos sobre las especies de flores.')
df = sns.load_dataset('iris')
st.dataframe(df)

#2. Analisis exploratorio de datos (EDA)
st.header('2. Analisis exploratorio de datos (EDA)')
st.write('Exploramos la estructura del dataset, estadisticas basicas y relaciones entre variables.')

st.subheader("Distribución de Clases")
st.write(df['species'].value_counts())

st.subheader("Matriz de Correlación")
# Excluir columnas no numéricas antes de calcular la correlación
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot(plt.gcf()) #obtiene la cifra actual

#3. Procesamiento de daos 
st.header("3. Preprocesamiento de Datos")
st.write("Dividimos los datos en conjuntos de entrenamiento y prueba y escalamos las características.")

x = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler =StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

st.write(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
st.write(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

#4. Entrenamiento del modelos
st.header("4. Entrenamiento del Modelo")
st.write("Entrenamos un modelo de Random Forest para clasificar las especies de flores.")

model = RandomForestClassifier(random_state=42)
model.fit(x_train_scaled, y_train)

#5. Evaluacion del modelo
st.header("5. Evaluación del Modelo")
st.write("Evaluamos el modelo utilizando el conjunto de prueba y mostramos las métricas de rendimiento.")

y_pred = model.predict(x_test_scaled)

st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot(plt.gcf())

st.subheader('Informe de clasificacion')
report = classification_report(y_test, y_pred, output_dict=True)
st.write(pd.DataFrame(report).transpose())

st.subheader('Precision del modelo')
accuracy = accuracy_score(y_test, y_pred)
st.write(f"La precisión del modelo es: {accuracy:.2f}")

# 6. Conclusiones
st.header("6. Conclusiones")
st.write("""
En este análisis, hemos utilizado un modelo de Random Forest para clasificar las especies de flores del dataset Iris.
El modelo ha mostrado un buen rendimiento, con una precisión superior al 90%.
Sin embargo, siempre es importante considerar diferentes modelos y realizar ajustes adicionales para mejorar los resultados.
""")