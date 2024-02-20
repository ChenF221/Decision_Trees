import pandas as pd
from matplotlib import pyplot as plot
import sklearn.tree as skl
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder


# Adquiere los datos desde un archivo csv usando biblioteca PANDAS
dataframe = pd.read_csv("p11_medicamento.csv", encoding="ISO-8859-1")

# Preprocesamiento (Transformar valores categóricos a numéricos)
edad = LabelEncoder()
sexo = LabelEncoder()
presionSanguinea = LabelEncoder()
colesterol = LabelEncoder()
medicamento = LabelEncoder()

dataframe["edad"] = edad.fit_transform(dataframe["edad"])
dataframe["sexo"] = sexo.fit_transform(dataframe["sexo"])
dataframe["presionSanguinea"] = presionSanguinea.fit_transform(dataframe["presionSanguinea"])
dataframe["colesterol"] = colesterol.fit_transform(dataframe["colesterol"])
dataframe["medicamento"] = medicamento.fit_transform(dataframe["medicamento"])

# Prepara los datos
features_cols = ["edad", "sexo", "presionSanguinea", "colesterol"]
X = dataframe[features_cols]
y = dataframe.medicamento

# Entrenamiento
tree = skl.DecisionTreeClassifier(criterion="gini")
tree.fit(X, y)

# Visualización
px = 1 / plot.rcParams["figure.dpi"]  # Pixel in pulgadas
fig = plot.figure(figsize=(800 * px, 800 * px))
_ = plot_tree(tree, feature_names=features_cols, class_names=["No", "Yes"], filled=True)

plot.show()


# Probar el Modelo

# ¿Qué medicamento se le debe recomendar a una chica de 
# mediana edad, con presión sanguínea baja y colesterol normal? 

df_prueba_nueva = pd.DataFrame({
    "edad": [1],  # joven(0), mediana-edad(1), senior(2)
    "sexo": [0],  # femenino(0), masculino(1)
    "presionSanguinea": [2],  # alta(0), normal(1), baja(2)
    "colesterol": [0]  # normal(0), alto(1)
})

# Realizar la predicción con el modelo entrenado
prediccion_nueva = tree.predict(df_prueba_nueva)

# Mostrar el resultado de la prueba
print("Resultado de la prueba para una chica de mediana edad con presión sanguínea baja y colesterol normal:")
print("*******************************************************")
print("Con los datos:")
print(df_prueba_nueva)
print("\nSe recomienda:")
if prediccion_nueva[0] == 0:
    print("Tomar Medicamento A")
else:
    print("Tomar Medicamento B")
print("*******************************************************")
