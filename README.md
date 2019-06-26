# Glaucoma Detection

## Requerimientos

### Python 3.7

Utilizando pip puedes instalar las bibliotecas requeridas para ejecutar este
proyecto de la siguiente manera:

```
$ pip install -r requirements.txt
```

### RIMONE

Es necesario tener RIMONE-r3 dentro de la carpeta `rimone` para obtener
resultados.

## Introducción

Con el objetivo de reutilizar el código creado, se creo una biblioteca. En esta
se encuentran las funciones utilizadas por cada uno de los scripts que se
describen a continuación. De igual manera, dentro de esta biblioteca se
encuentran una serie de valores constantes que se utilizan alrededor de todo el
proyecto.

La biblioteca creada puede ser encontrada dentro del directorio `lib`.

## Conjunto de datos

Para acceder a RIMONE durante la extracción de características, dentro de la
biblioteca existe una clase llamada RimoneDataset. Esta clase permite acceder al
conjunto completo de imágenes, máscaras y segmentaciones a través de métodos en
ella. Además, esta clase almacena la información del dataset en archivos
binarios dentro del directorio `rimone`, de este modo, el acceso a la
información es más rápido.

## Extracción de características

La extracción de características puede ser realizada ejecutando:

```
$ python3 extract_features.py
```

Además de extraer las características de cada imagen en RIMONE, el comando
anterior realizará una normalización estándar sobre el conjunto de patrones
obtenido y creará un conjunto de entrenamiento y otro de prueba para poder
realizar los demás pasos del proyecto. Estos conjuntos son guardados dentro del
directorio `datasets` bajo el nombre de `train.csv` y `test.csv`,
respectivamente.

El script `extract_features.py` hace uso de la función `extract_features`, la
cual se encuentra en `lib/extractor.py`. Esta función recupera información de
la clase `RimoneDataset` y extrae las diferentes características de cada
imagen (Haralick, LBP, Forma y CDR).

## Búsqueda en malla y selección del mejor clasificador

Para determinar cual es el clasificador que se va a utilizar para las siguientes
pruebas, se probaron 5 diferentes clasificadores:

- Bayesiano Ingenuo (`sklearn.naive?bayes.GaussianNB`)
- KNN (`sklearn.neighbors.KNeighborsClassifier`)
- Random Forest (`sklearn.ensemble.RandomForestClassifier`)
- Ada Boost (`sklearn.ensemble.AdaBoostClassifier`)
- Máquina de Soporte Vectorial (`sklearn.svm.SVC`)

La búsqueda en malla de los mejores parámetros para cada clasificador de acuerdo
a nuestro modelo es realizada ejecutando:

```
$ python3 determine_best_classifier.py
```

Este script, ejecuta una serie de búsquedas en malla sobre cada clasificador
para determinar la mejor combinación de parámetros de acuerdo a nuestro modelo.

La evaluación de cada clasificador es realizada a través de la función
`evaluate`, la cual se encuentra en `lib/evaluator.py`. Está función recibe un
el clasificador que se desee entrenar, junto con los parámetros deseados. Para
evaluar el clasificador, la función realiza un _Leave One Out_ sobre el conjunto
recibido.

## Selección de características

La selección de características es realizada ejecutando:

```
$ python3 select_features.py
```

El script `select_features.py` realiza _Forward Feature Selection_ con un
enfoque _wrapper_ tomando la sensibilidad como criterio de selección. La
evaluación de cada modelo generado es realizada con un _Leave One Out_ a través
de la función `evaluate`.

## Clasificación

El script `classify.py` considera el mejor clasificador y las mejores
características encontrados en los procedimientos anteriores. Para ejecutarlo y
ver los resultados del proyecto se utiliza el siguiente comando:

```
$ python3 classify.py
```

Este script, utiliza el conjunto de entrenamiento para entrenar una máquina de
soporte vectorial y clasifica los patrones en el conjunto de prueba con esta.
Los resultados de la clasificación son comparados con las observaciones para
evaluar el rendimiento de la clasificación. Las siguientes medidas son
consideradas:

- _Balanced Accuracy Score_ (`sklearn.metrics.balanced_accuracy_score`)
- Exactitud (Dentro de `lib.evaluator`)
- Coeficiente de correlación de Matthews (`sklearn.metrics.matthews_corrcoef`)
- _Balanced Error Rate_ (`lib.evaluator.ber`)
- Sensibilidad (Dentro de `lib.evaluator`)
- Especifidad (Dentro de `lib.evaluator`)
