# Proyecto PLN
Proyecto de la materia  de Procesamiento de Lenguaje Natural

## Toxic Spans Detection

Los datos se encuentran en [Codalab](https://competitions.codalab.org/competitions/25623), son alrededor de 10 mil publicaciones que provienen del [Civil Comments dataset](https://www.tensorflow.org/datasets/catalog/civil_comments) en las que se han etiquetado tramos específicos como lenguaje tóxico, la tarea es poder entrenar un modelo que detecte estos tramos tóxicos.

Para correr el modelo lo más sencillo es hacer un ambiente virtual:

```
virtualenv Toxic-Spans
```

Para activarlo:

```
cd Toxic-Spans
source bin/activate
```

Posteriormente clonar el repositorio e instalar los requerimientos:

```
git clone https://github.com/Mgczacki/Proyecto_PLN.git
cd Proyecto_PLN
pip install -r requirements.txt
```

La libreta [Results LSTM](https://github.com/Mgczacki/Proyecto_PLN/blob/main/Notebooks/Results%20LSTM.ipynb) contiene nuestros resultados con el mejor modelo que encontramos.
