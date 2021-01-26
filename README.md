# Tutorial Machine Learning: Árboles

## Árboles de Regresión

En tutoriales pasados hemos visto diferentes formas en las que podemos predecir datos futuros que son desconocidos. Y aunque [regresión lineal](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Regresion-Lineal) y [clasificación lineal](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Clasificacion-Lineal) son poderosos para solucionar ciertos problemas, pueden tener problemas en situaciones muy específicas.

Supongamos que queremos categorizar frutas por su dulzura. Para efectos de simpleza, solo distinguiremos entre naranjas y manzanas. Podemos en principio asignar pesos a un marcador de dulzura y usar regresión lineal para nuestras predicciones o catalogar cada fruta dentro de un plano usando KNN. Sin embargo, estos métodos fallarían de entrada al categorizar naranjas y manzanas. Aunque esto parece una obviedad, otros modelos lineales no empiezan categorizando. 

En este punto, hay distinciones importantes entre casos por lo que los árboles de regresión son bastante útiles. Los árboles funcionan recursivamente dividiendo los datos en muchas categorias y luego asignando valores a cada data point que cae en cierta area. Aunque no es necesairamente el método más preciso, son una buena alternativa cuando lidiamos con versatilidad e interpretabilidad. 

Cuando decimos que los árboles son generados recursivamente, nos referimos a que se dividen continuamente en el espacio donde están nuestros datos. Si hay límites que una simple igualdad, entonces no es viable utilizar árboles.

El punto final de un árbol son las hojas. Cada data point localizado en una hoja tiene un valor, que es el promedio de las hojas conocidas. Por ejemplo:

![árbol de decisión](https://ds055uzetaobb.cloudfront.net/brioche/uploads/1wYhnS7K24-3-1-2.png?width=1200)

Suponiendo que la dulzura se mide según el nivel de sucrosa en un rando de 0 a 1, el valor en cada hoja es el promedio de los valores conocidas en ella. Todo junto representa los datos de entrenamiento que utilizaremos para generar un árbol de decisiones.

Esto también tiene repercusiones en el plano donde graficamos nuestros data points. Miremos el siguiente gráfico:

![gráfico aŕbol decisión](https://ds055uzetaobb.cloudfront.net/brioche/uploads/TUZtEraFNk-3-1-4.png?width=1200)

Si tuvieramos un valor *x* = 5 al que quisieramos ubicar, tendríamos solamente seguir nuestro árbol. En este caso, *x* es mayor que 4 pero menor que *y*. Por tanto, sabemos que debe de caer en la zona azul. Para saber su valor en *y*, solo tenemos que sacar el promedio del resto de valores. Es decir:

    y = 3 + 3 + 3 + 11 / 4 = 5
    
Para hacer el cálculo, tenemos que usar la fórmula de **Suma de Errores Cuadrados** (SEC) que vimos en [regersión linear](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Regresion-Lineal) que se ve de la siguiente forma:

![suma de errores cuadrados](sec)

Aca la *f* (**x**) es la función representando nuestro árbol y cada punto está representado por variables predictoras **xi** y una variable resultante *yi*. En este sentido, para producir un buen árbol necesitamos un data set grande con valores resultantes conocidos que será nuestro set de entrenamiento.

Para construir entonces un árbol tenemos que asumir que la mejor división es aquella que más minimiza la SEC. Usemos el siguiente gráfico: 

![lineas sec](https://ds055uzetaobb.cloudfront.net/brioche/uploads/KBetrBRZQl-3-1-5.png?width=1200)

En este caso sabemos que la línea azul hace el mejor trabajo reduciendo la SEC por las siguientes razones:

* La línea roja puede ser descartada porque trata de dividir a través de variables resultantes, usando información para tratar de hacer estimaciones lo que lo hace una opción imposible.

## Árboles de Clasificación

## Consideraciones sobre los Árboles de Decisión

## Bagging

## Boosting 
