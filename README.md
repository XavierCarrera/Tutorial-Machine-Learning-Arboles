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

![suma de errores cuadrados](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Arboles/blob/main/img/suma_errores_cuadrados.png?raw=true)

Aca la *f* (**x**) es la función representando nuestro árbol y cada punto está representado por variables predictoras **xi** y una variable resultante *yi*. En este sentido, para producir un buen árbol necesitamos un data set grande con valores resultantes conocidos que será nuestro set de entrenamiento.

Para construir entonces un árbol tenemos que asumir que la mejor división es aquella que más minimiza la SEC. Usemos el siguiente gráfico: 

![lineas sec](https://ds055uzetaobb.cloudfront.net/brioche/uploads/KBetrBRZQl-3-1-5.png?width=1200)

En este caso sabemos que la línea azul hace el mejor trabajo reduciendo la SEC por las siguientes razones:

* La línea roja puede ser descartada porque trata de dividir a través de variables resultantes, usando información para tratar de hacer estimaciones lo que lo hace una opción imposible.
* Tenemos que generar una tabla que tenga promedios a la derecha e izquierda de cada línea en su eje y. Por ejemplo para la línea verde:
    
        Valores a la izquierda en y: 8 + 6 + 4 + 2 / 4 = 5
        Valores a la derecha en y: 10 + 22 / 2 = 11
        
Al completar la tabla tenemos los siguientes promedios, sus SEC y la SEC total.

| Valor Divisor | Promedio Izquierda | Promedio Derecha | SEC Izquierda | SEC Derecha | Total |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
| 2 | 2  | 8  | 0  | 40  | 40  |
| 6 | 4 | 10  | 8  | 8 | 16  |
| 8 | 5  | 11  | 20  | 2  | 22  |

De esta manera, vemos que la línea divisora en *x* = 6 es la mejor ya que tiene el menor SEC.

Normalmente, dividir el plano nos ayuda a decrecer la SEC. En el ejemplo anterior podríamos tener hasta cinco divisiones, dado que el máximo de data points son seis. Es decir, siempre el máximo de secciones es la cantidad de datapoints en el eje *x* - 1. Sin embargo, hay que estar conscientes que este enfoque nos puede llevar a overfitting. Si añadimos suficientes hojas a nuestro árbol, podemos reducir el SEC hasta 0. Esta acción tiene efectos detrimentales:

* Se genera un sobreajuste y especialización en los datos de entrenamiento.
* Los errores en los datos que usamos para generar el árbol darán mucho peso a sus prediccionwa, dado que no hay valores con los cuales promediar.
* El desempeño con datos nuevos será malo.

La solución clásica a este problema es hacer crecer un árbol extensamente y luego cortar secciones hasta que se ajuste a nuestros deseos. Esto se hace con una función de costo que castiga la inexactitud y la complejidad. Si tomamos *yi* de cada data point y **xi** como la variable predictora, tenemos una *f* (**x**) en un subarbol que predice por *yi* y dentro del cual contiene un divisor *T* que se formularía asi:

![función de cost](https://github.com/XavierCarrera/Tutorial-Machine-Learning-Arboles/blob/main/img/funcion-costo.png?raw=true)

Aquí, α es un parámetro de tunneo que controla que tanto castigamos la complexidad. Si α fuese al infinito, el número de hojas cortadas serían solo una ya que el costo de divisón se vuelve infinito. Dado que la SEC siempre será finita, el número de divisiones se acercará a 0 y el número de hojas se acercará a 1. 

## Árboles de Clasificación

## Consideraciones sobre los Árboles de Decisión

## Bagging

## Boosting 
