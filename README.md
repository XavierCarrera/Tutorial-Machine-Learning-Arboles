# Tutorial Machine Learning: Árboles

## 1. Árboles de Regresión

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

## 2. Árboles de Clasificación

Uno de los usos más comúnes de los árboles de decisión es la clasificación. Esto es algo común, por ejemplo, en las clases de biología donde tenemos la manera en que se clasifican los seres vivos dependiendo del reino al que pertenecen. Estas son simples clasificaciones en donde las ramas se dispersan dependiendo de los observaciones. Por ejemplo:

![arbol clasificacion](https://ds055uzetaobb.cloudfront.net/brioche/uploads/Y5enq1rj9n-3-2-1.png?width=1500)

Así como en los árboles de regresión, los árboles de clasificación están construido con un conjunto de datos. Se usa para decidir que divisiones usar y como clasificar datos en las hojas al terminar el árbol. A cada hoja se le asigna cada punto que termina en cada clase, concentrándose en la clase más común.

Veamos la siguiente clasificación:

![arbol clasificacion](https://ds055uzetaobb.cloudfront.net/brioche/uploads/FHi7WhIa11-sj6potzffz-3-2-3.png?width=1200)

En este caso, tenemos tres categorías: azul, rojo y verde. El punto negro es un data point de clase desconocida. Si tuvieramos que clasificarlo, lo pondríamos dentro de la clase verde porque la mayoría de puntos en esta sección son verdes.

El proceso para crear árboles de clasificación es también recursivo. Empezamos definiendo una métrica que represente la calidad de la hoja en el árbol y seleccionar repetidamente la división que más mejor esta métrica. En este caso, empero, no podemos medir la SEC porque estamos tratando con datos cualitativos. Pero tenemos la alternativa de usar clases *k* a los que se les definen funciones de error *m*. Las funciones *m* más conocidas son:

* Tasa de Error Clasificatorio: la fracción de puntos mal representados.
* Índice Gini: que varía según el nivel de predicción de cada hoja. 
* Entropia Cruzada: similar al Índice Gini, sin embargo se enfoca a medir el "ruido" en cada una de las hojas.

Cada una de estas utiliza *Pmk* como la proporción de puntos de clase *k* en una hoja *m*. 

Si utilizacemos la Tasa de Error, podriamos formular una ecuación en donde N1 y N2 representan las hojas finales, mientras que E1 y E2 representan los errores al clasificar:

    (N1 ⋅ E1) + (N2 ⋅ E2) / N1 + N2
    
Con esto tendríamos el promedio del peso de errores en contraposición al número de puntos. 

Sin embargo, el Índice Gini y la Entropia Cruzada son más usados para construir ábroles de decisión. Aunque suena contraintuitivo, la tasa de error solo se enfoca en reducir el error en los árboles de decisión. Tenemos que recordar que los árboles de decisión no toman en cuenta lo que sucede más adelante. En cambio, solo buscan solucionar un problema que sucede en el presente. Por ello, al usar el Índice Gini o la Entropía Cruzada aumentamos las posibilidades de crear divisiones que funcionen bien en el futuro.

Una forma de visualizar como el Índice Gini elimina el ruido es con la siguiente gráfica:

![indice gini](https://ds055uzetaobb.cloudfront.net/brioche/uploads/umSezzLgNY-3-2-7.png?width=1200)

Sin necesidad de usar fórmulas podemos ver que la línea azul en el eje *x* fue creada a partir de una tasa de error porque se enfocó a buscar un punto en donde se encontraban la mayoría de puntos rojos y azules. Sin embargo, podemos saber que la línea roja en el eje *y* fue creada con un Índice Gini porque redujo el ruido de todos los puntos azules arriba de *y* = 10.

Al igual que sus pares en regresión, los árboles de clasificación caen fácilmente en overfitting. En esta sección del tuturoial hemos tratado al error de nuestros árboles como la proporción de puntos del data set de entrenamiendo que clasificó mal. Podemos recortar el árbolde varias maneras basado en esta función de error. Algunas soluciones puedes ser 1) añadir el total del número de hojas a la función de error e intentar minimizar la cantidad resultante o 2) cortar individualmente las hojas cuando sea necesario. 

Tomemos el siguiente escenario:

![entropia](https://ds055uzetaobb.cloudfront.net/brioche/uploads/aZICzaLJtX-3-2-8.png?width=1200)

Si quisieramos mezclar algunas de las secciones para evitar overfitting, deberíamos hacerlo con E y F. La razón es que dado que F tiene solo un punto azúl más y al combinarlo con E (que tiene mayormente puntos azules) solo causiaríamos una mala clasificación. La clave es contar la cantidad de puntos azules y rojos que habrían en otras combinaciones.

Pero ¿que sucedería si ignoraramos los datos de entrenamiento y crearamos un árbol arbitrariamente? ¿la tasa de error incrementaría con los datos de entrenamiento? 

Hay que notar que la hoja original solo podrá dar puntos dentro de una clasificación, por lo que los puntos en la mayoría de una clase serán clasificados correctamente. Si tenemos una clase azul (mayoritaria) y otra roja (minoritaria), cada punto crearía dos clases nuevas. Si la hoja tiene más puntos azules que rojos, entonces nada cambia. Cada punto será clasificado correctamente solo si fue clasificado antes correctamente. Sin embargo, si contiene más puntos rojos que azules, entonces lo contrario será cierto. Dado que hay más puntos rojos que azules en cada hoja, el número total será clasificado correctamente con mayor frecuencia. Por tanto, el error se quedará estático o decrecerá. 

## 3. Consideraciones sobre los Árboles de Decisión

Una de las grandes ventajas de los árboles de decisión es que son muy visuales en comparación con otros algoritmos de Machine Learning. Esto hace que sean fácil de interpretar. Y aunque los árboles de decisiones no suelen ser los más precisos al hacer predicciones, nos ayuda a encontrar interacciones entre variables dados su nivles de importancia. 

Además, un árbol de decisión funciona bien para casí cualquier problema. Rara vez suele ser la mejor solución, pero se puede usar para casi cualquier problema. 

Otra ventaja es que suele ser una buena solución cuando estamos trabajando con variables cualitativas. Muchos otros algoritmos tienen que ajustarse para lidiar con estos problemas. Mientras tanto, los árboles de decisiones puede trabajar directamente con este tipo de problemas. Con esto podemos evitar hacer suposiciones problemáticas que afecten nuestro modelado como asumir que puede haber un valor numérico que describa diferentes clases. 

Sin embargo y aunque su interpretación es sencilla, existen limitaciones obvias a la hora de utilizar este enfoque. Rara vez podemos encontrar un fenómeno que pueda ser definido en bloques. Es por esta razón que los árboles de decisión no son la mejor opción para hacer predicciones. Un ejemplo es la siguiente imagen en la que es obvio que tenemos dos clases que pueden ser sencillamente divididos por una función linear. Sin embargo, un árbol de decisión tendría muchos problemas para este tipo de tareas.

![Problema arbol decision](https://ds055uzetaobb.cloudfront.net/brioche/uploads/X8198YzjGf-3-3-3.png?width=1200)

Siempre que usemos árboles de decisión, tenemos que recordar que su desempeño con la data de entrenamiento no necesariamente con los datos de la vida real. En especial porque los árboles son especialmente susceptibles a sufrir de sobreajustes. Es por esta razón que para este tipo de algoritmos solemos **dividir la data en entrenamiento y pruebas**. Los datos de entrenamiento sirven justo para lo que su nombre sugiere y con los de prueba evaluamos su desempeño. Debido a que estos son datos que el modelo no ha procesado, podemos saber que también es nuestro árbol de decisiones.

## 4. Bagging

## 5. Boosting 
