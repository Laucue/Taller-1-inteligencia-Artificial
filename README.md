# Taller-1-inteligencia-Artificial

# Algoritmo Evolutivo para el Problema de las N Reinas

Se hizo una interfaz en Flask que resuelve el problema clásico de las N Reinas utilizando un algoritmo evolutivo genético, en donde se muestra el tablero del ajedrez y unas graficas que describen el comportamiento de la sollución.

## Descripción del Problema

El problema de las N Reinas consiste en colocar N reinas en un tablero de ajedrez de N×N de manera que ninguna reina pueda atacar a otra. Esto significa que no puede haber dos reinas en la misma fila, columna o diagonal.

### Características del Algoritmo Evolutivo:
- **Representación**: Cada individuo es una permutación de N elementos (representando las posiciones de las reinas)
- **Población**: 100 individuos
- **Selección**: Torneo de tamaño 3 o mas
- **Mutación**: Intercambio de dos posiciones aleatorias 
- **Elitismo**: Mantiene los 10 mejores individuos en cada generación
- **Función de Fitness**: Número de conflictos (objetivo: minimizar a 0)

### Parámetros Configurables:
  - Numero de reinas
  - Tamaño de poblacion
  - Generaciones
  - Tasa de mutacion
  - Metodo de seleccion
  - k del torneo 

### Lenguaje
- Python 

### Algoritmo
- **Búsqueda evolutiva**: Encuentra soluciones óptimas usando principios genéticos
- **Convergencia inteligente**: Se detiene automáticamente al encontrar una solución perfecta
- **Robustez**: Maneja casos sin solución perfecta mostrando la mejor aproximación

### Algoritmo Genético:
1. **Inicialización**: Genera población aleatoria de permutaciones
2. **Evaluación**: Calcula fitness basado en conflictos
3. **Selección**: Aplica torneo para seleccionar padres
4. **Cruzamiento**: Order Crossover para generar descendencia
5. **Mutación**: Intercambio aleatorio con probabilidad fija
6. **Reemplazo**: Mantiene elite y reemplaza resto de población

### Optimizaciones:
- **Detección temprana**: Se detiene al encontrar solución perfecta
- **Elitismo**: Preserva mejores soluciones entre generaciones
- **Crossover eficiente**: Order Crossover mantiene restricciones del problema

## Rendimiento
- **Velocidad**: Típicamente encuentra soluciones en segundos para N≤12
- **Escalabilidad**: Funciona eficientemente hasta N=20
- **Convergencia**: Alta probabilidad de encontrar solución perfecta para N≤16


