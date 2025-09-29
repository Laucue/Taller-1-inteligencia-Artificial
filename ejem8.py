import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import base64
import io
from flask import Flask, request, render_template_string, url_for

class AlgoritmoNReinas:
    def __init__(self):
        self.n = 8
        self.tamaño_poblacion = 100
        self.generaciones_max = 500
        self.tasa_mutacion = 0.03
        self.metodo_seleccion = 'ruleta'
        self.k_torneo = 3
    
    # ========== REPRESENTACIÓN Y FITNESS ==========
    
    def generar_individuo(self, n=None):
        """Genera un individuo aleatorio (permutación)"""
        if n is None:
            n = self.n
        individuo = list(range(n))
        random.shuffle(individuo)
        return individuo
    
    def calcular_conflictos(self, individuo):
        """Calcula el número de conflictos (pares de reinas que se atacan)"""
        n = len(individuo)
        conflictos = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Misma columna
                if individuo[i] == individuo[j]:
                    conflictos += 1
                # Misma diagonal
                elif abs(i - j) == abs(individuo[i] - individuo[j]):
                    conflictos += 1
        
        return conflictos
    
    def fitness(self, individuo):
        """Función de fitness: máximo conflictos posibles - conflictos actuales"""
        n = len(individuo)
        max_conflictos = n * (n - 1) // 2
        return max_conflictos - self.calcular_conflictos(individuo)
    
    def fitness_detallado(self, individuo):
        """Devuelve fitness desglosado por tipo de conflicto"""
        n = len(individuo)
        conflictos_vertical = 0
        conflictos_diagonal = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if individuo[i] == individuo[j]:
                    conflictos_vertical += 1
                elif abs(i - j) == abs(individuo[i] - individuo[j]):
                    conflictos_diagonal += 1
        
        max_conflictos = n * (n - 1) // 2
        fitness_total = max_conflictos - (conflictos_vertical + conflictos_diagonal)
        
        return {
            'total': fitness_total,
            'vertical': conflictos_vertical,
            'diagonal': conflictos_diagonal,
            'maximo': max_conflictos
        }
    
    def calcular_diversidad(self, poblacion):
        """Calcula la diversidad genética de la población"""
        if not poblacion or len(poblacion) <= 1:
            return 0
        
        n = len(poblacion[0])
        diversidad_total = 0
        
        for gen in range(n):
            valores = [ind[gen] for ind in poblacion]
            # Calcular entropía de Shannon para este gen
            conteo = {val: valores.count(val) for val in set(valores)}
            total = len(valores)
            entropia = 0
            for count in conteo.values():
                p = count / total
                entropia -= p * np.log2(p) if p > 0 else 0
            diversidad_total += entropia
        
        return diversidad_total / n
    
    # ========== MÉTODOS DE SELECCIÓN ==========
    
    def seleccion_ruleta(self, poblacion, fitness_poblacion):
        """Selección por ruleta (proporcional al fitness)"""
        # Calcular probabilidades acumuladas
        fitness_total = sum(fitness_poblacion)
        if fitness_total == 0:  # Evitar división por cero
            return random.choices(poblacion, k=len(poblacion))
        
        probabilidades = [f / fitness_total for f in fitness_poblacion]
        prob_acumuladas = []
        acumulador = 0
        
        for prob in probabilidades:
            acumulador += prob
            prob_acumuladas.append(acumulador)
        
        # Seleccionar padres
        padres = []
        for _ in range(len(poblacion)):
            r = random.random()
            for i, prob in enumerate(prob_acumuladas):
                if r <= prob:
                    padres.append(poblacion[i])
                    break
        
        return padres
    
    def seleccion_torneo(self, poblacion, fitness_poblacion):
        """Selección por torneo"""
        padres = []
        
        for _ in range(len(poblacion)):
            # Seleccionar k individuos aleatorios
            torneo_indices = random.sample(range(len(poblacion)), self.k_torneo)
            # Elegir el mejor del torneo
            mejor_idx = max(torneo_indices, key=lambda i: fitness_poblacion[i])
            padres.append(poblacion[mejor_idx])
        
        return padres
    
    # ========== OPERADORES GENÉTICOS ==========
    
    def cruzamiento_orden(self, padre1, padre2):
        """Cruzamiento de orden para permutaciones"""
        n = len(padre1)
        i, j = sorted(random.sample(range(n), 2))
        
        hijo = [None] * n
        hijo[i:j+1] = padre1[i:j+1]
        
        # Completar con elementos del padre2 en orden
        pos = (j + 1) % n
        for k in range(n):
            elemento = padre2[(j + 1 + k) % n]
            if elemento not in hijo:
                hijo[pos] = elemento
                pos = (pos + 1) % n
        
        return hijo
    
    def mutacion_intercambio(self, individuo):
        """Mutación por intercambio de dos posiciones"""
        if random.random() < self.tasa_mutacion:
            i, j = random.sample(range(len(individuo)), 2)
            individuo[i], individuo[j] = individuo[j], individuo[i]
        return individuo
    
    # ========== INTERFAZ GRÁFICA DEL TABLERO ==========
    
    def visualizar_tablero_interactivo(self, individuo, generacion=None, fitness_info=None, conflicto_info=None, return_base64=False):
        """Muestra una interfaz gráfica del tablero con toda la información.
        Si return_base64=True devuelve una data URI PNG en lugar de mostrarla.
        """
        n = len(individuo)
        
        # Crear figura y ejes
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Crear tablero de ajedrez
        tablero = np.zeros((n, n))
        
        # Patrón de tablero de ajedrez
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 0:
                    tablero[i, j] = 1  # Casillas claras
        
        # Mostrar tablero
        ax.imshow(tablero, cmap='binary', extent=[0, n, 0, n])
        
        # Colocar las reinas en el tablero
        for fila, col in enumerate(individuo):
            # Dibujar la reina
            ax.text(col + 0.5, n - fila - 0.5, '♛', 
                   fontsize=30, ha='center', va='center', 
                   color='red' if self.esta_en_conflicto(individuo, fila, col) else 'red')
        
        # Configurar el gráfico
        ax.set_xticks(np.arange(0, n+1, 1))
        ax.set_yticks(np.arange(0, n+1, 1))
        ax.grid(True, color='black', linewidth=1)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect('equal')
        
        # Añadir información adicional
        info_text = f"Configuración: {individuo}\n"
        if generacion is not None:
            info_text += f"Generación: {generacion}\n"
        if fitness_info is not None:
            info_text += f"Fitness: {fitness_info['total']}/{fitness_info['maximo']}\n"
            info_text += f"Conflictos verticales: {fitness_info['vertical']}\n"
            info_text += f"Conflictos diagonales: {fitness_info['diagonal']}\n"
            info_text += f"Conflictos totales: {fitness_info['vertical'] + fitness_info['diagonal']}\n"
        
        # Añadir información de conflicto si está disponible
        if conflicto_info is not None:
            info_text += f"\nAnálisis de conflictos:\n"
            for tipo, detalles in conflicto_info.items():
                info_text += f"{tipo}: {detalles}\n"
        
        # Añadir texto de información
        ax.text(n + 1, n/2, info_text, fontsize=12, va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.title(f"Problema de las {n} Reinas", fontsize=16, fontweight='bold')
        plt.tight_layout()
        if return_base64:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            data = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{data}"
        else:
            plt.show()
    
    def esta_en_conflicto(self, individuo, fila, col):
        """Determina si una reina específica está en conflicto"""
        for i in range(len(individuo)):
            if i != fila:
                # Misma columna
                if individuo[i] == col:
                    return True
                # Misma diagonal
                if abs(i - fila) == abs(individuo[i] - col):
                    return True
        return False
    
    def analizar_conflictos_detallados(self, individuo):
        """Analiza los conflictos de manera detallada para mostrar en la interfaz"""
        n = len(individuo)
        conflictos_vertical = defaultdict(list)
        conflictos_diagonal = defaultdict(list)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Misma columna
                if individuo[i] == individuo[j]:
                    conflictos_vertical[individuo[i]].append((i, j))
                # Misma diagonal
                elif abs(i - j) == abs(individuo[i] - individuo[j]):
                    diagonal = (i + individuo[i], j + individuo[j])
                    conflictos_diagonal[diagonal].append((i, j))
        
        return {
            'vertical': dict(conflictos_vertical),
            'diagonal': dict(conflictos_diagonal)
        }
    
    # ========== ALGORITMO EVOLUTIVO CON MÉTRICAS AVANZADAS ==========
    
    def algoritmo_evolutivo_completo(self, modo='cli'):
        """Algoritmo evolutivo con métricas avanzadas - PARA AL ENCONTRAR SOLUCIÓN.
        modo: 'cli' para imprimir/mostrar con plt.show(); 'web' para devolver imágenes base64 sin prints.
        """
        # Inicializar población
        poblacion = [self.generar_individuo() for _ in range(self.tamaño_poblacion)]
        
        # Métricas avanzadas
        metricas = {
            'mejor_fitness': [],
            'promedio_fitness': [],
            'diversidad': [],
            'mejor_vertical': [],
            'mejor_diagonal': [],
            'convergencia': []  # Porcentaje de población con fitness similar al mejor
        }
        
        mejor_individuo = None
        mejor_fitness_global = -1
        generacion_solucion = None
        
        if modo == 'cli':
            print("Ejecutando algoritmo evolutivo...")

        imagen_solucion = None
        
        for generacion in range(self.generaciones_max):
            # Evaluación detallada
            evaluaciones = [self.fitness_detallado(ind) for ind in poblacion]
            fitness_poblacion = [e['total'] for e in evaluaciones]
            
            # Encontrar el mejor individuo
            mejor_idx = np.argmax(fitness_poblacion)
            mejor_individuo_actual = poblacion[mejor_idx]
            mejor_fitness = fitness_poblacion[mejor_idx]
            mejor_eval = evaluaciones[mejor_idx]
            
            # Actualizar mejor global
            if mejor_fitness > mejor_fitness_global:
                mejor_fitness_global = mejor_fitness
                mejor_individuo = mejor_individuo_actual.copy()
            
            # Calcular métricas avanzadas
            metricas['mejor_fitness'].append(mejor_fitness)
            metricas['promedio_fitness'].append(np.mean(fitness_poblacion))
            metricas['diversidad'].append(self.calcular_diversidad(poblacion))
            metricas['mejor_vertical'].append(mejor_eval['vertical'])
            metricas['mejor_diagonal'].append(mejor_eval['diagonal'])
            
            # Calcular convergencia (porcentaje de población con fitness cercano al mejor)
            if mejor_fitness > 0:
                fitness_umbral = mejor_fitness * 0.8  # 80% del mejor fitness
                poblacion_convergida = sum(1 for f in fitness_poblacion if f >= fitness_umbral)
                metricas['convergencia'].append(poblacion_convergida / len(poblacion))
            else:
                metricas['convergencia'].append(0)
            
            # CONDICIÓN DE PARADA MEJORADA: DETENERSE INMEDIATAMENTE AL ENCONTRAR SOLUCIÓN
            max_fitness_posible = self.n * (self.n - 1) // 2
            if mejor_fitness == max_fitness_posible:
                generacion_solucion = generacion + 1
                if modo == 'cli':
                    print(f"¡Solución perfecta encontrada en la generación {generacion_solucion}!")
                    # Mostrar interfaz gráfica de la solución
                    conflicto_info = self.analizar_conflictos_detallados(mejor_individuo)
                    self.visualizar_tablero_interactivo(
                        mejor_individuo, 
                        generacion_solucion, 
                        mejor_eval,
                        conflicto_info
                    )
                else:
                    conflicto_info = self.analizar_conflictos_detallados(mejor_individuo)
                    imagen_solucion = self.visualizar_tablero_interactivo(
                        mejor_individuo,
                        generacion_solucion,
                        mejor_eval,
                        conflicto_info,
                        return_base64=True
                    )
                
                # Añadir métricas finales y salir del bucle
                metricas['mejor_fitness'].append(mejor_fitness)
                metricas['promedio_fitness'].append(np.mean(fitness_poblacion))
                metricas['diversidad'].append(self.calcular_diversidad(poblacion))
                metricas['mejor_vertical'].append(mejor_eval['vertical'])
                metricas['mejor_diagonal'].append(mejor_eval['diagonal'])
                metricas['convergencia'].append(1.0)  # 100% convergencia
                
                break
            
            # Mostrar progreso cada 25 generaciones
            if modo == 'cli' and generacion % 25 == 0:
                print(f"Generación {generacion}: Mejor={mejor_fitness}, Prom={np.mean(fitness_poblacion):.1f}, "
                      f"Div={metricas['diversidad'][-1]:.3f}, Conv={metricas['convergencia'][-1]:.1%}")
            
            # Selección
            if self.metodo_seleccion == 'ruleta':
                padres = self.seleccion_ruleta(poblacion, fitness_poblacion)
            else:  # torneo
                padres = self.seleccion_torneo(poblacion, fitness_poblacion)
            
            # Cruzamiento
            hijos = []
            for i in range(0, len(padres), 2):
                if i + 1 < len(padres):
                    hijo1 = self.cruzamiento_orden(padres[i], padres[i+1])
                    hijo2 = self.cruzamiento_orden(padres[i+1], padres[i])
                    hijos.extend([hijo1, hijo2])
                else:
                    hijos.append(padres[i])
            
            # Mutación
            poblacion = [self.mutacion_intercambio(hijo) for hijo in hijos]
            
            # Elitismo: mantener el mejor individuo
            if mejor_individuo and mejor_individuo not in poblacion:
                peor_idx = min(range(len(poblacion)), key=lambda i: self.fitness(poblacion[i]))
                poblacion[peor_idx] = mejor_individuo.copy()
        
        # Si no se encontró solución perfecta, mostrar la mejor encontrada
        if generacion_solucion is None:
            conflicto_info = self.analizar_conflictos_detallados(mejor_individuo)
            if modo == 'cli':
                self.visualizar_tablero_interactivo(
                    mejor_individuo, 
                    self.generaciones_max, 
                    self.fitness_detallado(mejor_individuo),
                    conflicto_info
                )
            else:
                imagen_solucion = self.visualizar_tablero_interactivo(
                    mejor_individuo,
                    self.generaciones_max,
                    self.fitness_detallado(mejor_individuo),
                    conflicto_info,
                    return_base64=True
                )
        
        resultado = {
            'mejor_individuo': mejor_individuo,
            'mejor_fitness': mejor_fitness_global,
            'generacion_solucion': generacion_solucion,
            'metricas': metricas,
            'total_generaciones': generacion + 1
        }
        if modo == 'web':
            imagen_metricas = self.graficar_metricas_evolucion(metricas, return_base64=True)
            resultado['imagen_solucion'] = imagen_solucion
            resultado['imagen_metricas'] = imagen_metricas
        return resultado
    
    # ========== BÚSQUEDA ALEATORIA ==========
    
    def busqueda_aleatoria(self, max_intentos=10000, modo='cli'):
        """Búsqueda aleatoria para comparación. modo: 'cli' o 'web'"""
        mejor_individuo = None
        mejor_fitness = -1
        intentos_realizados = 0
        
        max_fitness = self.n * (self.n - 1) // 2
        
        for intento in range(max_intentos):
            individuo = self.generar_individuo()
            fit = self.fitness(individuo)
            
            if fit > mejor_fitness:
                mejor_fitness = fit
                mejor_individuo = individuo.copy()
            
            if fit == max_fitness:
                conflicto_info = self.analizar_conflictos_detallados(mejor_individuo)
                if modo == 'cli':
                    self.visualizar_tablero_interactivo(
                        mejor_individuo, 
                        None, 
                        self.fitness_detallado(mejor_individuo),
                        conflicto_info
                    )
                else:
                    imagen_solucion = self.visualizar_tablero_interactivo(
                        mejor_individuo,
                        None,
                        self.fitness_detallado(mejor_individuo),
                        conflicto_info,
                        return_base64=True
                    )
                break
            
            intentos_realizados = intento + 1
        
        # Si no se encontró solución perfecta, mostrar la mejor encontrada
        if mejor_fitness < max_fitness:
            conflicto_info = self.analizar_conflictos_detallados(mejor_individuo)
            if modo == 'cli':
                self.visualizar_tablero_interactivo(
                    mejor_individuo, 
                    None, 
                    self.fitness_detallado(mejor_individuo),
                    conflicto_info
                )
            else:
                imagen_solucion = self.visualizar_tablero_interactivo(
                    mejor_individuo,
                    None,
                    self.fitness_detallado(mejor_individuo),
                    conflicto_info,
                    return_base64=True
                )
        
        return {
            'mejor_individuo': mejor_individuo,
            'mejor_fitness': mejor_fitness,
            'intentos_realizados': intentos_realizados,
            'encontro_solucion': mejor_fitness == max_fitness,
            'imagen_solucion': imagen_solucion if 'imagen_solucion' in locals() else None
        }
    
    # ========== COMPARACIÓN SIMPLIFICADA ==========
    
    def comparar_metodos(self, ejecuciones=10, modo='cli'):
        """Compara algoritmo evolutivo vs búsqueda aleatoria. modo: 'cli' o 'web'"""
        if modo == 'cli':
            print(f"\n=== COMPARACIÓN: EVOLUTIVO vs ALEATORIO (N={self.n}) ===")
            print(f"Ejecuciones: {ejecuciones}")
        
        resultados_evolutivo = []
        resultados_aleatorio = []
        
        # Algoritmo evolutivo
        if modo == 'cli':
            print("\n--- ALGORITMO EVOLUTIVO ---")
            print(f"Configuración: Población={self.tamaño_poblacion}, Mutación={self.tasa_mutacion}, Selección={self.metodo_seleccion}")
        
        for i in range(ejecuciones):
            resultado = self.algoritmo_evolutivo_completo(modo=modo)
            resultados_evolutivo.append(resultado)
            if modo == 'cli':
                print(f"Ejecución {i+1}: ", end="")
                if resultado['generacion_solucion']:
                    print(f"Solución en generación {resultado['generacion_solucion']}")
                else:
                    print(f"Mejor fitness: {resultado['mejor_fitness']} (en {resultado['total_generaciones']} generaciones)")
        
        # Búsqueda aleatoria
        if modo == 'cli':
            print("\n--- BÚSQUEDA ALEATORIA ---")
        max_intentos = self.tamaño_poblacion * self.generaciones_max  # Mismo esfuerzo computacional aproximado
        
        for i in range(ejecuciones):
            resultado = self.busqueda_aleatoria(max_intentos, modo=modo)
            resultados_aleatorio.append(resultado)
            if modo == 'cli':
                print(f"Ejecución {i+1}: ", end="")
                if resultado['encontro_solucion']:
                    print(f"Solución en intento {resultado['intentos_realizados']}")
                else:
                    print(f"Mejor fitness: {resultado['mejor_fitness']} (en {resultado['intentos_realizados']} intentos)")
        
        # Estadísticas
        max_fitness = self.n * (self.n - 1) // 2
        
        # Algoritmo evolutivo
        exitos_evolutivo = sum(1 for r in resultados_evolutivo if r['mejor_fitness'] == max_fitness)
        if exitos_evolutivo > 0:
            generaciones_exitosas = [r['generacion_solucion'] for r in resultados_evolutivo if r['generacion_solucion']]
            generaciones_promedio = sum(generaciones_exitosas) / len(generaciones_exitosas)
            fitness_promedio_evolutivo = sum(r['mejor_fitness'] for r in resultados_evolutivo) / len(resultados_evolutivo)
        else:
            generaciones_promedio = self.generaciones_max
            fitness_promedio_evolutivo = sum(r['mejor_fitness'] for r in resultados_evolutivo) / len(resultados_evolutivo)
        
        # Búsqueda aleatoria
        exitos_aleatorio = sum(1 for r in resultados_aleatorio if r['encontro_solucion'])
        if exitos_aleatorio > 0:
            intentos_exitosos = [r['intentos_realizados'] for r in resultados_aleatorio if r['encontro_solucion']]
            intentos_promedio = sum(intentos_exitosos) / len(intentos_exitosos)
            fitness_promedio_aleatorio = sum(r['mejor_fitness'] for r in resultados_aleatorio) / len(resultados_aleatorio)
        else:
            intentos_promedio = max_intentos
            fitness_promedio_aleatorio = sum(r['mejor_fitness'] for r in resultados_aleatorio) / len(resultados_aleatorio)
        
        # Mostrar resultados comparativos
        if modo == 'cli':
            print(f"\n=== RESULTADOS COMPARATIVOS ===")
            print(f"\nALGORITMO EVOLUTIVO:")
            print(f"  Tasa de éxito: {exitos_evolutivo}/{ejecuciones} ({exitos_evolutivo/ejecuciones*100:.1f}%)")
            print(f"  Generaciones promedio hasta solución: {generaciones_promedio:.1f}")
            print(f"  Fitness promedio: {fitness_promedio_evolutivo:.1f}/{max_fitness}")
            
            print(f"\nBÚSQUEDA ALEATORIA:")
            print(f"  Tasa de éxito: {exitos_aleatorio}/{ejecuciones} ({exitos_aleatorio/ejecuciones*100:.1f}%)")
            print(f"  Intentos promedio hasta solución: {intentos_promedio:.1f}")
            print(f"  Fitness promedio: {fitness_promedio_aleatorio:.1f}/{max_fitness}")
        
        # Eficiencia relativa
        eficiencia_evolutivo = exitos_evolutivo / generaciones_promedio if exitos_evolutivo > 0 else 0
        eficiencia_aleatorio = exitos_aleatorio / intentos_promedio if exitos_aleatorio > 0 else 0
        
        if modo == 'cli':
            print(f"\nEFICIENCIA RELATIVA:")
            print(f"  Evolutivo: {eficiencia_evolutivo:.6f} (soluciones/generación)")
            print(f"  Aleatorio: {eficiencia_aleatorio:.6f} (soluciones/intento)")
            if eficiencia_aleatorio > 0:
                print(f"  Ventaja del evolutivo: {eficiencia_evolutivo/eficiencia_aleatorio:.1f}x")
            else:
                print(f"  Ventaja del evolutivo: ∞ (aleatorio no encontró soluciones)")
        
        # Gráfico comparativo
        imagen_comparacion = self.graficar_comparacion_simple(resultados_evolutivo, resultados_aleatorio, max_fitness, return_base64=(modo=='web'))
        
        return {
            'evolutivo': resultados_evolutivo,
            'aleatorio': resultados_aleatorio,
            'imagen_comparacion': imagen_comparacion if modo == 'web' else None,
            'resumen': {
                'exitos_evolutivo': exitos_evolutivo,
                'exitos_aleatorio': exitos_aleatorio,
                'generaciones_promedio': generaciones_promedio,
                'intentos_promedio': intentos_promedio,
                'fitness_promedio_evolutivo': fitness_promedio_evolutivo,
                'fitness_promedio_aleatorio': fitness_promedio_aleatorio,
                'eficiencia_evolutivo': eficiencia_evolutivo,
                'eficiencia_aleatorio': eficiencia_aleatorio
            }
        }
    
    # ========== VISUALIZACIÓN AVANZADA ==========
    
    def visualizar_tablero(self, individuo):
        """Visualiza el tablero con las reinas"""
        n = len(individuo)
        tablero = []
        
        for i in range(n):
            fila = ["."] * n
            # individuo[i] es la columna donde está la reina en la fila i
            fila[individuo[i]] = "Q"
            tablero.append(" ".join(fila))
        
        print("\nTablero solución:")
        for fila in tablero:
            print(fila)
    
    def graficar_metricas_evolucion(self, metricas, return_base64=False):
        """Grafica múltiples métricas de evolución. Si return_base64=True devuelve data URI PNG."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        generaciones = range(1, len(metricas['mejor_fitness']) + 1)
        max_fitness = self.n * (self.n - 1) // 2
        
        # Gráfico 1: Fitness
        ax1.plot(generaciones, metricas['mejor_fitness'], 'b-', label='Mejor Fitness', linewidth=2)
        ax1.plot(generaciones, metricas['promedio_fitness'], 'r--', label='Fitness Promedio', linewidth=2)
        ax1.axhline(y=max_fitness, color='g', linestyle='-', label='Fitness Máximo', alpha=0.7)
        ax1.set_xlabel('Generación')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Evolución del Fitness')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Diversidad y Convergencia
        ax2.plot(generaciones, metricas['diversidad'], 'purple', label='Diversidad', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(generaciones, metricas['convergencia'], 'orange', label='Convergencia', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Generación')
        ax2.set_ylabel('Diversidad', color='purple')
        ax2_twin.set_ylabel('Convergencia', color='orange')
        ax2.set_title('Diversidad y Convergencia')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Conflictos
        ax3.plot(generaciones, metricas['mejor_vertical'], 'red', label='Conflictos Verticales', linewidth=2)
        ax3.plot(generaciones, metricas['mejor_diagonal'], 'blue', label='Conflictos Diagonales', linewidth=2)
        ax3.set_xlabel('Generación')
        ax3.set_ylabel('Número de Conflictos')
        ax3.set_title('Evolución de Conflictos')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Progreso relativo
        ax4.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Umbral 90%')
        ax4.plot(generaciones, [f/max_fitness for f in metricas['mejor_fitness']], 
                'green', label='Progreso Relativo', linewidth=2)
        ax4.set_xlabel('Generación')
        ax4.set_ylabel('Progreso (0-1)')
        ax4.set_title('Progreso Relativo hacia Solución')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        plt.suptitle(f'Análisis Completo de la Evolución (N={self.n})', fontsize=16)
        plt.tight_layout()
        if return_base64:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            data = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{data}"
        else:
            plt.show()
    
    def graficar_comparacion_simple(self, resultados_evolutivo, resultados_aleatorio, max_fitness, return_base64=False):
        """Grafica comparación simple entre métodos. Si return_base64=True devuelve data URI PNG."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico 1: Tasas de éxito
        exitos_evolutivo = sum(1 for r in resultados_evolutivo if r['mejor_fitness'] == max_fitness)
        exitos_aleatorio = sum(1 for r in resultados_aleatorio if r['encontro_solucion'])
        
        metodos = ['Evolutivo', 'Aleatorio']
        tasas_exito = [exitos_evolutivo / len(resultados_evolutivo), 
                      exitos_aleatorio / len(resultados_aleatorio)]
        
        bars = ax1.bar(metodos, tasas_exito, color=['lightblue', 'lightcoral'])
        ax1.set_title('Tasa de Éxito')
        ax1.set_ylabel('Tasa de Éxito')
        ax1.set_ylim(0, 1)
        
        for bar, tasa in zip(bars, tasas_exito):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{tasa:.1%}', ha='center', va='bottom')
        
        # Gráfico 2: Fitness promedio
        fitness_evolutivo = sum(r['mejor_fitness'] for r in resultados_evolutivo) / len(resultados_evolutivo)
        fitness_aleatorio = sum(r['mejor_fitness'] for r in resultados_aleatorio) / len(resultados_aleatorio)
        
        bars = ax2.bar(metodos, [fitness_evolutivo, fitness_aleatorio], color=['lightblue', 'lightcoral'])
        ax2.axhline(y=max_fitness, color='g', linestyle='--', label=f'Máximo ({max_fitness})')
        ax2.set_title('Fitness Promedio')
        ax2.set_ylabel('Fitness')
        ax2.legend()
        
        for bar, fitness in zip(bars, [fitness_evolutivo, fitness_aleatorio]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{fitness:.1f}', ha='center', va='bottom')
        
        plt.suptitle(f'Comparación: Algoritmo Evolutivo vs Búsqueda Aleatoria (N={self.n})')
        plt.tight_layout()
        if return_base64:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            data = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{data}"
        else:
            plt.show()
    
    # ========== INTERFAZ DE CONFIGURACIÓN ==========
    
    def configurar_parametros(self):
        """Permite configurar los parámetros del algoritmo"""
        print("=== CONFIGURACIÓN DEL ALGORITMO EVOLUTIVO ===")
        
        self.n = int(input(f"Número de reinas (actual: {self.n}): ") or self.n)
        self.tamaño_poblacion = int(input(f"Tamaño de población (actual: {self.tamaño_poblacion}): ") or self.tamaño_poblacion)
        self.generaciones_max = int(input(f"Máximo de generaciones (actual: {self.generaciones_max}): ") or self.generaciones_max)
        self.tasa_mutacion = float(input(f"Tasa de mutación (actual: {self.tasa_mutacion}): ") or self.tasa_mutacion)
        
        metodo = input(f"Método de selección (ruleta/torneo) (actual: {self.metodo_seleccion}): ").lower()
        if metodo in ['ruleta', 'torneo']:
            self.metodo_seleccion = metodo
        
        if self.metodo_seleccion == 'torneo':
            self.k_torneo = int(input(f"Tamaño del torneo (actual: {self.k_torneo}): ") or self.k_torneo)
    
    def menu_principal(self):
        """Menú principal interactivo"""
        while True:
            print("\n" + "="*50)
            print("ALGORITMO EVOLUTIVO - PROBLEMA DE LAS N-REINAS")
            print("="*50)
            print(f"Configuración actual:")
            print(f"  Número de reinas (N): {self.n}")
            print(f"  Tamaño de población: {self.tamaño_poblacion}")
            print(f"  Generaciones máximas: {self.generaciones_max}")
            print(f"  Tasa de mutación: {self.tasa_mutacion}")
            print(f"  Método de selección: {self.metodo_seleccion}")
            if self.metodo_seleccion == 'torneo':
                print(f"  Tamaño torneo: {self.k_torneo}")
            print("\nOpciones:")
            print("1. Configurar parámetros")
            print("2. Ejecutar algoritmo evolutivo (con métricas avanzadas)")
            print("3. Ejecutar búsqueda aleatoria")
            print("4. Comparar evolutivo vs aleatorio")
            print("5. Salir")
            
            opcion = input("\nSeleccione una opción: ")
            
            if opcion == '1':
                self.configurar_parametros()
            
            elif opcion == '2':
                print(f"\nEjecutando algoritmo evolutivo con N={self.n}...")
                inicio = time.time()
                resultado = self.algoritmo_evolutivo_completo()
                tiempo = time.time() - inicio
                
                max_fitness = self.n * (self.n - 1) // 2
                print(f"\n=== RESULTADOS ===")
                print(f"Tiempo de ejecución: {tiempo:.2f} segundos")
                print(f"Mejor fitness encontrado: {resultado['mejor_fitness']}/{max_fitness}")
                
                if resultado['generacion_solucion']:
                    print(f"¡Solución encontrada en la generación {resultado['generacion_solucion']}!")
                    self.visualizar_tablero(resultado['mejor_individuo'])
                else:
                    print("No se encontró solución perfecta en el número máximo de generaciones.")
                    print(f"Mejor solución encontrada: {resultado['mejor_individuo']}")
                
                # Mostrar gráficos de métricas avanzadas
                self.graficar_metricas_evolucion(resultado['metricas'])
            
            elif opcion == '3':
                max_intentos = int(input(f"Máximo de intentos (default: 10000): ") or 10000)
                print(f"\nEjecutando búsqueda aleatoria con N={self.n}...")
                inicio = time.time()
                resultado = self.busqueda_aleatoria(max_intentos)
                tiempo = time.time() - inicio
                
                max_fitness = self.n * (self.n - 1) // 2
                print(f"\n=== RESULTADOS BÚSQUEDA ALEATORIA ===")
                print(f"Tiempo de ejecución: {tiempo:.2f} segundos")
                print(f"Intentos realizados: {resultado['intentos_realizados']}")
                print(f"Mejor fitness encontrado: {resultado['mejor_fitness']}/{max_fitness}")
                
                if resultado['encontro_solucion']:
                    print("¡Solución encontrada!")
                    self.visualizar_tablero(resultado['mejor_individuo'])
                else:
                    print("No se encontró solución perfecta en el número máximo de intentos.")
                    print(f"Mejor solución encontrada: {resultado['mejor_individuo']}")
            
            elif opcion == '4':
                ejecuciones = int(input("Número de ejecuciones para comparación (default: 5): ") or 5)
                self.comparar_metodos(ejecuciones)
            
            elif opcion == '5':
                print("¡Hasta luego!")
                break
            
            else:
                print("Opción no válida. Por favor, seleccione 1-5.")

# ========== EJECUCIÓN PRINCIPAL ==========

if __name__ == "__main__":
    algoritmo = AlgoritmoNReinas()
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def index():
        return render_template_string(
            """
            <html>
            <head>
                <title>N-Reinas - Interfaz Web</title>
                <style>
                    :root {
                        --bg-start: #0f172a; --bg-end: #1e293b; --card: #0b1220; --text: #e2e8f0;
                        --muted: #94a3b8; --primary: #22d3ee; --primary-600: #0ea5b7; --accent: #a78bfa;
                        --border: #1f2a44;
                    }
                    *{box-sizing:border-box}
                    body{margin:0;padding:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Helvetica Neue',Arial;color:var(--text);background:linear-gradient(135deg,var(--bg-start),var(--bg-end));min-height:100vh;display:flex;justify-content:center}
                    .container{width:100%;max-width:1000px;padding:32px 20px}
                    .header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}
                    .title{font-size:28px;font-weight:700}
                    .subtitle{color:var(--muted);margin-top:6px;font-size:14px}
                    .grid{display:grid;grid-template-columns:1fr;gap:16px}
                    @media(min-width:900px){.grid{grid-template-columns:2fr 1fr}}
                    .card{background:radial-gradient(1200px 600px at -10% -10%,rgba(34,211,238,.08),transparent 50%),radial-gradient(900px 500px at 110% 0,rgba(167,139,250,.06),transparent 50%),var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 18px 14px 18px;box-shadow:0 10px 30px rgba(0,0,0,.35),0 1px 0 rgba(255,255,255,.04) inset}
                    .card h2{margin:0 0 12px 0;font-size:18px}
                    .form-row{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
                    .form-group{display:flex;flex-direction:column;gap:6px}
                    label{font-size:12px;color:var(--muted)}
                    input[type=number],select{background:#0a1528;color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px 12px;height:40px;font-size:14px;outline:none;transition:border-color .2s,box-shadow .2s}
                    input[type=number]:focus,select:focus{border-color:var(--primary);box-shadow:0 0 0 3px rgba(34,211,238,.12)}
                    .actions{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap}
                    .btn{appearance:none;cursor:pointer;border:none;border-radius:10px;padding:10px 14px;font-weight:600;letter-spacing:.3px;color:#08121f;transition:transform .06s ease,filter .15s ease,box-shadow .15s ease;box-shadow:0 8px 20px rgba(34,211,238,.25)}
                    .btn:active{transform:translateY(1px)}
                    .btn-primary{background:linear-gradient(135deg,var(--primary),var(--primary-600));color:#062a2f}
                    .btn-secondary{background:linear-gradient(135deg,var(--accent),#6d28d9);color:#10052b}
                    .btn-outline{background:transparent;color:var(--text);border:1px solid var(--border);box-shadow:none}
                    .note{font-size:12px;color:var(--muted);margin-top:4px}
                    .footer{margin-top:16px;color:var(--muted);font-size:12px;text-align:center}
                    a{color:var(--primary);text-decoration:none}a:hover{text-decoration:underline}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div>
                            <div class="title">Algoritmo Evolutivo - N Reinas</div>
                            <div class="subtitle">Explora soluciones y compara métodos en tu navegador</div>
                        </div>
                    </div>
                    <div class="grid">
                        <div class="card">
                            <h2>Configuración</h2>
                            <form method="post" action="{{ url_for('run_evolutivo') }}">
                                <div class="form-row">
                                    <div class="form-group">
                                        <label>Número de reinas (N)</label>
                                        <input type="number" name="n" value="{{ n }}" min="4" max="50" required>
                                    </div>
                                    <div class="form-group">
                                        <label>Tamaño de población</label>
                                        <input type="number" name="tam_pob" value="{{ tam }}" min="10" max="1000" required>
                                    </div>
                                    <div class="form-group">
                                        <label>Generaciones máximas</label>
                                        <input type="number" name="gens" value="{{ gens }}" min="10" max="10000" required>
                                    </div>
                                    <div class="form-group">
                                        <label>Tasa de mutación</label>
                                        <input type="number" name="mut" value="{{ mut }}" min="0" max="1" step="0.001" required>
                                    </div>
                                    <div class="form-group">
                                        <label>Método de selección</label>
                                        <select name="metodo">
                                            <option value="ruleta" {{ 'selected' if metodo=='ruleta' else '' }}>Ruleta</option>
                                            <option value="torneo" {{ 'selected' if metodo=='torneo' else '' }}>Torneo</option>
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label>Tamaño del torneo</label>
                                        <input type="number" name="k" value="{{ k }}" min="2" max="10">
                                        <div class="note">Solo aplica si el método es torneo</div>
                                    </div>
                                </div>
                                <div class="actions">
                                    <button type="submit" class="btn btn-primary">Ejecutar Evolutivo</button>
                                    <button form="form-aleatorio" class="btn btn-secondary" type="submit">Búsqueda Aleatoria</button>
                                    <button form="form-comparar" class="btn btn-outline" type="submit">Comparar Métodos</button>
                                </div>
                            </form>
                        </div>
                        <div class="card">
                            <h2>Acciones rápidas</h2>
                            <form id="form-aleatorio" method="post" action="{{ url_for('run_aleatorio') }}">
                                <div class="form-group">
                                    <label>Máximo de intentos (aleatorio)</label>
                                    <input type="number" name="max_intentos" value="10000" min="100" max="1000000">
                                </div>
                            </form>
                            <form id="form-comparar" method="post" action="{{ url_for('run_comparar') }}">
                                <div class="form-group" style="margin-top:10px;">
                                    <label>Ejecuciones para comparación</label>
                                    <input type="number" name="runs" value="5" min="1" max="50">
                                </div>
                            </form>
                        </div>
                    </div>
                    <div class="footer">Hecho con ❤ para N-Reinas</div>
                </div>
            </body>
            </html>
            """,
            n=algoritmo.n, tam=algoritmo.tamaño_poblacion, gens=algoritmo.generaciones_max,
            mut=algoritmo.tasa_mutacion, metodo=algoritmo.metodo_seleccion, k=algoritmo.k_torneo
        )

    def actualizar_parametros_desde_form(algoritmo: AlgoritmoNReinas, form):
        algoritmo.n = int(form.get('n', algoritmo.n))
        algoritmo.tamaño_poblacion = int(form.get('tam_pob', algoritmo.tamaño_poblacion))
        algoritmo.generaciones_max = int(form.get('gens', algoritmo.generaciones_max))
        algoritmo.tasa_mutacion = float(form.get('mut', algoritmo.tasa_mutacion))
        metodo = form.get('metodo', algoritmo.metodo_seleccion)
        if metodo in ['ruleta', 'torneo']:
            algoritmo.metodo_seleccion = metodo
        if algoritmo.metodo_seleccion == 'torneo':
            algoritmo.k_torneo = int(form.get('k', algoritmo.k_torneo))

    @app.route('/run/evolutivo', methods=['POST'])
    def run_evolutivo():
        actualizar_parametros_desde_form(algoritmo, request.form)
        inicio = time.time()
        resultado = algoritmo.algoritmo_evolutivo_completo(modo='web')
        tiempo = time.time() - inicio
        max_fitness = algoritmo.n * (algoritmo.n - 1) // 2
        return render_template_string(
            """
            <html>
            <head>
                <title>Resultado Evolutivo</title>
                <style>
                    body { margin:0; background: linear-gradient(135deg, #0f172a, #1e293b); color:#e2e8f0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; }
                    .container { max-width: 1000px; margin: 0 auto; padding: 24px 18px; }
                    .back { margin-bottom: 12px; }
                    a { color: #22d3ee; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                    .card { background: #0b1220; border: 1px solid #1f2a44; border-radius: 14px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
                    .row { display: grid; grid-template-columns: 1fr; gap: 14px; }
                    @media (min-width: 900px) { .row { grid-template-columns: 1fr 1fr; } }
                    h2 { margin: 6px 0 12px 0; font-size: 22px; }
                    .meta { color:#94a3b8; margin-bottom: 8px; }
                    img { width: 100%; border-radius: 10px; border:1px solid #1f2a44; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="back"><a href="{{ url_for('index') }}">&larr; Volver</a></div>
                    <div class="card">
                        <h2>Resultado Evolutivo</h2>
                        <div class="meta">Tiempo: {{ '%.2f' % tiempo }} s · Mejor fitness: {{ resultado['mejor_fitness'] }}/{{ max_fitness }}</div>
                        {% if resultado['generacion_solucion'] %}
                            <div class="meta">Solución en generación {{ resultado['generacion_solucion'] }}</div>
                        {% else %}
                            <div class="meta">No se encontró solución perfecta.</div>
                        {% endif %}
                        <div class="row">
                            {% if resultado['imagen_solucion'] %}
                            <div>
                                <h3>Tablero</h3>
                                <img src="{{ resultado['imagen_solucion'] }}" />
                            </div>
                            {% endif %}
                            {% if resultado['imagen_metricas'] %}
                            <div>
                                <h3>Métricas</h3>
                                <img src="{{ resultado['imagen_metricas'] }}" />
                            </div>
                            {% endif %}
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """,
            resultado=resultado, tiempo=tiempo, max_fitness=max_fitness
        )

    @app.route('/run/aleatorio', methods=['POST'])
    def run_aleatorio():
        max_intentos = int(request.form.get('max_intentos', 10000))
        inicio = time.time()
        resultado = algoritmo.busqueda_aleatoria(max_intentos=max_intentos, modo='web')
        tiempo = time.time() - inicio
        max_fitness = algoritmo.n * (algoritmo.n - 1) // 2
        return render_template_string(
            """
            <html>
            <head>
                <title>Resultado Aleatorio</title>
                <style>
                    body { margin:0; background: linear-gradient(135deg, #0f172a, #1e293b); color:#e2e8f0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; }
                    .container { max-width: 900px; margin: 0 auto; padding: 24px 18px; }
                    .back { margin-bottom: 12px; }
                    a { color: #22d3ee; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                    .card { background: #0b1220; border: 1px solid #1f2a44; border-radius: 14px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
                    h2 { margin: 6px 0 12px 0; font-size: 22px; }
                    .meta { color:#94a3b8; margin-bottom: 8px; }
                    img { width: 100%; border-radius: 10px; border:1px solid #1f2a44; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="back"><a href="{{ url_for('index') }}">&larr; Volver</a></div>
                    <div class="card">
                        <h2>Búsqueda Aleatoria</h2>
                        <div class="meta">Tiempo: {{ '%.2f' % tiempo }} s · Intentos: {{ resultado['intentos_realizados'] }} · Mejor fitness: {{ resultado['mejor_fitness'] }}/{{ max_fitness }}</div>
                        {% if resultado['encontro_solucion'] %}
                            <div class="meta">¡Solución encontrada!</div>
                        {% else %}
                            <div class="meta">No se encontró solución perfecta.</div>
                        {% endif %}
                        {% if resultado['imagen_solucion'] %}
                            <h3>Tablero</h3>
                            <img src="{{ resultado['imagen_solucion'] }}" />
                        {% endif %}
                    </div>
                </div>
            </body>
            </html>
            """,
            resultado=resultado, tiempo=tiempo, max_fitness=max_fitness
        )

    @app.route('/run/comparar', methods=['POST'])
    def run_comparar():
        runs = int(request.form.get('runs', 5))
        inicio = time.time()
        resultado = algoritmo.comparar_metodos(ejecuciones=runs, modo='web')
        tiempo = time.time() - inicio
        return render_template_string(
            """
            <html>
            <head>
                <title>Comparación</title>
                <style>
                    body { margin:0; background: linear-gradient(135deg, #0f172a, #1e293b); color:#e2e8f0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; }
                    .container { max-width: 1000px; margin: 0 auto; padding: 24px 18px; }
                    .back { margin-bottom: 12px; }
                    a { color: #22d3ee; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                    .card { background: #0b1220; border: 1px solid #1f2a44; border-radius: 14px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
                    h2 { margin: 6px 0 12px 0; font-size: 22px; }
                    .meta { color:#94a3b8; margin-bottom: 8px; }
                    ul { line-height: 1.8; }
                    img { width: 100%; border-radius: 10px; border:1px solid #1f2a44; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="back"><a href="{{ url_for('index') }}">&larr; Volver</a></div>
                    <div class="card">
                        <h2>Comparación Evolutivo vs Aleatorio</h2>
                        <div class="meta">Tiempo: {{ '%.2f' % tiempo }} s</div>
                        <ul>
                            <li>Éxitos (Evolutivo): {{ resultado['resumen']['exitos_evolutivo'] }}</li>
                            <li>Éxitos (Aleatorio): {{ resultado['resumen']['exitos_aleatorio'] }}</li>
                            <li>Generaciones promedio (éxitos): {{ '%.1f' % resultado['resumen']['generaciones_promedio'] }}</li>
                            <li>Intentos promedio (éxitos): {{ '%.1f' % resultado['resumen']['intentos_promedio'] }}</li>
                            <li>Fitness promedio (Evolutivo): {{ '%.1f' % resultado['resumen']['fitness_promedio_evolutivo'] }}</li>
                            <li>Fitness promedio (Aleatorio): {{ '%.1f' % resultado['resumen']['fitness_promedio_aleatorio'] }}</li>
                            <li>Eficiencia Evolutivo: {{ '%.6f' % resultado['resumen']['eficiencia_evolutivo'] }}</li>
                            <li>Eficiencia Aleatorio: {{ '%.6f' % resultado['resumen']['eficiencia_aleatorio'] }}</li>
                        </ul>
                        {% if resultado['imagen_comparacion'] %}
                            <h3>Gráfico Comparativo</h3>
                            <img src="{{ resultado['imagen_comparacion'] }}" />
                        {% endif %}
                    </div>
                </div>
            </body>
            </html>
            """,
            resultado=resultado, tiempo=tiempo
        )

    app.run(host='127.0.0.1', port=5000, debug=False)