# Agente Inteligente para Snake Modificado (DQN)

Proyecto final de curso para la asignatura **SIS-252 Inteligencia Artificial** (UCB Sede Cochabamba). 

## Objetivo del Proyecto
Desarrollar un agente inteligente capaz de navegar y sobrevivir de forma autónoma en una variación progresiva del juego "Snake". A diferencia de los entornos estáticos, el agente debe equilibrar la recolección de objetivos (manzanas) con la evasión de obstáculos dinámicos (su propio cuerpo en crecimiento) a lo largo de múltiples niveles de dificultad.

El sistema de toma de decisiones está basado en **Aprendizaje por Refuerzo (Deep Q-Learning)** utilizando Redes Neuronales Feedforward, tomando como *baseline* la arquitectura de resolución de laberintos de Zafrany y expandiéndola para entornos dinámicos y penalizaciones complejas.

## Equipo de Desarrollo
* Javier Antezana Vallejos
* Edson Lamas Alcocer
* Mateo Wilson Quispe Aylas
* Isabel Hidalgo Avila

## Estructura del Repositorio
El código está diseñado bajo el Principio de Responsabilidad Única (SRP), separando el entorno físico de la inteligencia artificial:

* `core/`: Motor del juego y física de la matriz (`snake_env.py`), y el Replay Buffer (`experience.py`).
* `agent/`: Definición de la arquitectura de la Red Neuronal con TensorFlow/Keras (`dqn_model.py`).
* `training/`: Bucle principal de entrenamiento y función de recompensas (`train.py`).
* `levels/`: Archivos JSON que definen la topología, obstáculos y metas de cada nivel.
* `utils/`: Configuraciones globales e hiperparámetros (`config.py`) y generación de gráficas de rendimiento (`plotting.py`).

## Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd snake_ai_project

2. **Se Recomienda crear un entorno virtual**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

Y acto seguido instalar las dependencias:
    ```bash   
    pip install -r requirements.txt
    ```

## USO 

**python main.py**
