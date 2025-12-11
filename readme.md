# OpenCL CNN Pipeline para FPGA

Este proyecto implementa un pipeline para una **Red Neuronal Convolucional (CNN)** utilizando **OpenCL**, optimizado para su ejecución en **FPGA**. El repositorio contiene tanto el código host (CPU) como el código kernel (FPGA).

## Sobre el Proyecto

El objetivo principal de este proyecto es acelerar el proceso de inferencia de una CNN ligera (En este caso ULFGFD) en un sistema SoC compuesto por un HPS y una FPGA (En este caso DE10-Nano), usando OpenCL para describir el hardware.

Lo primero fue cuantizar el modelo a int8 para que fuera más eficiente implementarlo en hardware, principalmente por la limitada BRAM de la FPGA, tambien porque podemos empaquetar 2 operaciones MACs en 1 solo DSP.


---

## Estructura del Repositorio

El proyecto se divide en dos componentes principales:

* **/host**: Contiene el código C/C++ que se ejecuta en la CPU.
    * Responsable de la inicialización de la plataforma OpenCL.
    * Gestión de la comunicación y transferencia de datos con el dispositivo (FPGA).
    * Captura y preprocesamiento de frames con una webcam.
    * Ejecución del modelo onnx usando la libreria de onnxruntime. (Las 3 primeras etapas se ejecutan en la FPGA)
* **/fpga**: Contiene los kernels de OpenCL (`.cl`) que se sintetizan para la FPGA.
    * Lectura y escritura de los datos.
    * Convolución
    * Convolución Depthwise separable
    * Convolución Pointwise
* **/modelos**: Contiene los modelos usados:
    * modelo_int8 -> El modelo cuantizado a INT8
    * modelo_int8_cut -> Es la version cuantizada sin la primera capa, ya que la entrada es en int8
    * modelo_original -> Es la version original en FP32
    * modelo_simplificado -> La version con el BN fusionado con el Conv
    * modelo_sin_X -> Son los modelos que no tienen las capas aceleradas en la FPGA
* **/scripts**:  Contiene los scripts de python usados para modificar el Modelo
    * cutgraph -> Elimina la primera capa para que la entrada sea INT8
    * extract_data -> Extrae los pesos y los bias para guardarlos en un archivo .h
    * main -> Ejecuta el modelo en python
    * renombrar -> Es para darle nombres a las capas que no cuantizaremos
    * cuantizar -> Usa el dataset de validacion para cuantizar el modelo de FP32 a INT8
    * simplify -> Fusiona las capas BN con las Conv
    * test -> Muestra por pantalla el primer canal de la capa de convolucion que queramos para compararlo con la simulacion             de OpenCL

---

## Tecnologías Utilizadas

* **Lenguajes**: C, C++, OpenCL
* **Framework**: OpenCL, onnx
* **Hardware (Target)**: SoC DE10-Nano
* **Herramientas**: aoc windows, aocl linux, onnxruntime, opencv

---

## Ejecución

Para compilar y ejecutar este proyecto, necesitarás tener el entorno de desarrollo de OpenCL para tu FPGA instalado y configurado.

### Prerrequisitos

* SDK de OpenCL del fabricante (Solo lo he probado con la version 18.1 de intel)
* Drivers de la FPGA
* Onnxrutime
* Opencv

### Compilación y Ejecución

1.  **Compilar el Host:**
    ```bash
    ./build.sh
    ```

2.  **Compilar el Kernel (FPGA):**


    ```bash
    aoc pipeconv.cl
    ```

3.  **Ejecutar el Pipeline:**
    ```bash
    ./pipeonnx modelo.onnx opencl
    ```

---

## Resultados

Latencias

![Grafico de latencias](/gifs/Grafico.png)

Estas son las latencias medias de 100 frames:


| Modelo | Latencia (ms) | Speedup |
| ---------- | ---------- | ---------- |
| Float32  | 497,62  | 1 |
| Int8  | 274,5  | 1,81 |
| OpenCL FPGA (3 Layers) + ARM  | 198,06  | 2,51 |

### Base Float32
![Base F32](/gifs/Base.gif)

### Opencl Int8 primeras 3 convoluciones
![Base F32](/gifs/Opencl.gif)
