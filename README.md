# Trabajo Especial
## Evaluación de diferentes aproximaciones a la anonimización de textos médicos en español 

Este proyecto evalúa y compara diferentes enfoques para el reconocimiento de entidades nombradas (NER) en documentos médicos en español, utilizando cuatro datasets especializados y tres tipos de modelos.

## Estructura del Proyecto

#### Datasets (`/Datasets/`)
Contiene cuatro conjuntos de datos médicos anotados en formato BRAT:

- **MEDDOCAN**: 1.000 documentos médicos (500 train, 250 dev, 250 test)
- **SPG**: 1.000 documentos obtenidos a partir del módulo Synthetic Patient Generator (500 train, 250 dev, 250 test)  
- **SPGExt**: 448 documentos extendidos del SPG (358 train, 45 dev, 45 test)
- **CARMEN-I**: 1.697 documentos de casos clínicos (847 train, 425 dev, 425 test)

Cada dataset incluye archivos `.txt` (texto) y `.ann` (anotaciones) en formato BRAT.

#### Nota sobre CARMEN-I
Por motivos de licencia, todo lo referido a CARMEN-I no está público en este repositorio. El dataset puede accederse a través de PhysioNet: https://physionet.org/content/carmen-i/1.0.1/

#### Modelos (`/Modelos/`)
Implementación y evaluación de tres enfoques diferentes:

##### 1. **REGEX** - Reglas simbólicas basadas en conocimiento de dominio mediante expresiones regulares
- Enfoque basado en patrones y reglas
- Implementación en Python con procesamiento de texto
- Archivos de ejecución en cada subcarpeta (`ejecucion.md`)

##### 2. **BiLSTM-CRF** - Red Neuronal Recurrente
- Modelo de deep learning con arquitectura BiLSTM + CRF
- Basado en embeddings de palabras y contexto

##### 3. **Llama3.1** - Gran Modelo de Lenguaje
- Utiliza el modelo Llama 3.1 para NER
- Implementación con prompts (one-shot)
- Evaluación directa sobre los datasets

## Resultados
Los resultados se encuentran en la carpeta `/Resultados/`:

#### Métricas Cuantitativas
- **`resultados_dev.xlsx`**: Métricas de evaluación en conjunto de desarrollo
- **`resultados_test.xlsx`**: Métricas de evaluación en conjunto de prueba
- Contienen métricas detalladas (Precision, Recall, F1-Score) por modelo y dataset

#### Visualizaciones
- **`plots_dev/`**: Matrices de confusión y gráficos para conjunto de desarrollo
- **`plots_test/`**: Matrices de confusión y gráficos para conjunto de prueba
- Comparación visual entre modelos (REGEX, BiLSTM-CRF, Llama3.1)

## Ejecución

Para ejecutar evaluaciones específicas, consultar los archivos `ejecucion.md` en cada modelo:
- `/Modelos/REGEX/[dataset]/ejecucion.md`
- `/Modelos/BiLSTM-CRF/ejecucion.md` 
- `/Modelos/Llama3.1/[dataset]/ejecucion.md`

Para generar matrices de confusión globales:
```bash
cd Modelos/Scripts
python3 confusion_matrix_generator_global.py [MODELO] --partition [dev/test/both]
```
