# Generación de matrices de confusión

Este documento contiene las instrucciones para generar matrices de confusión token a token para comparar modelos de NER (REGEX, BiLSTM-CRF, Llama3.1) sobre los datasets MEDDOCAN, SPG, SPGExt y CARMEN-I.

Cada modelo y dataset tiene su propio script específico ubicado en la carpeta correspondiente.

## Estructura de scripts

### BiLSTM-CRF
- `Modelos/BiLSTM-CRF/a) MEDDOCAN/code/Extension2/confusion_matrix_generator.py`
- `Modelos/BiLSTM-CRF/b) SPG/code/Extension2/confusion_matrix_generator.py`
- `Modelos/BiLSTM-CRF/c) SPGExt/code/Extension2/confusion_matrix_generator.py`
- `Modelos/BiLSTM-CRF/d) CARMEN-I/code/Extension2/confusion_matrix_generator.py`

### REGEX
- `Modelos/REGEX/a) MEDDOCAN/code/confusion_matrix_generator.py`
- `Modelos/REGEX/b) SPG/code/confusion_matrix_generator.py`
- `Modelos/REGEX/c) SPGExt/code/confusion_matrix_generator.py`
- `Modelos/REGEX/d) CARMEN-I/code/confusion_matrix_generator.py`

### Llama3.1
- `Modelos/Llama3.1/a) MEDDOCAN/confusion_matrix_generator.py`
- `Modelos/Llama3.1/b) SPG/confusion_matrix_generator.py`
- `Modelos/Llama3.1/c) SPGExt/confusion_matrix_generator.py`
- `Modelos/Llama3.1/d) CARMEN-I/confusion_matrix_generator.py`

### Script Global (Todos los datasets combinados)
- `Modelos/utils/confusion_matrix_generator_global.py`

## Uso básico

Cada script acepta los siguientes argumentos:

```bash
python3 confusion_matrix_generator.py <directorio_gold> <directorio_predicciones> --output-prefix <prefijo_salida>
```

- `<directorio_gold>`: Carpeta con anotaciones gold standard en formato BRAT (.ann)
- `<directorio_predicciones>`: Carpeta con anotaciones predichas por el sistema en formato BRAT (.ann)
- `--output-prefix`: Prefijo para los archivos de salida (imágenes de las matrices)

## Comandos específicos por modelo y dataset

### REGEX

#### MEDDOCAN
```bash
cd "Modelos/REGEX/a) MEDDOCAN/code"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/MEDDOCAN/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/a) MEDDOCAN/brat/dev" --output-prefix confusion_matrix_REGEX_MEDDOCAN --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/MEDDOCAN/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/a) MEDDOCAN/brat/test" --output-prefix confusion_matrix_REGEX_MEDDOCAN --partition test
```

#### SPG
```bash
cd "Modelos/REGEX/b) SPG/code"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPG/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/b) SPG/brat/dev" --output-prefix confusion_matrix_REGEX_SPG --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPG/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/b) SPG/brat/test" --output-prefix confusion_matrix_REGEX_SPG --partition test
```

#### SPGExt
```bash
cd "Modelos/REGEX/c) SPGExt/code"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPGExt/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/c) SPGExt/brat/dev" --output-prefix confusion_matrix_REGEX_SPGExt --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPGExt/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/c) SPGExt/brat/test" --output-prefix confusion_matrix_REGEX_SPGExt --partition test
```

#### CARMEN-I
```bash
cd "Modelos/REGEX/d) CARMEN-I/code"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/CARMEN-I/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/d) CARMEN-I/brat/dev" --output-prefix confusion_matrix_REGEX_CARMEN_I --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/CARMEN-I/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/d) CARMEN-I/brat/test" --output-prefix confusion_matrix_REGEX_CARMEN_I --partition test
```

### BiLSTM-CRF

#### MEDDOCAN
```bash
cd "Modelos/BiLSTM-CRF/a) MEDDOCAN/code/Extension2"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/MEDDOCAN/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/a) MEDDOCAN/dev/system" --output-prefix confusion_matrix_BiLSTM_CRF_MEDDOCAN --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/MEDDOCAN/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/a) MEDDOCAN/output/test/system" --output-prefix confusion_matrix_BiLSTM_CRF_MEDDOCAN --partition test
```

#### SPG
```bash
cd "Modelos/BiLSTM-CRF/b) SPG/code/Extension2"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/b) SPG/dev/gold" "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/b) SPG/dev/system" --output-prefix confusion_matrix_BiLSTM_CRF_SPG --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/b) SPG/output/test/gold" "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/b) SPG/output/test/system" --output-prefix confusion_matrix_BiLSTM_CRF_SPG --partition test
```

#### SPGExt
```bash
cd "Modelos/BiLSTM-CRF/c) SPGExt/code/Extension2"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPGExt/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/c) SPGExt/dev/system" --output-prefix confusion_matrix_BiLSTM_CRF_SPGExt --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPGExt/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/c) SPGExt/output/test/system" --output-prefix confusion_matrix_BiLSTM_CRF_SPGExt --partition test
```

#### CARMEN-I
```bash
cd "Modelos/BiLSTM-CRF/d) CARMEN-I/code/Extension2"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/CARMEN-I/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/d) CARMEN-I/dev/system" --output-prefix confusion_matrix_BiLSTM_CRF_CARMEN_I --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/CARMEN-I/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/d) CARMEN-I/output/test/system" --output-prefix confusion_matrix_BiLSTM_CRF_CARMEN_I --partition test
```

### Llama3.1

#### MEDDOCAN
```bash
cd "Modelos/Llama3.1/a) MEDDOCAN/code"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/MEDDOCAN/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/Llama3.1/a) MEDDOCAN/dev_prompt_OneShot/ann" --output-prefix confusion_matrix_Llama3.1_MEDDOCAN --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/MEDDOCAN/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/Llama3.1/a) MEDDOCAN/test/ann" --output-prefix confusion_matrix_Llama3.1_MEDDOCAN --partition test
```

#### SPG
```bash
cd "Modelos/Llama3.1/b) SPG/code"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPG/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/Llama3.1/b) SPG/dev/ann" --output-prefix confusion_matrix_Llama3.1_SPG --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPG/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/Llama3.1/b) SPG/test/ann" --output-prefix confusion_matrix_Llama3.1_SPG --partition test
```

#### SPGExt
```bash
cd "Modelos/Llama3.1/c) SPGExt/code"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPGExt/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/Llama3.1/c) SPGExt/dev/ann" --output-prefix confusion_matrix_Llama3.1_SPGExt --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPGExt/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/Llama3.1/c) SPGExt/test/ann" --output-prefix confusion_matrix_Llama3.1_SPGExt --partition test
```

#### CARMEN-I
```bash
cd "Modelos/Llama3.1/d) CARMEN-I/code"
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/CARMEN-I/dev/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/Llama3.1/d) CARMEN-I/dev/ann" --output-prefix confusion_matrix_Llama3.1_CARMEN_I --partition dev
python3 confusion_matrix_generator.py "/home/usuario/Documentos/TrabajoEspecial/Datasets/CARMEN-I/test/brat" "/home/usuario/Documentos/TrabajoEspecial/Modelos/Llama3.1/d) CARMEN-I/test/ann" --output-prefix confusion_matrix_Llama3.1_CARMEN_I --partition test
```   

## Matrices Globales (Todos los datasets combinados)

A partir de ahora, las matrices globales se guardan en carpetas separadas por partición y tipo:
- **plots_1_dev/**: para las matrices estándar de desarrollo (NO cleaned, con las 28 entidades)
- **plots_2_dev/**: para las matrices "cleaned" de desarrollo (solo con las entidades realmente presentes)
- **plots_1_test/**: para las matrices estándar de test (NO cleaned, con las 28 entidades)
- **plots_2_test/**: para las matrices "cleaned" de test (solo con las entidades realmente presentes)

Puedes cambiar la carpeta de salida usando el argumento opcional `--plots-dir`.

Los scripts globales combinan automáticamente MEDDOCAN, SPG, SPGExt y CARMEN-I.

### REGEX Global (NO cleaned, por defecto en plots_1_dev y plots_1_test)
```bash
cd "Modelos/Scripts"
python3 confusion_matrix_generator_global.py REGEX --output-prefix confusion_matrix_global --partition dev
python3 confusion_matrix_generator_global.py REGEX --output-prefix confusion_matrix_global --partition test
python3 confusion_matrix_generator_global.py REGEX --output-prefix confusion_matrix_global --partition both
```

### BiLSTM-CRF Global (NO cleaned, por defecto en plots_1_dev y plots_1_test)
```bash
cd "Modelos/Scripts"
python3 confusion_matrix_generator_global.py BiLSTM-CRF --output-prefix confusion_matrix_global --partition dev
python3 confusion_matrix_generator_global.py BiLSTM-CRF --output-prefix confusion_matrix_global --partition test
python3 confusion_matrix_generator_global.py BiLSTM-CRF --output-prefix confusion_matrix_global --partition both
```

### Llama3.1 Global (NO cleaned, por defecto en plots_1_dev y plots_1_test)
```bash
cd "Modelos/Scripts"
python3 confusion_matrix_generator_global.py Llama3.1 --output-prefix confusion_matrix_global --partition dev
python3 confusion_matrix_generator_global.py Llama3.1 --output-prefix confusion_matrix_global --partition test
python3 confusion_matrix_generator_global.py Llama3.1 --output-prefix confusion_matrix_global --partition both
```

### REGEX Global (cleaned, por defecto en plots_2_dev y plots_2_test)
```bash
cd "Modelos/Scripts"
python3 confusion_matrix_generator_global.py REGEX --output-prefix confusion_matrix_global --cleaned --partition dev
python3 confusion_matrix_generator_global.py REGEX --output-prefix confusion_matrix_global --cleaned --partition test
python3 confusion_matrix_generator_global.py REGEX --output-prefix confusion_matrix_global --cleaned --partition both
```

### BiLSTM-CRF Global (cleaned, por defecto en plots_2_dev y plots_2_test)
```bash
cd "Modelos/Scripts"
python3 confusion_matrix_generator_global.py BiLSTM-CRF --output-prefix confusion_matrix_global --cleaned --partition dev
python3 confusion_matrix_generator_global.py BiLSTM-CRF --output-prefix confusion_matrix_global --cleaned --partition test
python3 confusion_matrix_generator_global.py BiLSTM-CRF --output-prefix confusion_matrix_global --cleaned --partition both
```

### Llama3.1 Global (cleaned, por defecto en plots_2_dev y plots_2_test)
```bash
cd "Modelos/Scripts"
python3 confusion_matrix_generator_global.py Llama3.1 --output-prefix confusion_matrix_global --cleaned --partition dev
python3 confusion_matrix_generator_global.py Llama3.1 --output-prefix confusion_matrix_global --cleaned --partition test
python3 confusion_matrix_generator_global.py Llama3.1 --output-prefix confusion_matrix_global --cleaned --partition both
```

#### Comandos directos (TEST cleaned, tres modelos)
```bash
cd "Modelos/Scripts"
python3 confusion_matrix_generator_global.py REGEX      --output-prefix confusion_matrix_global --cleaned --partition test
python3 confusion_matrix_generator_global.py BiLSTM-CRF --output-prefix confusion_matrix_global --cleaned --partition test
python3 confusion_matrix_generator_global.py Llama3.1   --output-prefix confusion_matrix_global --cleaned --partition test
```

### Generar solo colorbar independiente

Para generar únicamente la colorbar vertical independiente (por defecto en plots_1_{partition}, o la que indiques):

```bash
cd "Modelos/Scripts"
python3 confusion_matrix_generator_global.py --colorbar-only --partition dev
python3 confusion_matrix_generator_global.py --colorbar-only --partition test
python3 confusion_matrix_generator_global.py --colorbar-only --partition both
python3 confusion_matrix_generator_global.py --colorbar-only --plots-dir plots_1_dev
python3 confusion_matrix_generator_global.py --colorbar-only --plots-dir plots_2_test
```

### Salidas de los scripts globales

Cada ejecución del script global genera los siguientes archivos en la carpeta correspondiente (`plots_1_{partition}` o `plots_2_{partition}`):

1. **Matriz completa**: `confusion_matrix_global_{MODELO}.png`
   - Incluye etiquetas, colorbar horizontal y título específico
   - Formato: 25x18 pulgadas, etiquetas rotadas 45°

2. **Matriz limpia**: `confusion_matrix_global_{MODELO}_no_labels.png`
   - Sin etiquetas, sin colorbar, sin título
   - Solo la matriz de confusión visual

3. **Vectores de etiquetas**: `l_true_global_{partition}.txt` y `l_pred_global_{partition}.txt` (o `_cleaned.txt` para cleaned)
   - Etiquetas token a token para reproducibilidad

4. **Primer cuadrante (B-* y O)**: `confusion_matrix_global_{MODELO}_Q1.png`
   - Submatriz de la esquina superior izquierda (todas las etiquetas B-* y la clase `O`)
   - Mismo formato que la matriz completa: etiquetas, colorbar horizontal, título

5. **Primer cuadrante limpio**: `confusion_matrix_global_{MODELO}_Q1_no_labels.png`
   - Versión sin etiquetas ni título del primer cuadrante

### Salida del script con --colorbar-only

El script con la opción `--colorbar-only` genera:

- **Colorbar independiente**: `confusion_matrix_global_colorbar.png`
  - Colorbar vertical con escala de 0.0 a 1.0
  - Formato: 2x8 pulgadas, colormap viridis
  - Sin etiqueta "Proporción"
  - Etiquetas en intervalos de 0.2
  - Nombre fijo sin incluir el modelo

## Características de los scripts

Todos los scripts incluyen:

1. **28 etiquetas en orden fijo**: B-*, I-*, O para todas las entidades (NO cleaned)
2. **"Cleaned" solo entidades presentes**: B-*, I-*, O solo para entidades realmente presentes
3. **Matrices normalizadas**: Normalización por fila para mejor visualización
4. **Formato específico**: 
   - Colorbar horizontal
   - Etiquetas rotadas 45°
   - Títulos específicos por modelo y dataset
   - Además se guarda automáticamente el **primer cuadrante (B-* y O)** con y sin etiquetas
5. **Vectores de salida**: Genera `l_true_{partition}.txt` y `l_pred_{partition}.txt` (ej: `l_true_dev.txt`, `l_pred_test.txt`)
6. **Reportes detallados**: Métricas por entidad y resumen general
7. **Guardado en `plots_1_{partition}` o `plots_2_{partition}`**: Carpeta específica para las matrices generadas por partición

## Salidas generadas

Cada script genera:

- **Matriz de confusión**: Archivo PNG en la carpeta correspondiente
- **Vectores de etiquetas**: `l_true_{partition}.txt` y `l_pred_{partition}.txt` en el directorio del script
- **Reporte detallado**: Métricas por entidad y macro-métricas en consola

### Scripts Globales adicionales:
- **Vectores globales**: `l_true_global_{partition}.txt` y `l_pred_global_{partition}.txt` (o `_cleaned.txt`)
- **Matriz global**: Combina todos los datasets para cada modelo y partición
- **Matriz limpia**: Versión sin etiquetas, sin colorbar y sin título
- **Colorbar independiente**: Colorbar vertical con escala de 0.0 a 1.0

## Notas importantes

- Todos los scripts usan la misma metodología de tokenización y conversión BIO
- Las matrices incluyen las 28 entidades en el orden especificado (NO cleaned) o solo las presentes (cleaned)
- Los títulos de las matrices son específicos para cada modelo y dataset
- Las rutas de importación están ajustadas para cada ubicación específica
- Los scripts globales combinan automáticamente MEDDOCAN, SPG, SPGExt y CARMEN-I