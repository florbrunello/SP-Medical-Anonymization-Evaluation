"""
Script para generar matriz de confusión GLOBAL NER de cada modelo
sobre TODOS los datasets (MEDDOCAN, SPG, SPGExt, CARMEN-I) combinados
con las 28 etiquetas en orden fijo: B-*, I-*, O
Usa la misma metodología que confusion_matrix_generator.py pero para datasets globales
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import re

def tokenize_text(text):
    """Tokeniza el texto de la misma manera que el modelo"""
    import re
    tokens = []
    for match in re.finditer(r'\S+', text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens

def spans_to_bio_labels(tokens, spans):
    """Convierte spans a etiquetas BIO de la misma manera que el modelo"""
    labels = ["O"] * len(tokens)
    for label, start, end in spans:
        for i, (token, token_start, token_end) in enumerate(tokens):
            if token_end <= start:
                continue
            if token_start >= end:
                break
            if token_start >= start and token_end <= end:
                if labels[i] == "O":
                    labels[i] = "B-" + label
                elif labels[i].startswith("B-") or labels[i].startswith("I-"):
                    labels[i] = "I-" + label
    # Corregir etiquetas I- consecutivas
    for i in range(1, len(labels)):
        if labels[i].startswith("B-") and labels[i-1][2:] == labels[i][2:]:
            labels[i] = "I-" + labels[i][2:]
    return labels

def get_token_level_predictions(gold_ann, sys_ann):
    """Obtiene predicciones a nivel token usando la misma metodología que el modelo"""
    y_true, y_pred = [], []
    for doc_id in gold_ann:
        if doc_id not in sys_ann:
            continue
        gold_doc = gold_ann[doc_id]
        sys_doc = sys_ann[doc_id]
        text = gold_doc.text
        tokens = tokenize_text(text)
        gold_spans = gold_doc.get_phi()
        sys_spans = sys_doc.get_phi()
        gold_labels = spans_to_bio_labels(tokens, gold_spans)
        sys_labels = spans_to_bio_labels(tokens, sys_spans)
        if len(gold_labels) != len(sys_labels):
            print(f"Warning: Token count mismatch in {doc_id}")
            continue
        y_true.extend(gold_labels)
        y_pred.extend(sys_labels)
    return y_true, y_pred

def load_annotations_from_directory(ann_dir):
    """Carga anotaciones desde un directorio usando la misma metodología"""
    import sys
    # Agregar ruta para importar clases desde BiLSTM-CRF
    sys.path.append('/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/a) MEDDOCAN/code/')
    from classes import BratAnnotation
    
    annotations = {}
    for filename in os.listdir(ann_dir):
        if filename.endswith(".ann"):
            ann_path = os.path.join(ann_dir, filename)
            try:
                annotation = BratAnnotation(ann_path)
                annotations[annotation.id] = annotation
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping file {filename} due to parsing error: {e}")
                continue
    return annotations

# ====================
# LISTA FIJA DE 28 ETIQUETAS EN ORDEN SOLICITADO (SOLO UNA VEZ)
# ====================
ALL_ENTITIES = [
    "NOMBRE_SUJETO_ASISTENCIA",
    "EDAD_SUJETO_ASISTENCIA", 
    "SEXO_SUJETO_ASISTENCIA",
    "FAMILIARES_SUJETO_ASISTENCIA",
    "NOMBRE_PERSONAL_SANITARIO",
    "FECHAS",
    "PROFESION",
    "HOSPITAL",
    "CENTRO_SALUD",
    "INSTITUCION",
    "CALLE",
    "TERRITORIO",
    "PAIS",
    "NUMERO_TELEFONO",
    "NUMERO_FAX",
    "CORREO_ELECTRONICO",
    "ID_SUJETO_ASISTENCIA",
    "ID_CONTACTO_ASISTENCIAL",
    "ID_ASEGURAMIENTO",
    "ID_TITULACION_PERSONAL_SANITARIO",
    "ID_EMPLEO_PERSONAL_SANITARIO",
    "IDENTIF_VEHICULOS_NRSERIE_PLACAS",
    "IDENTIF_DISPOSITIVOS_NRSERIE",
    "DIREC_PROT_INTERNET",
    "URL_WEB",
    "IDENTIF_BIOMETRICOS",
    "OTRO_NUMERO_IDENTIF",
    "OTROS_SUJETO_ASISTENCIA"
]

def generate_global_confusion_matrix(model_name, gold_dirs, system_dirs, output_prefix="confusion_matrix_global", plots_dir=None, partition="dev"):
    """
    Genera matriz de confusión GLOBAL del modelo sobre TODOS los datasets combinados.
    La matriz incluye las 28 etiquetas en orden fijo.
    """

    # ====================
    # LISTA FIJA DE 28 ETIQUETAS EN ORDEN SOLICITADO
    # ====================
    base_entities = ALL_ENTITIES
    
    # Construir etiquetas BIO en orden: primero todos los B-, luego todos los I-, luego O
    labels = [f"B-{ent}" for ent in base_entities] + [f"I-{ent}" for ent in base_entities] + ["O"]
    
    print(f"Total de etiquetas: {len(labels)}")
    print(f"Orden de etiquetas: {labels}")
    
    # ====================
    # CARGAR ANOTACIONES DE TODOS LOS DATASETS Y GENERAR VECTORES L_TRUE Y L_PRED
    # ====================
    print(f"Procesando {len(gold_dirs)} datasets para modelo {model_name}")
    
    all_y_true, all_y_pred = [], []
    
    for i, (gold_dir, system_dir) in enumerate(zip(gold_dirs, system_dirs)):
        dataset_name = ["MEDDOCAN", "SPG", "SPGExt", "CARMEN-I"][i]
        print(f"\nProcesando dataset {dataset_name}:")
        print(f"  Gold: {gold_dir}")
        print(f"  System: {system_dir}")
        
        # Cargar anotaciones gold standard
        gold_ann = load_annotations_from_directory(gold_dir)
        
        # Cargar anotaciones del sistema
        sys_ann = load_annotations_from_directory(system_dir)
        
        # Generar vectores l_true y l_pred para este dataset
        y_true, y_pred = get_token_level_predictions(gold_ann, sys_ann)
        
        print(f"  Etiquetas verdaderas: {len(y_true)}")
        print(f"  Etiquetas predichas: {len(y_pred)}")
        
        # Agregar al conjunto global
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    
    print(f"\nTotal global:")
    print(f"  Etiquetas verdaderas: {len(all_y_true)}")
    print(f"  Etiquetas predichas: {len(all_y_pred)}")
    
    if len(all_y_true) != len(all_y_pred):
        raise ValueError(f"Error: número de etiquetas verdaderas ({len(all_y_true)}) != número de predichas ({len(all_y_pred)})")
    
    # ====================
    # GUARDAR VECTORES L_TRUE Y L_PRED POR PARTICION
    # ====================
    print(f"Guardando vectores l_true_global_{partition}.txt y l_pred_global_{partition}.txt...")
    with open(f'l_true_global_{partition}.txt', 'w') as f:
        for label in all_y_true:
            f.write(f"{label}\n")
    
    with open(f'l_pred_global_{partition}.txt', 'w') as f:
        for label in all_y_pred:
            f.write(f"{label}\n")
    
    print(f"Guardados {len(all_y_true)} etiquetas verdaderas y {len(all_y_pred)} predichas en l_true_global_{partition}.txt y l_pred_global_{partition}.txt")
    
    # ====================
    # GENERAR MATRIZ DE CONFUSIÓN
    # ====================
    print("Generando matriz de confusión global...")
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    
    # ====================
    # GUARDAR MATRIZ CON FORMATO ESPECÍFICO
    # ====================
    if plots_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "plots_1"))
    else:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), plots_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalizar la matriz por fila
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
    
    # Crear figura con formato específico
    fig, ax = plt.subplots(figsize=(25, 18))
    
    # Gráfico de la matriz con aspecto cuadrado
    im = ax.imshow(cm_normalized, cmap="viridis", aspect="equal", vmin=0, vmax=1)
    
    # Configurar etiquetas a 45 grados
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    
    # Etiquetas del eje X: inclinadas a 45 grados y alineadas a la derecha
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    # Etiquetas del eje Y: sin rotación, alineadas a la derecha
    ax.set_yticklabels(labels, ha='right', fontsize=10)
    
    # Título específico para matriz global con partición dinámica
    partition_name = "desarrollo" if partition == "dev" else "evaluación"
    if model_name == "REGEX":
        titulo = f"Matriz de confusión de las REGEX\nsobre la partición de {partition_name} que conforman todos los datasets"
    elif model_name == "BiLSTM-CRF":
        titulo = f"Matriz de confusión del modelo BiLSTM-CRF\nsobre la partición de {partition_name} que conforman todos los datasets"
    elif model_name == "Llama3.1":
        titulo = f"Matriz de confusión del modelo Llama 3.1\nsobre la partición de {partition_name} que conforman todos los datasets"
    else:
        titulo = f"Matriz de confusión del modelo {model_name}\nsobre la partición de {partition_name} que conforman todos los datasets"
    
    plt.title(titulo, fontsize=18, pad=30, linespacing=1.5)
    
    # Barra de color horizontal bien ubicada sin solapamientos
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                        fraction=0.035, pad=0.15, shrink=0.70)
    cbar.set_label('Proporción', fontsize=12)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()
    
    # Guardar matriz
    output_path = os.path.join(output_dir, f"{output_prefix}_{model_name}.png")
    print(f"Guardando matriz en: {output_path}")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    # ====================
    # GENERAR VERSIÓN LIMPIA SIN ETIQUETAS, COLORBAR NI TÍTULO
    # ====================
    print("Generando versión limpia sin etiquetas...")
    
    # Crear figura limpia
    fig_clean, ax_clean = plt.subplots(figsize=(25, 18))
    
    # Gráfico de la matriz con aspecto cuadrado
    im_clean = ax_clean.imshow(cm_normalized, cmap="viridis", aspect="equal", vmin=0, vmax=1)
    
    # Sin etiquetas, sin colorbar, sin título
    ax_clean.set_xticks([])
    ax_clean.set_yticks([])
    
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()
    
    # Guardar matriz limpia
    output_path_clean = os.path.join(output_dir, f"{output_prefix}_{model_name}_no_labels.png")
    print(f"Guardando matriz limpia en: {output_path_clean}")
    plt.savefig(output_path_clean, bbox_inches="tight", dpi=300)
    plt.close(fig_clean)

    # ====================
    # GENERAR PRIMER CUADRANTE (B-* y O)
    # ====================
    q1_indices = [i for i, lab in enumerate(labels) if lab.startswith("B-") or lab == "O"]
    if q1_indices:
        cm_q1 = cm_normalized[q1_indices][:, q1_indices]
        labels_q1 = [labels[i] for i in q1_indices]

        fig_q1, ax_q1 = plt.subplots(figsize=(25, 18))
        im_q1 = ax_q1.imshow(cm_q1, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax_q1.set_xticks(range(len(labels_q1)))
        ax_q1.set_yticks(range(len(labels_q1)))
        ax_q1.set_xticklabels(labels_q1, rotation=45, ha='right', fontsize=10)
        ax_q1.set_yticklabels(labels_q1, ha='right', fontsize=10)

        partition_name = "desarrollo" if partition == "dev" else "evaluación"
        if model_name == "REGEX":
            titulo_q1 = f"Matriz de confusión de las REGEX\nsobre la partición de {partition_name} que conforman todos los datasets"
        elif model_name == "BiLSTM-CRF":
            titulo_q1 = f"Matriz de confusión del modelo BiLSTM-CRF\nsobre la partición de {partition_name} que conforman todos los datasets"
        elif model_name == "Llama3.1":
            titulo_q1 = f"Matriz de confusión del modelo Llama 3.1\nsobre la partición de {partition_name} que conforman todos los datasets"
        else:
            titulo_q1 = f"Matriz de confusión del modelo {model_name}\nsobre la partición de {partition_name} que conforman todos los datasets"
        plt.title(titulo_q1, fontsize=18, pad=30, linespacing=1.5)
        cbar_q1 = plt.colorbar(im_q1, ax=ax_q1, orientation='horizontal', 
                            fraction=0.035, pad=0.15, shrink=0.70)
        cbar_q1.set_label('Proporción', fontsize=12)
        cbar_q1.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar_q1.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.tight_layout()
        output_path_q1 = os.path.join(output_dir, f"{output_prefix}_{model_name}_Q1.png")
        print(f"Guardando primer cuadrante en: {output_path_q1}")
        plt.savefig(output_path_q1, bbox_inches="tight", dpi=300)
        plt.close(fig_q1)

        # Versión limpia del Q1
        fig_q1_clean, ax_q1_clean = plt.subplots(figsize=(25, 18))
        ax_q1_clean.imshow(cm_q1, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax_q1_clean.set_xticks([])
        ax_q1_clean.set_yticks([])
        plt.tight_layout()
        output_path_q1_clean = os.path.join(output_dir, f"{output_prefix}_{model_name}_Q1_no_labels.png")
        plt.savefig(output_path_q1_clean, bbox_inches="tight", dpi=300)
        plt.close(fig_q1_clean)
    
    # ====================
    # REPORTE DETALLADO
    # ====================
    print("\n=== REPORTE DETALLADO GLOBAL ===")
    print(classification_report(all_y_true, all_y_pred, labels=labels, digits=4))
    
    # Métricas por entidad
    print("\n=== MÉTRICAS POR TIPO DE ENTIDAD ===")
    entity_metrics = {}
    
    for true_label, pred_label in zip(all_y_true, all_y_pred):
        if true_label.startswith('B-') or true_label.startswith('I-'):
            true_entity = true_label[2:]
        else:
            true_entity = true_label
        if pred_label.startswith('B-') or pred_label.startswith('I-'):
            pred_entity = pred_label[2:]
        else:
            pred_entity = pred_label
            
        if true_entity not in entity_metrics:
            entity_metrics[true_entity] = {'tp': 0, 'fp': 0, 'fn': 0}
        if pred_entity not in entity_metrics:
            entity_metrics[pred_entity] = {'tp': 0, 'fp': 0, 'fn': 0}
            
        if true_entity == pred_entity and true_entity != 'O':
            entity_metrics[true_entity]['tp'] += 1
        elif true_entity != 'O' and pred_entity == 'O':
            entity_metrics[true_entity]['fn'] += 1
        elif true_entity == 'O' and pred_entity != 'O':
            entity_metrics[pred_entity]['fp'] += 1
        elif true_entity != 'O' and pred_entity != 'O' and true_entity != pred_entity:
            entity_metrics[true_entity]['fn'] += 1
            entity_metrics[pred_entity]['fp'] += 1
    
    print(f"{'Entidad':<30} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
    print("-" * 80)
    
    for entity in base_entities:
        if entity in entity_metrics:
            metrics = entity_metrics[entity]
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            # Agregar las métricas calculadas al diccionario
            entity_metrics[entity]['precision'] = precision
            entity_metrics[entity]['recall'] = recall
            entity_metrics[entity]['f1'] = f1
            print(f"{entity:<30} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {tp:<5} {fp:<5} {fn:<5}")
    
    print("-" * 80)
    
    # Resumen general
    total_entities = sum(m['tp'] + m['fn'] for m in entity_metrics.values())
    total_predicted = sum(m['tp'] + m['fp'] for m in entity_metrics.values())
    total_correct = sum(m['tp'] for m in entity_metrics.values())
    
    if entity_metrics:
        # Solo considerar entidades que tienen métricas calculadas
        entities_with_metrics = [entity for entity in base_entities if entity in entity_metrics and 'precision' in entity_metrics[entity]]
        if entities_with_metrics:
            macro_precision = sum(entity_metrics[entity]['precision'] for entity in entities_with_metrics) / len(entities_with_metrics)
            macro_recall = sum(entity_metrics[entity]['recall'] for entity in entities_with_metrics) / len(entities_with_metrics)
            macro_f1 = sum(entity_metrics[entity]['f1'] for entity in entities_with_metrics) / len(entities_with_metrics)
            
            print(f"\nRESUMEN GENERAL GLOBAL:")
            print(f"Total entidades: {total_entities}")
            print(f"Total predichas: {total_predicted}")
            print(f"Total correctas: {total_correct}")
            print(f"Macro Precision: {macro_precision:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
        else:
            print(f"\nRESUMEN GENERAL GLOBAL:")
            print(f"Total entidades: {total_entities}")
            print(f"Total predichas: {total_predicted}")
            print(f"Total correctas: {total_correct}")
            print("No hay entidades con métricas calculadas para macro-averaging")
    
    print(f"\nMatriz global generada exitosamente con {len(labels)} etiquetas en orden fijo.")
    print(f"Archivos guardados en: {output_dir}")
    print(f"- Matriz completa: {output_prefix}_{model_name}.png")
    print(f"- Matriz limpia (sin etiquetas): {output_prefix}_{model_name}_no_labels.png")
    print(f"Vectores l_true_global_{partition}.txt y l_pred_global_{partition}.txt guardados en el directorio actual.")

def generate_global_confusion_matrix_cleaned(model_name, gold_dirs, system_dirs, output_prefix="confusion_matrix_global_cleaned", plots_dir=None, partition="dev"):
    """
    Genera matriz de confusión GLOBAL del modelo sobre TODOS los datasets combinados,
    pero solo con las entidades de base_entities_2 (sin las que no aparecen nunca).
    El formato, colorbar, etiquetas, layout y título son idénticos a la función NO cleaned.
    """
    # Lista reducida de entidades
    cleaned_entities = [ent for ent in ALL_ENTITIES if ent not in {"DIREC_PROT_INTERNET", "URL_WEB", "IDENTIF_BIOMETRICOS"}]
    labels = [f"B-{ent}" for ent in cleaned_entities] + [f"I-{ent}" for ent in cleaned_entities] + ["O"]
    print(f"Total de etiquetas (cleaned): {len(labels)}")
    print(f"Orden de etiquetas (cleaned): {labels}")

    all_y_true, all_y_pred = [], []
    for i, (gold_dir, system_dir) in enumerate(zip(gold_dirs, system_dirs)):
        gold_ann = load_annotations_from_directory(gold_dir)
        sys_ann = load_annotations_from_directory(system_dir)
        y_true, y_pred = get_token_level_predictions(gold_ann, sys_ann)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    if len(all_y_true) != len(all_y_pred):
        raise ValueError(f"Error: número de etiquetas verdaderas ({len(all_y_true)}) != número de predichas ({len(all_y_pred)})")

    # NO filtrar los vectores, solo cambiar los labels
    with open(f'l_true_global_{partition}_cleaned.txt', 'w') as f:
        for label in all_y_true:
            f.write(f"{label}\n")
    with open(f'l_pred_global_{partition}_cleaned.txt', 'w') as f:
        for label in all_y_pred:
            f.write(f"{label}\n")

    # Matriz de confusión
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

    # Eliminar filas y columnas de etiquetas cuya suma de fila es 0 (toda la fila violeta)
    rows_to_keep = np.where(cm.sum(axis=1) > 0)[0]
    cm = cm[rows_to_keep][:, rows_to_keep]
    cm_normalized = cm_normalized[rows_to_keep][:, rows_to_keep]
    labels = [labels[i] for i in rows_to_keep]

    # Carpeta de salida
    if plots_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "plots_2"))
    else:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), plots_dir))
    os.makedirs(output_dir, exist_ok=True)

    # === FORMATO Y ESTILO IDÉNTICO AL NO CLEANED ===
    fig, ax = plt.subplots(figsize=(25, 18))
    im = ax.imshow(cm_normalized, cmap="viridis", aspect="equal", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, ha='right', fontsize=10)

    # Título específico para matriz global con partición dinámica
    partition_name = "desarrollo" if partition == "dev" else "evaluación"
    if model_name == "REGEX":
        titulo = f"Matriz de confusión de las REGEX\nsobre la partición de {partition_name} que conforman todos los datasets"
    elif model_name == "BiLSTM-CRF":
        titulo = f"Matriz de confusión del modelo BiLSTM-CRF\nsobre la partición de {partition_name} que conforman todos los datasets"
    elif model_name == "Llama3.1":
        titulo = f"Matriz de confusión del modelo Llama 3.1\nsobre la partición de {partition_name} que conforman todos los datasets"
    else:
        titulo = f"Matriz de confusión del modelo {model_name}\nsobre la partición de {partition_name} que conforman todos los datasets"
    plt.title(titulo, fontsize=18, pad=30, linespacing=1.5)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                        fraction=0.035, pad=0.15, shrink=0.70)
    cbar.set_label('Proporción', fontsize=12)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{output_prefix}_{model_name}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Versión limpia
    fig_clean, ax_clean = plt.subplots(figsize=(25, 18))
    im_clean = ax_clean.imshow(cm_normalized, cmap="viridis", aspect="equal", vmin=0, vmax=1)
    ax_clean.set_xticks([])
    ax_clean.set_yticks([])
    plt.tight_layout()
    output_path_clean = os.path.join(output_dir, f"{output_prefix}_{model_name}_no_labels.png")
    plt.savefig(output_path_clean, bbox_inches="tight", dpi=300)
    plt.close(fig_clean)
    print(f"Guardadas matrices cleaned en: {output_path} y {output_path_clean}")

    # ====================
    # GENERAR PRIMER CUADRANTE (B-* y O) PARA CLEANED
    # ====================
    q1_indices = [i for i, lab in enumerate(labels) if lab.startswith("B-") or lab == "O"]
    if q1_indices:
        cm_q1 = cm_normalized[q1_indices][:, q1_indices]
        labels_q1 = [labels[i] for i in q1_indices]

        fig_q1, ax_q1 = plt.subplots(figsize=(25, 18))
        im_q1 = ax_q1.imshow(cm_q1, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax_q1.set_xticks(range(len(labels_q1)))
        ax_q1.set_yticks(range(len(labels_q1)))
        ax_q1.set_xticklabels(labels_q1, rotation=45, ha='right', fontsize=10)
        ax_q1.set_yticklabels(labels_q1, ha='right', fontsize=10)

        partition_name = "desarrollo" if partition == "dev" else "evaluación"
        if model_name == "REGEX":
            titulo_q1 = f"Matriz de confusión de las REGEX\nsobre la partición de {partition_name} que conforman todos los datasets"
        elif model_name == "BiLSTM-CRF":
            titulo_q1 = f"Matriz de confusión del modelo BiLSTM-CRF\nsobre la partición de {partition_name} que conforman todos los datasets"
        elif model_name == "Llama3.1":
            titulo_q1 = f"Matriz de confusión del modelo Llama 3.1\nsobre la partición de {partition_name} que conforman todos los datasets"
        else:
            titulo_q1 = f"Matriz de confusión del modelo {model_name}\nsobre la partición de {partition_name} que conforman todos los datasets"
        plt.title(titulo_q1, fontsize=18, pad=30, linespacing=1.5)
        cbar_q1 = plt.colorbar(im_q1, ax=ax_q1, orientation='horizontal', 
                            fraction=0.035, pad=0.15, shrink=0.70)
        cbar_q1.set_label('Proporción', fontsize=12)
        cbar_q1.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar_q1.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.tight_layout()
        output_path_q1 = os.path.join(output_dir, f"{output_prefix}_{model_name}_Q1.png")
        print(f"Guardando primer cuadrante (cleaned) en: {output_path_q1}")
        plt.savefig(output_path_q1, bbox_inches="tight", dpi=300)
        plt.close(fig_q1)

        # Versión limpia del Q1 cleaned
        fig_q1_clean, ax_q1_clean = plt.subplots(figsize=(25, 18))
        ax_q1_clean.imshow(cm_q1, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax_q1_clean.set_xticks([])
        ax_q1_clean.set_yticks([])
        plt.tight_layout()
        output_path_q1_clean = os.path.join(output_dir, f"{output_prefix}_{model_name}_Q1_no_labels.png")
        plt.savefig(output_path_q1_clean, bbox_inches="tight", dpi=300)
        plt.close(fig_q1_clean)

def generate_standalone_colorbar(output_dir, model_name, output_prefix="confusion_matrix_global"):
    """
    Genera colorbar vertical independiente sin etiqueta
    """
    print("Generando colorbar independiente...")
    
    # Crear figura para la colorbar
    fig_cbar, ax_cbar = plt.subplots(figsize=(2, 8))
    
    # Crear colorbar vertical con formato específico
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    
    # Configurar colorbar vertical sin etiqueta
    cbar = plt.colorbar(sm, ax=ax_cbar, orientation='vertical', 
                        fraction=1.0, pad=0.05)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Ocultar el eje principal
    ax_cbar.set_visible(False)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar colorbar independiente
    output_path_cbar = os.path.join(output_dir, f"{output_prefix}_colorbar.png")
    print(f"Guardando colorbar en: {output_path_cbar}")
    plt.savefig(output_path_cbar, bbox_inches="tight", dpi=300)
    plt.close(fig_cbar)
    
    print(f"Colorbar independiente generada: {output_path_cbar}")
    
    print(f"\nColorbar independiente generada exitosamente.")
    print(f"Archivo guardado en: {output_dir}")
    print(f"- Colorbar: {output_prefix}_colorbar.png")

def main():
    # Configuración de rutas para cada modelo (desde Modelos/Scripts/)
    MODEL_CONFIGS = {
        "REGEX": {
            "dev": {
                "gold_dirs": [
                    "../../Datasets/MEDDOCAN/dev/brat",
                    "../../Datasets/SPG/dev/brat", 
                    "../../Datasets/SPGExt/dev/brat",
                    "../../Datasets/CARMEN-I/dev/brat"
                ],
                "system_dirs": [
                    "../REGEX/a) MEDDOCAN/brat/dev",
                    "../REGEX/b) SPG/brat/dev",
                    "../REGEX/c) SPGExt/brat/dev",
                    "../REGEX/d) CARMEN-I/brat/dev"
                ]
            },
            "test": {
                "gold_dirs": [
                    "../../Datasets/MEDDOCAN/test/brat",
                    "../../Datasets/SPG/test/brat", 
                    "../../Datasets/SPGExt/test/brat",
                    "../../Datasets/CARMEN-I/test/brat"
                ],
                "system_dirs": [
                    "../REGEX/a) MEDDOCAN/brat/test",
                    "../REGEX/b) SPG/brat/test",
                    "../REGEX/c) SPGExt/brat/test",
                    "../REGEX/d) CARMEN-I/brat/test"
                ]
            }
        },
        "BiLSTM-CRF": {
            "dev": {
                "gold_dirs": [
                    "../../Datasets/MEDDOCAN/dev/brat",
                    "../../Datasets/SPG/dev/brat",
                    "../../Datasets/SPGExt/dev/brat",
                    "../../Datasets/CARMEN-I/dev/brat"
                ],
                "system_dirs": [
                    "../BiLSTM-CRF/a) MEDDOCAN/dev/system",
                    "../BiLSTM-CRF/b) SPG/dev/system", 
                    "../BiLSTM-CRF/c) SPGExt/dev/system",
                    "../BiLSTM-CRF/d) CARMEN-I/dev/system"
                ]
            },
            "test": {
                "gold_dirs": [
                    "../../Datasets/MEDDOCAN/test/brat",
                    "../../Datasets/SPG/test/brat",
                    "../../Datasets/SPGExt/test/brat",
                    "../../Datasets/CARMEN-I/test/brat"
                ],
                "system_dirs": [
                    "../BiLSTM-CRF/a) MEDDOCAN/output/test/system",
                    "../BiLSTM-CRF/b) SPG/output/test/system", 
                    "../BiLSTM-CRF/c) SPGExt/output/test/system",
                    "../BiLSTM-CRF/d) CARMEN-I/output/test/system"
                ]
            }
        },
        "Llama3.1": {
            "dev": {
                "gold_dirs": [
                    "../../Datasets/MEDDOCAN/dev/brat",
                    "../../Datasets/SPG/dev/brat",
                    "../../Datasets/SPGExt/dev/brat",
                    "../../Datasets/CARMEN-I/dev/brat"
                ],
                "system_dirs": [
                    "../Llama3.1/a) MEDDOCAN/dev_prompt_OneShot/ann",
                    "../Llama3.1/b) SPG/dev/ann",
                    "../Llama3.1/c) SPGExt/dev/ann",
                    "../Llama3.1/d) CARMEN-I/dev/ann"
                ]
            },
            "test": {
                "gold_dirs": [
                    "../../Datasets/MEDDOCAN/test/brat",
                    "../../Datasets/SPG/test/brat",
                    "../../Datasets/SPGExt/test/brat",
                    "../../Datasets/CARMEN-I/test/brat"
                ],
                "system_dirs": [
                    "../Llama3.1/a) MEDDOCAN/test/ann",
                    "../Llama3.1/b) SPG/test/ann",
                    "../Llama3.1/c) SPGExt/test/ann",
                    "../Llama3.1/d) CARMEN-I/test/ann"
                ]
            }
        }
    }
    
    parser = argparse.ArgumentParser(description="Genera matriz de confusión GLOBAL NER de cada modelo sobre TODOS los datasets combinados")
    parser.add_argument("model", nargs='?', choices=["REGEX", "BiLSTM-CRF", "Llama3.1"], help="Modelo a evaluar")
    parser.add_argument("--output-prefix", default="confusion_matrix_global", help="Prefijo para archivos de salida")
    parser.add_argument("--colorbar-only", action="store_true", help="Generar solo la colorbar independiente")
    parser.add_argument("--cleaned", action="store_true", help="Generar matriz solo con entidades base_entities_2 (limpia)")
    parser.add_argument("--plots-dir", default=None, help="Directorio donde guardar las matrices de confusión y salidas")
    parser.add_argument("--partition", choices=["dev", "test", "both"], default="dev", help="Partición a procesar (dev, test, o both)")
    
    args = parser.parse_args()
    
    # Si solo se quiere generar la colorbar
    if args.colorbar_only:
        # Determinar qué particiones procesar para colorbar
        partitions_to_process = []
        if args.partition == "both":
            partitions_to_process = ["dev", "test"]
        else:
            partitions_to_process = [args.partition]
        
        for partition in partitions_to_process:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.plots_dir or f"plots_1_{partition}"))
            os.makedirs(output_dir, exist_ok=True)
            generate_standalone_colorbar(output_dir, "GLOBAL", args.output_prefix)
        return
    
    # Verificar que se especifique un modelo si no es solo colorbar
    if not args.model:
        print("Error: Debe especificar un modelo (REGEX, BiLSTM-CRF, o Llama3.1)")
        sys.exit(1)
    
    # Determinar qué particiones procesar
    partitions_to_process = []
    if args.partition == "both":
        partitions_to_process = ["dev", "test"]
    else:
        partitions_to_process = [args.partition]
    
    # Procesar cada partición
    for partition in partitions_to_process:
        print(f"\n=== Procesando partición: {partition.upper()} ===")
        
        # Verificar que existan todos los directorios para esta partición
        config = MODEL_CONFIGS[args.model][partition]
        for gold_dir, system_dir in zip(config["gold_dirs"], config["system_dirs"]):
            if not os.path.exists(gold_dir):
                print(f"Error: Directorio gold {gold_dir} no encontrado")
                sys.exit(1)
            if not os.path.exists(system_dir):
                print(f"Error: Directorio system {system_dir} no encontrado")
                sys.exit(1)
        
        # Generar matriz para esta partición
        output_prefix = f"{args.output_prefix}_{partition}"
        
        # Determinar directorio de salida basado en cleaned y partición
        if args.plots_dir:
            plots_dir = args.plots_dir
        else:
            plots_dir = f"plots_2_{partition}" if args.cleaned else f"plots_1_{partition}"
        
        if args.cleaned:
            generate_global_confusion_matrix_cleaned(args.model, config["gold_dirs"], config["system_dirs"], output_prefix + "_cleaned", plots_dir=plots_dir, partition=partition)
        else:
            generate_global_confusion_matrix(args.model, config["gold_dirs"], config["system_dirs"], output_prefix, plots_dir=plots_dir, partition=partition)

if __name__ == "__main__":
    main() 