"""
Script para generar matriz de confusión NER del modelo Llama3.1
sobre la partición de desarrollo del dataset SPGExt
con las 28 etiquetas en orden fijo: B-*, I-*, O
Genera los vectores l_true.txt y l_pred.txt usando la misma metodología que el modelo Llama3.1
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import argparse

def tokenize_text(text):
    """Tokeniza el texto de la misma manera que el modelo Llama3.1"""
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

def generate_confusion_matrix_from_annotations(gold_dir, system_dir, output_prefix="confusion_matrix", partition="dev", cleaned=False):
    """
    Genera matriz de confusión del modelo Llama3.1 sobre la partición de desarrollo del dataset SPGExt.
    La matriz incluye las 28 etiquetas en orden fijo.
    """

    # ====================
    # LISTA FIJA DE 28 ETIQUETAS EN ORDEN SOLICITADO
    # ====================
    base_entities = [
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
    
    # Construir etiquetas BIO en orden: primero todos los B-, luego todos los I-, luego O
    labels = [f"B-{ent}" for ent in base_entities] + [f"I-{ent}" for ent in base_entities] + ["O"]
    
    print(f"Total de etiquetas: {len(labels)}")
    print(f"Orden de etiquetas: {labels}")
    
    # ====================
    # CARGAR ANOTACIONES Y GENERAR VECTORES L_TRUE Y L_PRED
    # ====================
    print(f"Cargando anotaciones desde: {gold_dir} y {system_dir}")
    
    # Importar clases de anotación con ruta ajustada
    import sys
    sys.path.append('../')
    from classes import BratAnnotation
    
    # Cargar anotaciones gold standard
    gold_ann = {}
    for filename in os.listdir(gold_dir):
        if filename.endswith(".ann"):
            ann_path = os.path.join(gold_dir, filename)
            annotation = BratAnnotation(ann_path)
            gold_ann[annotation.id] = annotation
    
    # Cargar anotaciones del sistema
    sys_ann = {}
    for filename in os.listdir(system_dir):
        if filename.endswith(".ann"):
            ann_path = os.path.join(system_dir, filename)
            annotation = BratAnnotation(ann_path)
            sys_ann[annotation.id] = annotation
    
    # Generar vectores l_true y l_pred usando la misma metodología que el modelo
    print("Generando vectores l_true y l_pred usando metodología del modelo Llama3.1...")
    y_true, y_pred = get_token_level_predictions(gold_ann, sys_ann)
    
    print(f"Etiquetas verdaderas generadas: {len(y_true)}")
    print(f"Etiquetas predichas generadas: {len(y_pred)}")
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Error: número de etiquetas verdaderas ({len(y_true)}) != número de predichas ({len(y_pred)})")
    
    # ====================
    # GUARDAR VECTORES L_TRUE Y L_PRED POR PARTICION
    # ====================
    print(f"Guardando vectores l_true_{partition}.txt y l_pred_{partition}.txt...")
    with open(f'l_true_{partition}.txt', 'w') as f:
        for label in y_true:
            f.write(f"{label}\n")
    
    with open(f'l_pred_{partition}.txt', 'w') as f:
        for label in y_pred:
            f.write(f"{label}\n")
    
    print(f"Guardados {len(y_true)} etiquetas verdaderas y {len(y_pred)} predichas en l_true_{partition}.txt y l_pred_{partition}.txt")
    
    # ====================
    # GENERAR MATRIZ DE CONFUSIÓN
    # ====================
    print("Generando matriz de confusión...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # ====================
    # GUARDAR MATRIZ CON FORMATO ESPECÍFICO
    # ====================
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "plots"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalizar la matriz por fila
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

    # Eliminar filas y columnas de etiquetas cuya suma de fila es 0 (toda la fila violeta)
    rows_to_keep = np.where(cm.sum(axis=1) > 0)[0]
    cm = cm[rows_to_keep][:, rows_to_keep]
    cm_normalized = cm_normalized[rows_to_keep][:, rows_to_keep]
    labels = [labels[i] for i in rows_to_keep]
    
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
    
    # Título específico para Llama3.1 SPGExt con partición dinámica
    partition_name = "desarrollo" if partition == "dev" else "evaluación"
    titulo = f"Matriz de confusión del modelo Llama3.1\nsobre la partición de {partition_name} del dataset SPGExt"
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
    output_filename = f"{output_prefix}_{partition}.png"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Guardando matriz en: {output_path}")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    # ====================
    # REPORTE DETALLADO
    # ====================
    print("\n=== REPORTE DETALLADO ===")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))
    
    # Métricas por entidad
    print("\n=== MÉTRICAS POR TIPO DE ENTIDAD ===")
    entity_metrics = {}
    
    for true_label, pred_label in zip(y_true, y_pred):
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
            
            print(f"\nRESUMEN GENERAL:")
            print(f"Total entidades: {total_entities}")
            print(f"Total predichas: {total_predicted}")
            print(f"Total correctas: {total_correct}")
            print(f"Macro Precision: {macro_precision:.4f}")
            print(f"Macro Recall: {macro_recall:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
        else:
            print(f"\nRESUMEN GENERAL:")
            print(f"Total entidades: {total_entities}")
            print(f"Total predichas: {total_predicted}")
            print(f"Total correctas: {total_correct}")
            print("No hay entidades con métricas calculadas para macro-averaging")
    
    print(f"\nMatriz generada exitosamente con {len(labels)} etiquetas en orden fijo.")
    print(f"Archivos guardados en: {output_dir}")
    print(f"Vectores l_true_{partition}.txt y l_pred_{partition}.txt guardados en el directorio actual.")

def main():
    parser = argparse.ArgumentParser(description="Genera matriz de confusión NER del modelo Llama3.1 sobre las particiones dev/test del dataset SPGExt con 28 etiquetas en orden fijo")
    parser.add_argument("gold_dir", help="Directorio con anotaciones gold standard (.ann)")
    parser.add_argument("system_dir", help="Directorio con anotaciones del sistema (.ann)")
    parser.add_argument("--output-prefix", default="confusion_matrix", help="Prefijo para archivos de salida")
    parser.add_argument("--partition", choices=["dev", "test"], default="dev", help="Partición del dataset (dev o test)")
    parser.add_argument("--cleaned", action="store_true", help="Indica que se usan datos limpiados")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gold_dir):
        print(f"Error: Directorio {args.gold_dir} no encontrado")
        sys.exit(1)
    
    if not os.path.exists(args.system_dir):
        print(f"Error: Directorio {args.system_dir} no encontrado")
        sys.exit(1)
    
    generate_confusion_matrix_from_annotations(args.gold_dir, args.system_dir, args.output_prefix, args.partition, args.cleaned)

if __name__ == "__main__":
    main() 