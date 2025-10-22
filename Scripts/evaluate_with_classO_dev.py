#!/usr/bin/env python3
"""
Script para calcular el rendimiento de todos los modelos sobre los conjuntos dev y test
de todos los datasets, considerando la clase O en las métricas.

Este script evalúa:
- REGEX
- BiLSTM-CRF  
- Llama3.1

Sobre los datasets:
- MEDDOCAN
- SPG
- SPG Extendido
- CARMEN-I

En las particiones:
- dev (desarrollo)
- test (evaluación)

Considerando la clase O en el cálculo de métricas.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import re
from collections import defaultdict

def load_annotations_from_directory(ann_dir):
    """Carga anotaciones desde un directorio"""
    import sys
    # Agregar ruta para importar clases desde BiLSTM-CRF
    sys.path.append('/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/a) MEDDOCAN/code/')
    from classes import BratAnnotation, NER_Evaluation, EntityTypeMetrics
    
    annotations = {}
    for filename in os.listdir(ann_dir):
        if filename.endswith(".ann"):
            ann_path = os.path.join(ann_dir, filename)
            try:
                annotation = BratAnnotation(ann_path)
                annotations[annotation.id] = annotation
            except Exception as e:
                print(f"Error cargando {ann_path}: {e}")
    return annotations

def evaluate_model_on_dataset(model_name, dataset_name, gold_dir, system_dir):
    """Evalúa un modelo en un dataset específico usando la misma metodología que evaluate.py"""
    
    try:
        # Cargar anotaciones
        gold_ann = load_annotations_from_directory(gold_dir)
        sys_ann = load_annotations_from_directory(system_dir)
        
        if not gold_ann:
            print(f"  Error: No se encontraron anotaciones gold en {gold_dir}")
            return None
        if not sys_ann:
            print(f"  Error: No se encontraron anotaciones system en {system_dir}")
            return None
        
        # Importar clases necesarias
        sys.path.append('/home/usuario/Documentos/TrabajoEspecial/Modelos/BiLSTM-CRF/a) MEDDOCAN/code/')
        from classes import NER_Evaluation, EntityTypeMetrics
        
        # Evaluar usando la misma metodología que evaluate.py (SIN O)
        evaluation = NER_Evaluation(sys_ann, gold_ann)
        
        # Aggregate tp, fp, and fn globally for all documents (SIN O)
        global_tp = []
        global_fp = []
        global_fn = []
        
        for eval_subtrack in evaluation.evaluations:
            for i in range(len(eval_subtrack.doc_ids)):
                global_tp.extend(eval_subtrack.tp[i])
                global_fp.extend(eval_subtrack.fp[i])
                global_fn.extend(eval_subtrack.fn[i])
        
        # Calcular métricas SIN O (como en evaluate.py)
        metrics_without_o = EntityTypeMetrics.calculate_metrics(global_tp, global_fp, global_fn)
        
        # Calcular micro y macro SIN O
        total_tp = len(global_tp)
        total_fp = len(global_fp)
        total_fn = len(global_fn)
        micro_precision_without_o = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall_without_o = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1_without_o = (2 * micro_precision_without_o * micro_recall_without_o) / (micro_precision_without_o + micro_recall_without_o) if (micro_precision_without_o + micro_recall_without_o) > 0 else 0.0
        
        macro_precision_without_o = sum([v['precision'] for v in metrics_without_o.values()]) / len(metrics_without_o) if metrics_without_o else 0.0
        macro_recall_without_o = sum([v['recall'] for v in metrics_without_o.values()]) / len(metrics_without_o) if metrics_without_o else 0.0
        macro_f1_without_o = sum([v['f1'] for v in metrics_without_o.values()]) / len(metrics_without_o) if metrics_without_o else 0.0
        
        # Calcular métricas CON O
        # Para esto, necesitamos contar todos los tokens y calcular las métricas incluyendo O
        
        # Contar total de tokens en todos los documentos
        total_tokens = 0
        o_tp = 0  # Tokens O correctamente identificados como O
        o_fp = 0  # Tokens no-O incorrectamente identificados como O  
        o_fn = 0  # Tokens O incorrectamente identificados como no-O
        
        for doc_id in gold_ann:
            if doc_id in sys_ann:
                gold_doc = gold_ann[doc_id]
                sys_doc = sys_ann[doc_id]
                
                # Obtener texto y tokenizarlo
                text = gold_doc.text
                tokens = []
                for match in re.finditer(r'\S+', text):
                    tokens.append((match.group(), match.start(), match.end()))
                
                # Obtener spans de entidades
                gold_spans = gold_doc.get_phi()
                sys_spans = sys_doc.get_phi()
                
                # Crear etiquetas BIO para gold y system
                gold_labels = ["O"] * len(tokens)
                sys_labels = ["O"] * len(tokens)
                
                # Marcar entidades en gold
                for label, start, end in gold_spans:
                    for i, (token, token_start, token_end) in enumerate(tokens):
                        if token_end <= start:
                            continue
                        if token_start >= end:
                            break
                        if token_start >= start and token_end <= end:
                            if gold_labels[i] == "O":
                                gold_labels[i] = "B-" + label
                            elif gold_labels[i].startswith("B-") or gold_labels[i].startswith("I-"):
                                gold_labels[i] = "I-" + label
                
                # Marcar entidades en system
                for label, start, end in sys_spans:
                    for i, (token, token_start, token_end) in enumerate(tokens):
                        if token_end <= start:
                            continue
                        if token_start >= end:
                            break
                        if token_start >= start and token_end <= end:
                            if sys_labels[i] == "O":
                                sys_labels[i] = "B-" + label
                            elif sys_labels[i].startswith("B-") or sys_labels[i].startswith("I-"):
                                sys_labels[i] = "I-" + label
                
                # Corregir etiquetas I- consecutivas
                for i in range(1, len(gold_labels)):
                    if gold_labels[i].startswith("B-") and gold_labels[i-1][2:] == gold_labels[i][2:]:
                        gold_labels[i] = "I-" + gold_labels[i][2:]
                
                for i in range(1, len(sys_labels)):
                    if sys_labels[i].startswith("B-") and sys_labels[i-1][2:] == sys_labels[i][2:]:
                        sys_labels[i] = "I-" + sys_labels[i][2:]
                
                # Contar métricas para O
                for i in range(len(tokens)):
                    if gold_labels[i] == "O" and sys_labels[i] == "O":
                        o_tp += 1
                    elif gold_labels[i] == "O" and sys_labels[i] != "O":
                        o_fn += 1
                    elif gold_labels[i] != "O" and sys_labels[i] == "O":
                        o_fp += 1
                
                total_tokens += len(tokens)
        
        # Calcular métricas para O
        o_precision = o_tp / (o_tp + o_fp) if (o_tp + o_fp) > 0 else 0.0
        o_recall = o_tp / (o_tp + o_fn) if (o_tp + o_fn) > 0 else 0.0
        o_f1 = (2 * o_precision * o_recall) / (o_precision + o_recall) if (o_precision + o_recall) > 0 else 0.0
        
        # Calcular micro y macro CON O
        total_tp_with_o = total_tp + o_tp
        total_fp_with_o = total_fp + o_fp
        total_fn_with_o = total_fn + o_fn
        
        micro_precision_with_o = total_tp_with_o / (total_tp_with_o + total_fp_with_o) if (total_tp_with_o + total_fp_with_o) > 0 else 0.0
        micro_recall_with_o = total_tp_with_o / (total_tp_with_o + total_fn_with_o) if (total_tp_with_o + total_fn_with_o) > 0 else 0.0
        micro_f1_with_o = (2 * micro_precision_with_o * micro_recall_with_o) / (micro_precision_with_o + micro_recall_with_o) if (micro_precision_with_o + micro_recall_with_o) > 0 else 0.0
        
        # Macro CON O incluye O en el promedio
        all_metrics_with_o = list(metrics_without_o.values()) + [{'precision': o_precision, 'recall': o_recall, 'f1': o_f1}]
        macro_precision_with_o = sum([v['precision'] for v in all_metrics_with_o]) / len(all_metrics_with_o) if all_metrics_with_o else 0.0
        macro_recall_with_o = sum([v['recall'] for v in all_metrics_with_o]) / len(all_metrics_with_o) if all_metrics_with_o else 0.0
        macro_f1_with_o = sum([v['f1'] for v in all_metrics_with_o]) / len(all_metrics_with_o) if all_metrics_with_o else 0.0

        return {
            'with_o': {
                'model': model_name,
                'dataset': dataset_name,
                'macro_precision': macro_precision_with_o,
                'macro_recall': macro_recall_with_o,
                'macro_f1': macro_f1_with_o,
                'micro_precision': micro_precision_with_o,
                'micro_recall': micro_recall_with_o,
                'micro_f1': micro_f1_with_o,
                'o_precision': o_precision,
                'o_recall': o_recall,
                'o_f1': o_f1,
                'total_tokens': total_tokens,
                'unique_labels': len(metrics_without_o) + 1,
                'class_metrics': metrics_without_o  # No incluye O en las métricas por clase
            },
            'without_o': {
                'model': model_name,
                'dataset': dataset_name,
                'macro_precision': macro_precision_without_o,
                'macro_recall': macro_recall_without_o,
                'macro_f1': macro_f1_without_o,
                'micro_precision': micro_precision_without_o,
                'micro_recall': micro_recall_without_o,
                'micro_f1': micro_f1_without_o,
                'total_tokens': total_tp + total_fp + total_fn,
                'unique_labels': len(metrics_without_o),
                'class_metrics': metrics_without_o
            }
        }
        
    except Exception as e:
        print(f"  Error evaluando {model_name} en {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_summary_report(all_results):
    """Genera un reporte resumen de todos los modelos"""
    
    # Crear DataFrame con resultados CON O
    data_with_o = []
    data_without_o = []
    
    for result in all_results:
        if result and result['with_o']:
            data_with_o.append({
                'Modelo': result['with_o']['model'],
                'Dataset': result['with_o']['dataset'],
                'Macro Precision': result['with_o']['macro_precision'],
                'Macro Recall': result['with_o']['macro_recall'],
                'Macro F1': result['with_o']['macro_f1'],
                'Micro Precision': result['with_o']['micro_precision'],
                'Micro Recall': result['with_o']['micro_recall'],
                'Micro F1': result['with_o']['micro_f1'],
                'O Precision': result['with_o']['o_precision'],
                'O Recall': result['with_o']['o_recall'],
                'O F1': result['with_o']['o_f1'],
                'Total Tokens': result['with_o']['total_tokens'],
                'Unique Labels': result['with_o']['unique_labels']
            })
        
        if result and result['without_o']:
            data_without_o.append({
                'Modelo': result['without_o']['model'],
                'Dataset': result['without_o']['dataset'],
                'Macro Precision': result['without_o']['macro_precision'],
                'Macro Recall': result['without_o']['macro_recall'],
                'Macro F1': result['without_o']['macro_f1'],
                'Micro Precision': result['without_o']['micro_precision'],
                'Micro Recall': result['without_o']['micro_recall'],
                'Micro F1': result['without_o']['micro_f1'],
                'Total Tokens': result['without_o']['total_tokens'],
                'Unique Labels': result['without_o']['unique_labels']
            })
    
    df_with_o = pd.DataFrame(data_with_o)
    df_without_o = pd.DataFrame(data_without_o)
    
    # Generar reporte por modelo CON O
    if not df_with_o.empty:
        model_summary_with_o = df_with_o.groupby('Modelo').agg({
            'Macro F1': 'mean',
            'Micro F1': 'mean',
            'O F1': 'mean',
            'Total Tokens': 'sum'
        }).round(4)
    
    # Generar reporte por modelo SIN O
    if not df_without_o.empty:
        model_summary_without_o = df_without_o.groupby('Modelo').agg({
            'Macro F1': 'mean',
            'Micro F1': 'mean',
            'Total Tokens': 'sum'
        }).round(4)
    
    # Generar reporte por dataset CON O
    if not df_with_o.empty:
        dataset_summary_with_o = df_with_o.groupby('Dataset').agg({
            'Macro F1': 'mean',
            'Micro F1': 'mean',
            'O F1': 'mean',
            'Total Tokens': 'sum'
        }).round(4)
    
    # Generar reporte por dataset SIN O
    if not df_without_o.empty:
        dataset_summary_without_o = df_without_o.groupby('Dataset').agg({
            'Macro F1': 'mean',
            'Micro F1': 'mean',
            'Total Tokens': 'sum'
        }).round(4)
    
    return df_with_o, df_without_o, model_summary_with_o if not df_with_o.empty else None, model_summary_without_o if not df_without_o.empty else None, dataset_summary_with_o if not df_with_o.empty else None, dataset_summary_without_o if not df_without_o.empty else None

def print_results_summary(df_with_o, df_without_o, model_summary_with_o, model_summary_without_o, dataset_summary_with_o, dataset_summary_without_o, partition_name="DEV"):
    """Imprime un resumen de los resultados"""
    
    print("\n" + "="*80)
    print(f"RESUMEN DE EVALUACIÓN - CONJUNTO {partition_name}")
    print("="*80)
    
    # Resultados CON O
    if not df_with_o.empty:
        print("\n📊 RESULTADOS CON O (incluyendo clase O):")
        print("-" * 50)
        for model in df_with_o['Modelo'].unique():
            model_data = df_with_o[df_with_o['Modelo'] == model]
            print(f"\n{model}:")
            for _, row in model_data.iterrows():
                print(f"  {row['Dataset']}:")
                # print(f"    Macro F1: {row['Macro F1']:.4f}")
                print(f"    Micro F1: {row['Micro F1']:.4f}")
                # print(f"    O F1: {row['O F1']:.4f}")
        
        if model_summary_with_o is not None:
            print("\n🏆 PROMEDIO POR MODELO (CON O):")
            print("-" * 30)
            for model, row in model_summary_with_o.iterrows():
                print(f"{model}:")
                # print(f"  Macro F1: {row['Macro F1']:.4f}")
                print(f"  Micro F1: {row['Micro F1']:.4f}")
                # print(f"  O F1: {row['O F1']:.4f}")
            
            # Calcular promedio de los tres modelos
            avg_macro = model_summary_with_o['Macro F1'].mean()
            avg_micro = model_summary_with_o['Micro F1'].mean()
            avg_o = model_summary_with_o['O F1'].mean()
            print(f"\nPROMEDIO GENERAL:")
            # print(f"  Macro F1: {avg_macro:.4f}")
            print(f"  Micro F1: {avg_micro:.4f}")
            # print(f"  O F1: {avg_o:.4f}")
        
        if dataset_summary_with_o is not None:
            print("\n📈 PROMEDIO POR DATASET (CON O):")
            print("-" * 30)
            for dataset, row in dataset_summary_with_o.iterrows():
                print(f"{dataset}:")
                # print(f"  Macro F1: {row['Macro F1']:.4f}")
                print(f"  Micro F1: {row['Micro F1']:.4f}")
                # print(f"  O F1: {row['O F1']:.4f}")
            
            # Calcular promedio de los tres datasets
            avg_macro = dataset_summary_with_o['Macro F1'].mean()
            avg_micro = dataset_summary_with_o['Micro F1'].mean()
            avg_o = dataset_summary_with_o['O F1'].mean()
            print(f"\nPROMEDIO GENERAL:")
            # print(f"  Macro F1: {avg_macro:.4f}")
            print(f"  Micro F1: {avg_micro:.4f}")
            # print(f"  O F1: {avg_o:.4f}")
    
    # Resultados SIN O
    if not df_without_o.empty:
        print("\n📊 RESULTADOS SIN O (excluyendo clase O):")
        print("-" * 50)
        for model in df_without_o['Modelo'].unique():
            model_data = df_without_o[df_without_o['Modelo'] == model]
            print(f"\n{model}:")
            for _, row in model_data.iterrows():
                print(f"  {row['Dataset']}:")
                # print(f"    Macro F1: {row['Macro F1']:.4f}")
                print(f"    Micro F1: {row['Micro F1']:.4f}")
        
        if model_summary_without_o is not None:
            print("\n🏆 PROMEDIO POR MODELO (SIN O):")
            print("-" * 30)
            for model, row in model_summary_without_o.iterrows():
                print(f"{model}:")
                # print(f"  Macro F1: {row['Macro F1']:.4f}")
                print(f"  Micro F1: {row['Micro F1']:.4f}")
            
            # Calcular promedio de los tres modelos
            avg_macro = model_summary_without_o['Macro F1'].mean()
            avg_micro = model_summary_without_o['Micro F1'].mean()
            print(f"\nPROMEDIO GENERAL:")
            # print(f"  Macro F1: {avg_macro:.4f}")
            print(f"  Micro F1: {avg_micro:.4f}")
        
        if dataset_summary_without_o is not None:
            print("\n📈 PROMEDIO POR DATASET (SIN O):")
            print("-" * 30)
            for dataset, row in dataset_summary_without_o.iterrows():
                print(f"{dataset}:")
                # print(f"  Macro F1: {row['Macro F1']:.4f}")
                print(f"  Micro F1: {row['Micro F1']:.4f}")
            
            # Calcular promedio de los tres datasets
            avg_macro = dataset_summary_without_o['Macro F1'].mean()
            avg_micro = dataset_summary_without_o['Micro F1'].mean()
            print(f"\nPROMEDIO GENERAL:")
            # print(f"  Macro F1: {avg_macro:.4f}")
            print(f"  Micro F1: {avg_micro:.4f}")

def main():
    """Función principal"""
    
    # Obtener la ruta base del proyecto
    base_path = "/home/usuario/Documentos/TrabajoEspecial"
    
    # Configuración de rutas para cada modelo con rutas absolutas
    MODEL_CONFIGS = {
        "REGEX": {
            "dev": {
                "gold_dirs": [
                    f"{base_path}/Datasets/MEDDOCAN/dev/brat",
                    f"{base_path}/Datasets/SPG/dev/brat", 
                    f"{base_path}/Datasets/SPGExt/dev/brat",
                    f"{base_path}/Datasets/CARMEN-I/dev/brat"
                ],
                "system_dirs": [
                    f"{base_path}/Modelos/REGEX/a) MEDDOCAN/brat/dev",
                    f"{base_path}/Modelos/REGEX/b) SPG/brat/dev",
                    f"{base_path}/Modelos/REGEX/c) SPGExt/brat/dev",
                    f"{base_path}/Modelos/REGEX/d) CARMEN-I/brat/dev"
                ]
            },
            "test": {
                "gold_dirs": [
                    f"{base_path}/Datasets/MEDDOCAN/test/brat",
                    f"{base_path}/Datasets/SPG/test/brat", 
                    f"{base_path}/Datasets/SPGExt/test/brat",
                    f"{base_path}/Datasets/CARMEN-I/test/brat"
                ],
                "system_dirs": [
                    f"{base_path}/Modelos/REGEX/a) MEDDOCAN/brat/test",
                    f"{base_path}/Modelos/REGEX/b) SPG/brat/test",
                    f"{base_path}/Modelos/REGEX/c) SPGExt/brat/test",
                    f"{base_path}/Modelos/REGEX/d) CARMEN-I/brat/test"
                ]
            }
        },
        "BiLSTM-CRF": {
            "dev": {
                "gold_dirs": [
                    f"{base_path}/Datasets/MEDDOCAN/dev/brat",
                    f"{base_path}/Datasets/SPG/dev/brat",
                    f"{base_path}/Datasets/SPGExt/dev/brat",
                    f"{base_path}/Datasets/CARMEN-I/dev/brat"
                ],
                "system_dirs": [
                    f"{base_path}/Modelos/BiLSTM-CRF/a) MEDDOCAN/dev/system",
                    f"{base_path}/Modelos/BiLSTM-CRF/b) SPG/dev/system", 
                    f"{base_path}/Modelos/BiLSTM-CRF/c) SPGExt/dev/system",
                    f"{base_path}/Modelos/BiLSTM-CRF/d) CARMEN-I/dev/system"
                ]
            },
            "test": {
                "gold_dirs": [
                    f"{base_path}/Datasets/MEDDOCAN/test/brat",
                    f"{base_path}/Datasets/SPG/test/brat",
                    f"{base_path}/Datasets/SPGExt/test/brat",
                    f"{base_path}/Datasets/CARMEN-I/test/brat"
                ],
                "system_dirs": [
                    f"{base_path}/Modelos/BiLSTM-CRF/a) MEDDOCAN/output/test/system",
                    f"{base_path}/Modelos/BiLSTM-CRF/b) SPG/output/test/system", 
                    f"{base_path}/Modelos/BiLSTM-CRF/c) SPGExt/output/test/system",
                    f"{base_path}/Modelos/BiLSTM-CRF/d) CARMEN-I/output/test/system"
                ]
            }
        },
        "Llama3.1": {
            "dev": {
                "gold_dirs": [
                    f"{base_path}/Datasets/MEDDOCAN/dev/brat",
                    f"{base_path}/Datasets/SPG/dev/brat",
                    f"{base_path}/Datasets/SPGExt/dev/brat",
                    f"{base_path}/Datasets/CARMEN-I/dev/brat"
                ],
                "system_dirs": [
                    f"{base_path}/Modelos/Llama3.1/a) MEDDOCAN/dev_prompt_OneShot/ann",
                    f"{base_path}/Modelos/Llama3.1/b) SPG/dev/ann",
                    f"{base_path}/Modelos/Llama3.1/c) SPGExt/dev/ann",
                    f"{base_path}/Modelos/Llama3.1/d) CARMEN-I/dev/ann"
                ]
            },
            "test": {
                "gold_dirs": [
                    f"{base_path}/Datasets/MEDDOCAN/test/brat",
                    f"{base_path}/Datasets/SPG/test/brat",
                    f"{base_path}/Datasets/SPGExt/test/brat",
                    f"{base_path}/Datasets/CARMEN-I/test/brat"
                ],
                "system_dirs": [
                    f"{base_path}/Modelos/Llama3.1/a) MEDDOCAN/test/ann",
                    f"{base_path}/Modelos/Llama3.1/b) SPG/test/ann",
                    f"{base_path}/Modelos/Llama3.1/c) SPGExt/test/ann",
                    f"{base_path}/Modelos/Llama3.1/d) CARMEN-I/test/ann"
                ]
            }
        }
    }
    
    DATASET_NAMES = ["MEDDOCAN", "SPG", "SPGExt", "CARMEN-I"]
    
    parser = argparse.ArgumentParser(description="Evalúa todos los modelos sobre los conjuntos dev y test de todos los datasets")
    parser.add_argument("--models", nargs="+", choices=["REGEX", "BiLSTM-CRF", "Llama3.1"], 
                       default=["REGEX", "BiLSTM-CRF", "Llama3.1"], 
                       help="Modelos a evaluar")
    parser.add_argument("--datasets", nargs="+", choices=["MEDDOCAN", "SPG", "SPGExt", "CARMEN-I"], 
                       default=["MEDDOCAN", "SPG", "SPGExt", "CARMEN-I"], 
                       help="Datasets a evaluar")
    parser.add_argument("--partitions", nargs="+", choices=["dev", "test"], 
                       default=["dev", "test"], 
                       help="Particiones a evaluar")
    
    args = parser.parse_args()
    
    print("🚀 INICIANDO EVALUACIÓN DE MODELOS")
    print("="*50)
    print(f"Modelos: {', '.join(args.models)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Particiones: {', '.join(args.partitions)}")
    print("="*50)
    
    # Evaluar cada partición por separado
    for partition in args.partitions:
        print(f"\n🔄 EVALUANDO PARTICION: {partition.upper()}")
        print("="*60)
        
        all_results = []
        
        # Evaluar cada modelo en cada dataset para esta partición
        for model_name in args.models:
            if model_name not in MODEL_CONFIGS:
                print(f"Error: Configuración no encontrada para modelo {model_name}")
                continue
                
            if partition not in MODEL_CONFIGS[model_name]:
                print(f"Error: Partición {partition} no encontrada para modelo {model_name}")
                continue
                
            config = MODEL_CONFIGS[model_name][partition]
            
            for i, dataset_name in enumerate(args.datasets):
                if i < len(config["gold_dirs"]) and i < len(config["system_dirs"]):
                    gold_dir = config["gold_dirs"][i]
                    system_dir = config["system_dirs"][i]
                    
                    # Verificar que existan los directorios
                    if not os.path.exists(gold_dir):
                        print(f"Error: Directorio gold {gold_dir} no encontrado")
                        continue
                    if not os.path.exists(system_dir):
                        print(f"Error: Directorio system {system_dir} no encontrado")
                        continue
                    
                    # Evaluar modelo en dataset
                    result = evaluate_model_on_dataset(model_name, dataset_name, gold_dir, system_dir)
                    if result:
                        all_results.append(result)
        
        if not all_results:
            print(f" No se pudieron obtener resultados para ningún modelo/dataset en {partition}")
            continue
        
        # Generar reportes para esta partición
        print(f"\n📋 Generando reportes para {partition}...")
        df_with_o, df_without_o, model_summary_with_o, model_summary_without_o, dataset_summary_with_o, dataset_summary_without_o = generate_summary_report(all_results)
        
        # Imprimir resumen para esta partición
        print_results_summary(df_with_o, df_without_o, model_summary_with_o, model_summary_without_o, dataset_summary_with_o, dataset_summary_without_o, partition.upper())
    
    print(f"\n✅ Evaluación completada para todas las particiones.")

if __name__ == "__main__":
    main() 