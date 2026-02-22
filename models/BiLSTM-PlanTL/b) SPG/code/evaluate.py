# coding=utf-8
###############################################################################
#
#   Copyright 2019 Secretaría de Estado para el Avance Digital (SEAD)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#                        MEDDOCAN Evaluation Script
#
# This script is distributed as apart of the Medical Document Anonymization
# (MEDDOCAN) task. It is inspired on the evaluation script from the i2b2
# 2014 Cardiac Risk and Personal Health-care Information (PHI) tasks. It is
# intended to be used via command line:
#
# $> python evaluate.py [i2b2|brat] [ner|spans] GOLD SYSTEM
#
# It produces Precision, Recall and F1 (P/R/F1) and leak score measures for
# the NER subtrack and P/R/F1 for the SPAN subtrack. The latter includes a
# relaxed metric where the spans are merged if only non-alphanumerical
# characters are found between them.
#
# SYSTEM and GOLD may be individual files or also directories in which case
# all files in SYSTEM will be compared to files the GOLD directory based on
# their file names.
#
# Basic Examples:
#
# $> python evaluate.py i2b2 ner gold/01.xml system/run1/01.xml
#
# Evaluate the single system output file '01.xml' against the gold standard
# file '01.xml' NER subtrack. Input files in i2b2 format.
#
# $> python evaluate.py brat ner gold/01.ann system/run1/01.ann
#
# Evaluate the single system output file '01.ann' against the gold standard
# file '01.ann' NER subtrack. Input files in BRAT format.
#
# $> python evaluate.py i2b2 spans gold/ system/run1/
#
# Evaluate the set of system outputs in the folder system/run1 against the
# set of gold standard annotations in gold/ using the SPANS subtrack. Input
# files in i2b2 format.
#
# $> python evaluate.py brat ner gold/ system/run1/ system/run2/ system/run3/
#
# Evaluate the set of system outputs in the folder system/run1, system/run2
# and in the folder system/run3 against the set of gold standard annotations
# in gold/ using the NER subtrack. Input files in BRAT format.
import os
import argparse
from classes import i2b2Annotation, BratAnnotation, NER_Evaluation, Span_Evaluation, EntityTypeMetrics, BinaryEntityMetrics
from collections import defaultdict


def get_document_dict_by_system_id(system_dirs, annotation_format):
    """Takes a list of directories and returns annotations. """

    documents = defaultdict(lambda: defaultdict(int))

    for d in system_dirs:
        for fn in os.listdir(d):
            if fn.endswith(".ann") or fn.endswith(".xml"):
                sa = annotation_format(os.path.join(d, fn))
                documents[sa.sys_id][sa.id] = sa

    return documents


def evaluate(gs, system, annotation_format, subtrack, **kwargs):
    """Evaluate the system by calling either NER_evaluation or Span_Evaluation.
    'system' can be a list containing either one file,  or one or more
    directories. 'gs' can be a file or a directory. """

    gold_ann = {}
    evaluations = []

    # Strip verbose keyword if it exists
    try:
        verbose = kwargs['verbose']
        del kwargs['verbose']
    except KeyError:
        verbose = False

    # Handle if two files were passed on the command line
    if os.path.isfile(system[0]) and os.path.isfile(gs):
        if (system[0].endswith(".ann") and gs.endswith(".ann")) or \
                (system[0].endswith(".xml") or gs.endswith(".xml")):
            gs = annotation_format(gs)
            sys = annotation_format(system[0])
            e = subtrack({sys.id: sys}, {gs.id: gs}, **kwargs)
            e.print_docs()
            evaluations.append(e)

    # Handle the case where 'gs' is a directory and 'system' is a list of directories.
    elif all([os.path.isdir(sys) for sys in system]) and os.path.isdir(gs):
        # Get a dict of gold annotations indexed by id

        for filename in os.listdir(gs):
            if filename.endswith(".ann") or filename.endswith(".xml"):
                annotations = annotation_format(os.path.join(gs, filename))
                gold_ann[annotations.id] = annotations

        for system_id, system_ann in sorted(get_document_dict_by_system_id(system, annotation_format).items()):
            e = subtrack(system_ann, gold_ann, **kwargs)
            # e.print_report(verbose=verbose)  # Comentado para evitar duplicación de métricas
            evaluations.append(e)

    else:
        Exception("Must pass file file or [directory/]+ directory/"
                  "on command line!")

    return evaluations[0] if len(evaluations) == 1 else evaluations


# Orden fijo de entidades para imprimir métricas
ENTITY_ORDER = [
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


def print_entity_metrics_table(metrics, entity_order):
    """Print the metrics grouped by entity type table."""
    print("\nMetrics grouped by entity type:")
    print(f"{'Entity Type':<35}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Accuracy':>10}")
    print("-" * 75)
    
    # Separar entidades de ENTITY_ORDER y otras entidades
    entity_order_entities = {}
    other_entities = []
    
    for entity_type, values in metrics.items():
        if entity_type in entity_order:
            entity_order_entities[entity_type] = values
        else:
            other_entities.append(entity_type)
    
    # Imprimir entidades de ENTITY_ORDER en el orden especificado (incluso si tienen 0.0000)
    for entity_type in entity_order:
        if entity_type in entity_order_entities:
            values = entity_order_entities[entity_type]
            # Calcular accuracy span-level para esta entidad
            entity_total = values['tp'] + values['fp'] + values['fn']
            entity_accuracy = values['tp'] / entity_total if entity_total > 0 else 0.0
            print(f"{entity_type:<35}{values['precision']:>10.4f}{values['recall']:>10.4f}{values['f1']:>10.4f}{entity_accuracy:>10.4f}")
    
    # Agrupar y mostrar otras entidades como OTRAS_ETIQUETAS
    if other_entities:
        # Calcular métricas promedio para otras entidades
        other_precision = sum([metrics[et]['precision'] for et in other_entities]) / len(other_entities) if other_entities else 0.0
        other_recall = sum([metrics[et]['recall'] for et in other_entities]) / len(other_entities) if other_entities else 0.0
        other_f1 = sum([metrics[et]['f1'] for et in other_entities]) / len(other_entities) if other_entities else 0.0
        
        # Calcular accuracy promedio para otras entidades (span-level)
        other_accuracy = sum([metrics[et]['tp'] / (metrics[et]['tp'] + metrics[et]['fp'] + metrics[et]['fn']) if (metrics[et]['tp'] + metrics[et]['fp'] + metrics[et]['fn']) > 0 else 0.0 for et in other_entities]) / len(other_entities) if other_entities else 0.0
        print(f"{'OTRAS_ETIQUETAS (' + str(len(other_entities)) + ' tipos)':<35}{other_precision:>10.4f}{other_recall:>10.4f}{other_f1:>10.4f}{other_accuracy:>10.4f}")
    
        print("-" * 75)

    # Mostrar lista breve de OTRAS_ETIQUETAS
    if other_entities:
        print(f"\nOTRAS_ETIQUETAS incluye: {', '.join(other_entities[:5])}")
        if len(other_entities) > 5:
            print(f"Y {len(other_entities) - 5} entidades adicionales.")


def print_macro_report(num_docs, num_entity_types, macro_precision, macro_recall, macro_f1, macro_accuracy):
    """Print the macro average report."""
    print("\n" + "-"*60)
    print(f"{'Report (SYSTEM: system)':<60}")
    print("-"*60)
    print(f"{'SubTrack 1 [NER]':<35}{'Measure':<15}{'Macro':<20}")
    print("-"*60)
    print(f"{'Total (' + str(num_docs) + ' docs, ' + str(num_entity_types) + ' entity types)':<35}{'Precision':<15}{macro_precision:<10.4f}")
    print(f"{'':<35}{'Recall':<15}{macro_recall:<10.4f}")
    print(f"{'':<35}{'F1':<15}{macro_f1:<10.4f}")
    print(f"{'':<35}{'Accuracy':<15}{macro_accuracy:<10.4f}")
    print("-"*60)

def print_micro_report(num_docs, micro_precision, micro_recall, micro_f1, micro_accuracy):
    """Print the micro average report."""
    print("\n" + "-"*60)
    print(f"{'Report (SYSTEM: system)':<60}")
    print("-"*60)
    print(f"{'SubTrack 1 [NER]':<35}{'Measure':<15}{'Micro':<10}")
    print("-"*60)
    print(f"{'Total (' + str(num_docs) + ' docs)':<35}{'Precision':<15}{micro_precision:<10.4f}")
    print(f"{'':<35}{'Recall':<15}{micro_recall:<10.4f}")
    print(f"{'':<35}{'F1':<15}{micro_f1:<10.4f}")
    print(f"{'':<35}{'Accuracy':<15}{micro_accuracy:<10.4f}")
    print("-"*60)


def print_binary_metrics(binary_metrics):
    """Print the binary metrics: Entity vs Non-entity (O class)."""
    print("\n" + "-"*75)
    print(f"{'Classification: Entity vs Non-entity (O class)':<75}")
    print("-"*75)
    
    print(f"{'Entity Type':<35}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Accuracy':>10}")
    print("-" * 75)
    
    # Calculate span-level accuracy
    entity_accuracy = binary_metrics['entity']['tp'] / (binary_metrics['entity']['tp'] + binary_metrics['entity']['fp'] + binary_metrics['entity']['fn']) if (binary_metrics['entity']['tp'] + binary_metrics['entity']['fp'] + binary_metrics['entity']['fn']) > 0 else 0.0
    non_entity_accuracy = binary_metrics['o']['tp'] / (binary_metrics['o']['tp'] + binary_metrics['o']['fp'] + binary_metrics['o']['fn']) if (binary_metrics['o']['tp'] + binary_metrics['o']['fp'] + binary_metrics['o']['fn']) > 0 else 0.0
    
    entity_types_count = binary_metrics.get('distinct_entity_types', 'N/A')
    print(f"{'ENTITY (' + str(entity_types_count) + ' types)':<35}{binary_metrics['entity']['precision']:>10.4f}{binary_metrics['entity']['recall']:>10.4f}{binary_metrics['entity']['f1']:>10.4f}{entity_accuracy:>10.4f}")
    print(f"{'NON-ENTITY':<35}{binary_metrics['o']['precision']:>10.4f}{binary_metrics['o']['recall']:>10.4f}{binary_metrics['o']['f1']:>10.4f}{non_entity_accuracy:>10.4f}")
    print("-" * 75)
    
    # Print simplified error breakdown
    if binary_metrics['has_error_breakdown'] and 'error_breakdown' in binary_metrics:
        # Calculate total error rate for entity class
        entity_error_rate = 1 - binary_metrics['entity']['f1']
        
        # Entity errors breakdown
        entity_to_o = binary_metrics['error_breakdown']['entity_to_o_errors']
        entity_to_other = binary_metrics['error_breakdown']['entity_to_other_entity_errors']
        
        total_entity_errors = binary_metrics['error_breakdown']['total_entity_errors']
        
        if total_entity_errors > 0:
            entity_to_o_pct = binary_metrics['error_breakdown']['entity_to_o_percentage']
            entity_to_other_pct = binary_metrics['error_breakdown']['entity_to_other_entity_percentage']
            
            print(f"\n{'ERROR BREAKDOWN ANALYSIS':<80}")
            print("-" * 80)
            print(f"{'Error Type':<35}{'Decimal':<12}{'Percentage':<15}{'Description':<18}")
            print("-" * 80)
            print(f"{'Entity → O':<35}{entity_to_o_pct:<12.4f}{entity_to_o_pct:<15.2%}{'Missed':<18}")
            print(f"{'Entity → Other Entity':<35}{entity_to_other_pct:<12.4f}{entity_to_other_pct:<15.2%}{'Wrong Type':<18}")
            print("-" * 80)
            print(f"{'Total Entity Error Rate':<35}{entity_error_rate:<12.4f}{entity_error_rate:<15.2%}{'Combined':<18}")
            print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for the MEDDOCAN track.")

    parser.add_argument("format",
                        choices=["i2b2", "brat"],
                        help="Format")
    parser.add_argument("subtrack",
                        choices=["ner", "spans"],
                        help="Subtrack")
    parser.add_argument('-v', '--verbose',
                        help="List also scores for each document",
                        action="store_true")
    parser.add_argument("gs_dir",
                        help="Directory to load GS from")
    parser.add_argument("sys_dir",
                        help="Directories with system outputs (one or more)",
                        nargs="+")

    args = parser.parse_args()

    evaluation = evaluate(args.gs_dir, args.sys_dir, i2b2Annotation if args.format == "i2b2" else BratAnnotation,
            NER_Evaluation if args.subtrack == "ner" else Span_Evaluation,
            verbose=args.verbose)

    # Aggregate tp, fp, and fn globally for all documents
    global_tp = []
    global_fp = []
    global_fn = []
    num_docs = 0

    for eval_instance in evaluation if isinstance(evaluation, list) else [evaluation]:
        for eval_subtrack in eval_instance.evaluations:  # Accede a las evaluaciones individuales
            for i in range(len(eval_subtrack.doc_ids)):
                global_tp.extend(eval_subtrack.tp[i])
                global_fp.extend(eval_subtrack.fp[i])
                global_fn.extend(eval_subtrack.fn[i])
                num_docs += 1

    # Calculate and print metrics grouped by entity type
    metrics = EntityTypeMetrics.calculate_metrics(global_tp, global_fp, global_fn)

    # Get gold and system annotations for binary metrics calculation
    gold_ann = {}
    sys_ann = {}
    
    # Handle if two files were passed on the command line
    if os.path.isfile(args.sys_dir[0]) and os.path.isfile(args.gs_dir):
        if (args.sys_dir[0].endswith(".ann") and args.gs_dir.endswith(".ann")) or \
                (args.sys_dir[0].endswith(".xml") or args.gs_dir.endswith(".xml")):
            gs_annotation = i2b2Annotation if args.format == "i2b2" else BratAnnotation
            gold_ann = {gs_annotation(args.gs_dir).id: gs_annotation(args.gs_dir)}
            sys_ann = {gs_annotation(args.sys_dir[0]).id: gs_annotation(args.sys_dir[0])}
    
    # Handle the case where 'gs' is a directory and 'system' is a list of directories.
    elif all([os.path.isdir(sys) for sys in args.sys_dir]) and os.path.isdir(args.gs_dir):
        # Get a dict of gold annotations indexed by id
        for filename in os.listdir(args.gs_dir):
            if filename.endswith(".ann") or filename.endswith(".xml"):
                annotations = i2b2Annotation if args.format == "i2b2" else BratAnnotation
                ann = annotations(os.path.join(args.gs_dir, filename))
                gold_ann[ann.id] = ann
        
        # Get system annotations
        for d in args.sys_dir:
            for fn in os.listdir(d):
                if fn.endswith(".ann") or fn.endswith(".xml"):
                    annotations = i2b2Annotation if args.format == "i2b2" else BratAnnotation
                    ann = annotations(os.path.join(d, fn))
                    sys_ann[ann.id] = ann

    # Calculate span-level metrics (same granularity as EvaluateSubtrack1)
    metrics = EntityTypeMetrics.calculate_metrics(global_tp, global_fp, global_fn)
    
    # Calculate binary metrics: Entity vs Non-Entity (using span-level metrics)
    binary_metrics = BinaryEntityMetrics.calculate_binary_metrics(global_tp, global_fp, global_fn, gold_ann, sys_ann)

    # Calcular y mostrar el promedio micro (micro-averaging) - span level
    total_tp = len(global_tp)
    total_fp = len(global_fp)
    total_fn = len(global_fn)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    micro_accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    # Calcular y mostrar el promedio macro (macro-averaging) - span level
    macro_precision = sum([v['precision'] for v in metrics.values()]) / len(metrics) if metrics else 0.0
    macro_recall = sum([v['recall'] for v in metrics.values()]) / len(metrics) if metrics else 0.0
    macro_f1 = sum([v['f1'] for v in metrics.values()]) / len(metrics) if metrics else 0.0
    macro_accuracy = sum([v['tp'] / (v['tp'] + v['fp'] + v['fn']) if (v['tp'] + v['fp'] + v['fn']) > 0 else 0.0 for v in metrics.values()]) / len(metrics) if metrics else 0.0

    # Calcular promedio solo sobre entidades de ENTITY_ORDER (excluyendo OTRAS_ETIQUETAS) - span level
    entity_order_metrics = {k: v for k, v in metrics.items() if k in ENTITY_ORDER}
    entity_order_precision = sum([v['precision'] for v in entity_order_metrics.values()]) / len(entity_order_metrics) if entity_order_metrics else 0.0
    entity_order_recall = sum([v['recall'] for v in entity_order_metrics.values()]) / len(entity_order_metrics) if entity_order_metrics else 0.0
    entity_order_f1 = sum([v['f1'] for v in entity_order_metrics.values()]) / len(entity_order_metrics) if entity_order_metrics else 0.0

    # Calcular accuracy general (span-level): total de entidades correctas / total de entidades
    total_entities = total_tp + total_fp + total_fn
    accuracy_general = total_tp / total_entities if total_entities > 0 else 0.0
    
    # Print all reports using organized functions
    print_entity_metrics_table(metrics, ENTITY_ORDER)
    print_macro_report(num_docs, len(metrics), macro_precision, macro_recall, macro_f1, macro_accuracy)
    print_micro_report(num_docs, micro_precision, micro_recall, micro_f1, micro_accuracy)
    print_binary_metrics(binary_metrics)
