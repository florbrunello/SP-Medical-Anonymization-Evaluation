import pandas as pd
import re

# Texto con métricas extraído del análisis
text = """ """

# Orden objetivo para secciones/etiquetas clínicas
ordered_labels = [
    "CC",
    "IA_ANTECEDENTES",
    "IA_EVOL",
    "IA_EXPLORACION_CLINICA",
    "IA_EXPLORACION_COMPLEMENTARIA",
    "IA_INTERVENCION_QUIRURGICA",
    "IA_PLAN_TERAPEUTICO",
    "IA_PROCESO_ACTUAL",
    "IA_RADIOGRAFIA",
    "IA_SEGUIMIENTO",
    "IE",
    "IR",
    "IT_ANTECEDENTES",
    "IT_EVOL",
    "IT_EXPLORACION_CLINICA",
    "IT_EXPLORACION_COMPLEMENTARIA",
    "IT_INTERVENCION_QUIRURGICA",
    "IT_PLAN_TERAPEUTICO",
    "IT_PROCESO_ACTUAL",
    "IT_SEGUIMIENTO"
]

# Regex mejorado para capturar etiquetas con espacios y métricas
pattern = re.compile(r"^([A-Z0-9_]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", re.MULTILINE)
metrics = {}

for match in pattern.finditer(text):
    label, support, prec, rec, f1 = match.groups()
    label = label.strip()  # Limpiar espacios extra
    support_i = int(support) if support is not None else None
    prec_f, rec_f, f1_f = float(prec), float(rec), float(f1)
    metrics[label] = (support_i, prec_f, rec_f, f1_f)

# Regex separado para MICRO y MACRO (sin soporte)
pattern_micro_macro = re.compile(r"^(MICRO|MACRO)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", re.MULTILINE)

for match in pattern_micro_macro.finditer(text):
    label, prec, rec, f1 = match.groups()
    prec_f, rec_f, f1_f = float(prec), float(rec), float(f1)
    metrics[label] = (None, prec_f, rec_f, f1_f)

# Construcción del DataFrame
data = {
    "Tipo de Documento": [],
    "Documentos": [],
    "Precisión": [],
    "Recall": [],
    "F1": [],
}

# Para promediar (solo sobre etiquetas presentes con métricas)
sum_prec, sum_rec, sum_f1 = 0.0, 0.0, 0.0
count = 0
# Suma total de soportes (Docs)
Total_docs = 0

for label in ordered_labels:
    data["Tipo de Documento"].append(label)
    if label in metrics:
        support, prec, rec, f1 = metrics[label]
        data["Documentos"].append("" if support is None else support)
        data["Precisión"].append(f"{prec:.4f}".replace(".", ","))
        data["Recall"].append(f"{rec:.4f}".replace(".", ","))
        data["F1"].append(f"{f1:.4f}".replace(".", ","))
        sum_prec += prec
        sum_rec += rec
        sum_f1 += f1
        count += 1
        if support is not None:
            Total_docs += support
    else:
        data["Documentos"].append("")
        data["Precisión"].append("")
        data["Recall"].append("")
        data["F1"].append("")

# Agregar fila final con suma de Docs y promedio si hay entidades válidas
if count > 0:
    data["Tipo de Documento"].append("PROMEDIO")
    data["Documentos"].append(Total_docs)
    data["Precisión"].append(f"{(sum_prec/count):.4f}".replace(".", ","))
    data["Recall"].append(f"{(sum_rec/count):.4f}".replace(".", ","))
    data["F1"].append(f"{(sum_f1/count):.4f}".replace(".", ","))

# Agregar métricas micro y macro si están presentes
if "MICRO" in metrics:
    support, prec, rec, f1 = metrics["MICRO"]
    data["Tipo de Documento"].append("MICRO")
    data["Documentos"].append("")
    data["Precisión"].append(f"{prec:.4f}".replace(".", ","))
    data["Recall"].append(f"{rec:.4f}".replace(".", ","))
    data["F1"].append(f"{f1:.4f}".replace(".", ","))

if "MACRO" in metrics:
    support, prec, rec, f1 = metrics["MACRO"]
    data["Tipo de Documento"].append("MACRO")
    data["Documentos"].append("")
    data["Precisión"].append(f"{prec:.4f}".replace(".", ","))
    data["Recall"].append(f"{rec:.4f}".replace(".", ","))
    data["F1"].append(f"{f1:.4f}".replace(".", ","))

# Guardar en Excel en la ruta deseada
output_path = "/home/usuario/Documentos/TrabajoEspecial/Modelos/Scripts/output_2.xlsx"
df = pd.DataFrame(data)
df.to_excel(output_path, index=False)
print(f"Excel generado como '{output_path}' con promedio, micro y macro incluidos") 