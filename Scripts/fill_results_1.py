import pandas as pd
import re

# Texto con métricas extraído del análisis
text = """ """

# Lista objetivo
ordered_labels = [
    "NOMBRE_SUJETO_ASISTENCIA", "EDAD_SUJETO_ASISTENCIA", "SEXO_SUJETO_ASISTENCIA",
    "FAMILIARES_SUJETO_ASISTENCIA", "NOMBRE_PERSONAL_SANITARIO", "FECHAS",
    "PROFESION", "HOSPITAL", "CENTRO_SALUD", "INSTITUCION", "CALLE", "TERRITORIO",
    "PAIS", "NUMERO_TELEFONO", "NUMERO_FAX", "CORREO_ELECTRONICO", "ID_SUJETO_ASISTENCIA",
    "ID_CONTACTO_ASISTENCIAL", "ID_ASEGURAMIENTO", "ID_TITULACION_PERSONAL_SANITARIO",
    "ID_EMPLEO_PERSONAL_SANITARIO", "IDENTIF_VEHICULOS_NRSERIE_PLACAS", "IDENTIF_DISPOSITIVOS_NRSERIE",
    "DIREC_PROT_INTERNET", "URL_WEB", "IDENTIF_BIOMETRICOS", "OTRO_NUMERO_IDENTIF", "OTROS_SUJETO_ASISTENCIA"
]

# Extraer métricas (ahora incluye accuracy)
pattern = re.compile(r'([A-Z_]+)\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
metrics = {}

for match in pattern.finditer(text):
    label, prec, rec, f1, acc = match.groups()
    prec_f, rec_f, f1_f, acc_f = float(prec), float(rec), float(f1), float(acc)
    metrics[label] = (prec_f, rec_f, f1_f, acc_f)

# Construcción del DataFrame
data = {
    "Etiqueta": [],
    "Precisión": [],
    "Recall": [],
    "F1": [], 
    "Accuracy": []
}

# Para promediar
sum_prec, sum_rec, sum_f1, sum_acc = 0, 0, 0, 0
count = 0

for label in ordered_labels:
    data["Etiqueta"].append(label)
    if label in metrics:
        prec, rec, f1, acc = metrics[label]
        data["Precisión"].append(f"{prec:.4f}".replace(".", ","))
        data["Recall"].append(f"{rec:.4f}".replace(".", ","))
        data["F1"].append(f"{f1:.4f}".replace(".", ","))
        data["Accuracy"].append(f"{acc:.4f}".replace(".", ","))
        # Acumular para promedio
        sum_prec += prec
        sum_rec += rec
        sum_f1 += f1
        sum_acc += acc
        count += 1
    else:
        data["Precisión"].append("")
        data["Recall"].append("")
        data["F1"].append("")
        data["Accuracy"].append("")

# Agregar promedio si hay entidades válidas
if count > 0:
    data["Etiqueta"].append("PROMEDIO")
    data["Precisión"].append(f"{(sum_prec/count):.4f}".replace(".", ","))
    data["Recall"].append(f"{(sum_rec/count):.4f}".replace(".", ","))
    data["F1"].append(f"{(sum_f1/count):.4f}".replace(".", ","))
    data["Accuracy"].append(f"{(sum_acc/count):.4f}".replace(".", ","))

# Guardar en Excel en la ruta deseada
output_path = "/home/usuario/Documentos/TrabajoEspecial/Modelos/Scripts/output_1.xlsx"
df = pd.DataFrame(data)
df.to_excel(output_path, index=False)
print(f"Excel generado como '{output_path}' con promedio incluido.")
