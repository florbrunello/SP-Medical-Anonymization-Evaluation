import os
import re
from bs4 import BeautifulSoup 

# Lista blanca de etiquetas permitidas
ALLOWED_TAGS = {
    "NOMBRE_SUJETO_ASISTENCIA", "EDAD_SUJETO_ASISTENCIA", "SEXO_SUJETO_ASISTENCIA",
    "FAMILIARES_SUJETO_ASISTENCIA", "NOMBRE_PERSONAL_SANITARIO", "FECHAS", "PROFESION",
    "HOSPITAL", "CENTRO_SALUD", "INSTITUCION", "CALLE", "TERRITORIO", "PAIS",
    "NUMERO_TELEFONO", "NUMERO_FAX", "CORREO_ELECTRONICO", "ID_SUJETO_ASISTENCIA",
    "ID_CONTACTO_ASISTENCIAL", "ID_ASEGURAMIENTO", "ID_TITULACION_PERSONAL_SANITARIO",
    "ID_EMPLEO_PERSONAL_SANITARIO", "IDENTIF_VEHICULOS_NRSERIE_PLACAS",
    "IDENTIF_DISPOSITIVOS_NRSERIE", "DIREC_PROT_INTERNET", "URL_WEB", "IDENTIF_BIOMETRICOS",
    "OTRO_NUMERO_IDENTIF", "OTROS_SUJETO_ASISTENCIA" 
}

def extract_tags(xml_text):
    soup = BeautifulSoup(xml_text, "lxml")
    tags = soup.find_all("tag")
    tag_data = []
    for tag in tags:
        tag_type = tag.get("type", "").strip()
        if tag_type in ALLOWED_TAGS:
            tag_data.append((tag_type, tag.text.strip()))
    return tag_data

def find_all_occurrences(text, substring):
    matches = []
    start = 0
    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        matches.append((start, start + len(substring)))
        start += len(substring)
    return matches

def process_file_pair(txt_path, xml_path, ann_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        plain_text = f.read()
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    tag_list = extract_tags(xml_content)
    used_offsets = set()
    ann_lines = []
    tid = 1
    for tag_type, value in tag_list:
        matches = find_all_occurrences(plain_text, value)
        for start, end in matches:
            if (start, end) not in used_offsets:
                used_offsets.add((start, end))
                ann_lines.append(f"T{tid}\t{tag_type} {start} {end}\t{value}")
                tid += 1
                break
        else:
            print(f" WARNING: No match found for '{value}' in {os.path.basename(txt_path)}")

    with open(ann_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ann_lines))

def process_folder(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".xml"):
            base = filename[:-4]
            xml_path = os.path.join(folder, f"{base}.xml")
            txt_path = os.path.join(folder, f"{base}.txt")
            ann_path = os.path.join(folder, f"{base}.ann")

            if os.path.exists(txt_path):
                process_file_pair(txt_path, xml_path, ann_path)
                print(f" Procesado: {base}")
            else:
                print(f" Falta el archivo .txt para {base}")

# CAMBIA ESTE PATH A TU CARPETA
carpeta = '/home/usuario/Descargas/in'
process_folder(carpeta)
