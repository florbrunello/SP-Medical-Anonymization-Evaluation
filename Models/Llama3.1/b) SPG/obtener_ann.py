import os
import re
from bs4 import BeautifulSoup 


def normalize_text(text):
    """Normalize text for better matching"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[.,;:!?]', '', text)  # Remove punctuation
    text = text.lower()
    return text

def extract_tags(xml_text):
    soup = BeautifulSoup(xml_text, "html.parser")
    tags = soup.find_all("tag")
    tag_data = []
    for tag in tags:
        tag_type = tag.get("type", "").strip()
        # Procesar las etiquetas encontradas
        if tag_type:  # Solo verificar que no esté vacío
            text = tag.text.strip()
            tag_data.append((tag_type, text))
    return tag_data

def find_matches(text, substring):
    """Find matches with multiple matching strategies"""
    matches = []
    
    # Strategy 1: Exact match
    start = 0
    sub_len = len(substring)
    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        matches.append((start, start + sub_len))
        start += sub_len
    
    # Strategy 2: Case-insensitive match
    if not matches:
        normalized_text = text.lower()
        normalized_sub = substring.lower()
        start = 0
        sub_len = len(normalized_sub)
        while True:
            start = normalized_text.find(normalized_sub, start)
            if start == -1:
                break
            matches.append((start, start + sub_len))
            start += sub_len
    
    # Strategy 3: Partial match (for medical terms with possible typos)
    if not matches and len(substring.split()) > 1:
        words = substring.split()
        # Try matching first word + part of second word
        partial_pattern = re.compile(
            re.escape(words[0]) + r'.{0,10}?' + re.escape(words[1][:3]),
            re.IGNORECASE
        )
        for match in partial_pattern.finditer(text):
            matches.append((match.start(), match.end()))
    
    return matches

def process_file_pair(txt_path, xml_path, ann_path):
    try:
        with open(txt_path, "r", encoding="utf-8", errors='replace') as f:
            plain_text = f.read()
        with open(xml_path, "r", encoding="utf-8", errors='replace') as f:
            xml_content = f.read()

        tag_list = extract_tags(xml_content)
        used_offsets = set()
        ann_lines = []
        tid = 1
        
        # Contador para estadísticas
        total_tags = len(tag_list)
        processed_tags = 0
        
        for tag_type, value in tag_list:
            if not value:
                continue
                
            processed_tags += 1
            print(f" INFO: Procesando tag '{tag_type}' con valor '{value}' en {os.path.basename(txt_path)}")
                
            matches = find_matches(plain_text, value)
            if not matches:
                print(f" WARNING: No match found for '{value}' (tipo: {tag_type}) in {os.path.basename(txt_path)}")
                continue
                
            for start, end in matches:
                if (start, end) not in used_offsets:
                    used_offsets.add((start, end))
                    matched_text = plain_text[start:end]
                    ann_lines.append(f"T{tid}\t{tag_type} {start} {end}\t{matched_text}")
                    tid += 1
                    break

        # Mostrar estadísticas al final
        print(f" INFO: {os.path.basename(txt_path)} - Tags procesados: {processed_tags}, Total encontrados: {total_tags}")

        with open(ann_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ann_lines))
            
    except Exception as e:
        print(f" ERROR processing {os.path.basename(txt_path)}: {str(e)}")

def process_folder(folder):
    try:
        # Crear carpeta ann si no existe
        ann_folder = os.path.join(os.path.dirname(folder), "ann")
        if not os.path.exists(ann_folder):
            os.makedirs(ann_folder)
            print(f"📁 Carpeta creada: {ann_folder}")
        
        files = sorted([f for f in os.listdir(folder) if f.endswith(".xml")])
        total = len(files)
        processed = 0
        
        for filename in files:
            base = filename[:-4]
            xml_path = os.path.join(folder, f"{base}.xml")
            txt_path = os.path.join(folder, f"{base}.txt")
            ann_path = os.path.join(ann_folder, f"{base}.ann")

            if os.path.exists(txt_path):
                process_file_pair(txt_path, xml_path, ann_path)
                processed += 1
                print(f" Procesado: {base} ({processed}/{total})")
            else:
                print(f" Falta el archivo .txt para {base}")
                
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario. Guardando progreso...")
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")
    finally:
        print(f"\nProceso completado. Archivos procesados: {processed}/{total}")
        print(f"📁 Archivos .ann guardados en: {ann_folder}")

if __name__ == "__main__":
    carpeta = '/home/usuario/Documentos/TrabajoEspecial/Modelos/Llama3.1/b) SPG/dev/xml'
    process_folder(carpeta)