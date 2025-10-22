import csv
import os
import re
import unicodedata
import datetime

# Lista para almacenar entidades UNKNOWN
unknown_entities = []

LABEL_MAP = {
    'XFECHAX':             'FECHAS',
    'XTELEFONOX':          'NUMERO_TELEFONO',
    'XCORREO_ELECTRONICOX':'CORREO_ELECTRONICO',
    'XNUM_DNIX':           'ID_SUJETO_ASISTENCIA',
    'XNUM_CUIT_CUILX':     'OTRO_NUMERO_IDENTIF',
    'XPASAPORTEX':         'OTRO_NUMERO_IDENTIF',
    'XMATRICULAX':         'ID_TITULACION_PERSONAL_SANITARIO',
    'XPATENTEX':           'IDENTIF_VEHICULOS_NRSERIE_PLACAS',
    'XOTROS_NUMX':         'OTRO_NUMERO_IDENTIF',
    'XDIRECCIONX':         'CALLE',
    'XCAPSX':              'CENTRO_SALUD',
    'XHOSPX':              'HOSPITAL',
    'XINSTITUCIONX':       'INSTITUCION',
    'XTURNOX':             'OTROS_SUJETO_ASISTENCIA',
    'XZONAX':              'TERRITORIO',
    'XPAISX':              'PAIS',
    'XPERSONAX':           'NOMBRE_SUJETO_ASISTENCIA',
    'XEPoFX':              'OTROS_SUJETO_ASISTENCIA'
}

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def generate_ann_strict(original_text, anonymized_text, lookahead=50, max_search=200):
    ann_list = []
    tid = 1
    i_orig = 0
    i_anon = 0

    placeholder_pattern = re.compile(r'X[A-Za-z_]+X')

    while i_anon < len(anonymized_text) and i_orig < len(original_text):
        m = placeholder_pattern.match(anonymized_text[i_anon:])
        if m:
            placeholder = m.group()
            # Clasificar como UNKNOWN todo lo que no esté en LABEL_MAP
            if placeholder in LABEL_MAP:
                label = LABEL_MAP[placeholder]
            else:
                label = 'UNKNOWN'
                # Registrar la entidad UNKNOWN
                unknown_entities.append({
                    'placeholder': placeholder,
                    'label': label,
                    'doc_id': f"doc_{len(unknown_entities) + 1}",
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            start_offset = i_orig

            # Determine next context after placeholder
            next_anon_index = i_anon + len(placeholder)
            next_context = ''
            j = next_anon_index
            while j < len(anonymized_text) and not placeholder_pattern.match(anonymized_text[j:]):
                next_context += anonymized_text[j]
                if len(next_context) >= lookahead:
                    break
                j += 1

            # Slide in original_text to find where next_context appears (accent-insensitive)
            end_offset = i_orig
            found = False
            norm_next_context = strip_accents(next_context)
            for offset in range(i_orig, min(i_orig + max_search, len(original_text))):
                orig_slice = original_text[offset:offset+len(next_context)]
                if strip_accents(orig_slice) == norm_next_context:
                    end_offset = offset
                    found = True
                    break

            if not found:
                # fallback: end at next space
                space_pos = original_text[i_orig:].find(' ')
                end_offset = i_orig + space_pos if space_pos != -1 else len(original_text)

            entity_text = original_text[start_offset:end_offset].strip()
            if entity_text:
                # Generar línea con etiqueta UNKNOWN solo si está en LABEL_MAP
                if placeholder in LABEL_MAP:
                    ann_list.append(f"T{tid}\t{label} {start_offset} {end_offset}\t{entity_text}")
                    tid += 1

            i_anon += len(placeholder)
            i_orig = end_offset
        else:
            # accent-insensitive comparison
            if strip_accents(anonymized_text[i_anon]) == strip_accents(original_text[i_orig]):
                i_anon += 1
                i_orig += 1
            else:
                # skip unmatched original chars until we realign
                i_orig += 1

    return ann_list


# ---- Process CSV and generate .ann files ----

csv_file = "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/c) SPGExt/out_post.csv"
ann_dir = "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/c) SPGExt/brat"
os.makedirs(ann_dir, exist_ok=True)

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader, start=1):
        original_text = row['original']
        anonymized_text = row['anonymized']

        ann_list = generate_ann_strict(original_text, anonymized_text)

        # Save .ann file
        ann_path = os.path.join(ann_dir, f"doc_{idx}.ann")
        with open(ann_path, 'w', encoding='utf-8') as ann_f:
            ann_f.write("\n".join(ann_list))

# Imprimir entidades UNKNOWN
if unknown_entities:
    print(f"\n{'='*60}")
    print(f"ENTIDADES UNKNOWN - {len(unknown_entities)} encontradas")
    print(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Archivo CSV procesado: {csv_file}")
    print(f"{'='*60}")
    print("Lista de entidades UNKNOWN:")
    print(f"{'Nº':<4} {'Placeholder':<25} {'Doc':<10} {'Timestamp'}")
    print("-" * 60)
    for i, entity in enumerate(unknown_entities, 1):
        print(f"{i:<4} {entity['placeholder']:<25} {entity['doc_id']:<10} {entity['timestamp']}")
    print(f"{'='*60}")
    print(f"Total de entidades UNKNOWN encontradas: {len(unknown_entities)}")
else:
    print("\nNo se encontraron entidades UNKNOWN")