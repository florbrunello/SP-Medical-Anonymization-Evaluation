import os

# Rutas a los datasets
brat_dir = "Datasets/SPGExt/dev/brat"
system_dir = "Modelos/BiLSTM-CRF/c) SPGExt/dev/system"

# Entidades a buscar
target_entities = {"ID_TITULACION_PERSONAL_SANITARIO"}

def get_entities_from_ann(file_path):
    entities = set()
    if not os.path.exists(file_path):
        return entities
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                entity_type = parts[1].split()[0]
                if entity_type in target_entities:
                    # Usamos el tipo, el span y el valor para máxima precisión
                    entities.add(tuple(parts[1:]))
    return entities

for fname in os.listdir(brat_dir):
    if not fname.endswith(".ann"):
        continue
    brat_path = os.path.join(brat_dir, fname)
    system_path = os.path.join(system_dir, fname)
    brat_entities = get_entities_from_ann(brat_path)
    system_entities = get_entities_from_ann(system_path)
    # Entidades que están en manual pero no en automática
    missing = brat_entities - system_entities
    if missing:
        print(f"\nArchivo: {fname}")
        print("Entidades presentes en manual pero faltantes en automática:")
        for ent in missing:
            print("  ", ent)