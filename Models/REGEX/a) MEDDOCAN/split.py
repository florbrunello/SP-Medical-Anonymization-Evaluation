import os
import shutil

# Rutas
brat_source = '/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/a) MEDDOCAN/brat'
meddocan_base = '/home/usuario/Documentos/TrabajoEspecial/Datasets/MEDDOCAN'

# Subcarpetas destino
splits = ['train', 'dev', 'test']

# Crear subcarpetas de destino en brat/
for split in splits:
    os.makedirs(os.path.join(brat_source, split), exist_ok=True)

# Recorremos cada split
for split in splits:
    reference_dir = os.path.join(meddocan_base, split, "brat")
    target_dir = os.path.join(brat_source, split)

    # Obtener nombres base (sin extensión)
    file_basenames = {os.path.splitext(f)[0] for f in os.listdir(reference_dir) if f.endswith(".txt")}

    for base in file_basenames:
        for ext in [".txt", ".ann"]:
            src = os.path.join(brat_source, base + ext)
            dst = os.path.join(target_dir, base + ext)
            if os.path.exists(src):
                shutil.move(src, dst)
            else:
                print(f"⚠️ No se encontró: {src}")