import os
import re
import uuid
import shutil

# Carpetas (cambiar si es necesario)
ANN_DIR = "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/b) SPG/brat"
INTEGRAL_DIR = "/home/usuario/Documentos/TrabajoEspecial/Datasets/SPG/integral"

def natural_key(s: str):
    """Orden 'humano': 1,2,10 en vez de 1,10,2"""
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", s)]

def main():
    # Archivos .ann
    ann_files = [f for f in os.listdir(ANN_DIR) if f.lower().endswith(".ann")]
    integral_ann_files = [f for f in os.listdir(INTEGRAL_DIR) if f.lower().endswith(".ann")]

    ann_files.sort(key=natural_key)
    integral_ann_files.sort(key=natural_key)

    if len(ann_files) != len(integral_ann_files):
        print("⚠️ Error: las carpetas no tienen la misma cantidad de .ann")
        print(f"{ANN_DIR}: {len(ann_files)}  |  {INTEGRAL_DIR}: {len(integral_ann_files)}")
        return

    tmp_suffix = f".__tmp_ren_{uuid.uuid4().hex}__"

    print("Plan de renombrado y copiado:\n")
    for old_name, new_name in zip(ann_files, integral_ann_files):
        new_txt = os.path.splitext(new_name)[0] + ".txt"
        print(f"  {old_name}  →  {new_name}  + copia {new_txt}")
    print("\nIniciando...\n")

    # Fase 1: mover a nombres temporales
    tmp_names = []
    for old_name in ann_files:
        src = os.path.join(ANN_DIR, old_name)
        tmp = os.path.join(ANN_DIR, old_name + tmp_suffix)
        os.replace(src, tmp)
        tmp_names.append((tmp, old_name))

    # Fase 2: renombrar definitivos y copiar txt
    try:
        for (tmp_path, _old_name), target_ann in zip(tmp_names, integral_ann_files):
            dst_ann = os.path.join(ANN_DIR, target_ann)
            os.replace(tmp_path, dst_ann)
            print(f"✔️  {os.path.basename(tmp_path).replace(tmp_suffix, '')} → {target_ann}")

            # copiar su .txt
            target_txt = os.path.splitext(target_ann)[0] + ".txt"
            src_txt = os.path.join(INTEGRAL_DIR, target_txt)
            dst_txt = os.path.join(ANN_DIR, target_txt)
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
                print(f"   Copiado: {target_txt}")
            else:
                print(f"   ⚠️ No se encontró {target_txt} en {INTEGRAL_DIR}")
    except Exception as e:
        print(f"\n❌ Error: {e}\nRevirtiendo...")
        for tmp_path, old_name in tmp_names:
            if os.path.exists(tmp_path):
                os.replace(tmp_path, os.path.join(ANN_DIR, old_name))
        print("Reversión completa.")
        return

    print("\n✅ Proceso finalizado. Los .ann fueron renombrados y los .txt copiados a 'brat/'.")

if __name__ == "__main__":
    main()
