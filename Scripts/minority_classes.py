import os
from collections import Counter

# Rutas a los datasets
DATASETS = [
    "Datasets/SPG/dev/brat",
]

def contar_clases_brat(brat_dir):
    conteo = Counter()
    for fname in os.listdir(brat_dir):
        if fname.endswith(".ann"):
            with open(os.path.join(brat_dir, fname), encoding="utf8") as f:
                for line in f:
                    if line.startswith("T"):
                        parts = line.strip().split()
                        if len(parts) > 1:
                            clase = parts[1]
                            conteo[clase] += 1
    return conteo

def main():
    total_conteo = Counter()
    for dataset in DATASETS:
        conteo = contar_clases_brat(dataset)
        total_conteo.update(conteo)

    print("\n=== Conteo total de clases en todos los datasets ===")
    for clase, cantidad in total_conteo.most_common():
        print(f"{clase}: {cantidad}")

if __name__ == "__main__":
    main()