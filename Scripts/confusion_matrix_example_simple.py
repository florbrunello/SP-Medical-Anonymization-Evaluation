"""
Genera una matriz de confusión SINTÉTICA (de ejemplo) con el mismo formato
que las figuras "cleaned" del primer cuadrante (B-* y O), tal como
confusion_matrix_global_test_cleaned_BiLSTM-CRF_Q1.png.

- Etiquetas: B-<ENTIDAD> para un conjunto reducido (cleaned) + O
- Estilo: cmap viridis, vmin=0, vmax=1, figsize=(25, 18), xticklabels a 45°,
  colorbar horizontal y título en español
- Datos ficticios con: diagonal fuerte (aciertos), bloque de confusiones entre
  dos entidades (cuatro celdas), columna de O destacada y ruido general.

Se guardan dos imágenes en `Scripts/plots_example/`:
- confusion_matrix_example_simple.png
- confusion_matrix_example_simple_no_labels.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# Lista de entidades base (misma que en los generadores globales),
# pero luego se aplica la versión "cleaned" excluyendo 3 entidades que
# no aparecen: DIREC_PROT_INTERNET, URL_WEB, IDENTIF_BIOMETRICOS
ALL_ENTITIES = [
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
    "OTROS_SUJETO_ASISTENCIA",
]


def build_labels_cleaned_q1():
    """Devuelve labels B-* + O en el mismo orden que el Q1 cleaned."""
    cleaned_entities = [
        ent
        for ent in ALL_ENTITIES
        if ent not in {"DIREC_PROT_INTERNET", "URL_WEB", "IDENTIF_BIOMETRICOS"}
    ]
    b_labels = [f"B-{ent}" for ent in cleaned_entities]
    return b_labels + ["O"]


def fabricate_confusion_matrix(labels: list[str], random_seed: int = 7) -> np.ndarray:
    """
    Crea una matriz de confusión sintética normalizada por fila con:
    - Fuerte diagonal (aciertos predominantes)
    - Bloque de confusión entre dos entidades adyacentes (2x2)
    - Columna de O con valores apreciables (confusión con O)
    - Ruido suave en el resto
    """
    rng = np.random.default_rng(random_seed)
    n = len(labels)

    # Base: casi todos aciertos muy altos en la diagonal
    cm = np.eye(n) * 0.97

    # Ruido general muy bajo fuera de la diagonal
    noise = rng.uniform(0.0, 0.005, size=(n, n))
    np.fill_diagonal(noise, 0.0)
    cm += noise

    # 1) Un caso de confusión entre entidades como bloque 2x2 (cuatro cuadrados)
    #    Específicamente entre B-NUMERO_TELEFONO y B-NUMERO_FAX
    try:
        a = labels.index("B-NUMERO_TELEFONO")
        b = labels.index("B-NUMERO_FAX")
    except ValueError:
        # Fallback por si cambian etiquetas: usar dos posiciones vecinas válidas
        a, b = 13, min(14, n - 2)
    # Reajustar el 2x2: diagonales más bajas, off-diagonales altas
    cm[a, a] = max(0.12, cm[a, a] * 0.2)
    cm[b, b] = max(0.12, cm[b, b] * 0.2)
    cm[a, b] += 0.55
    cm[b, a] += 0.55

    # 2) Un caso de confusión con clase O (columna O)
    #    Específicamente: B-CENTRO_SALUD -> O
    col_o = n - 1
    try:
        r = labels.index("B-CENTRO_SALUD")
    except ValueError:
        r = 2 if n > 3 else 0
    # Dejar algo en la diagonal para que no quede tan oscuro (más amarillo)
    desired_remaining = 0.10
    current_diag = cm[r, r]
    move = max(0.0, min(0.95, current_diag - desired_remaining))
    cm[r, r] = max(0.0, current_diag - move)
    cm[r, col_o] += move

    # 3) Asegurar que B-FAMILIARES_SUJETO_ASISTENCIA sea amarillo (alto en diagonal)
    try:
        r_fam = labels.index("B-FAMILIARES_SUJETO_ASISTENCIA")
        cm[r_fam, :] = 0.0
        cm[r_fam, r_fam] = 0.98
        cm[r_fam, (r_fam + 1) % n] = 1e-6
    except ValueError:
        pass

    # 4) Asegurar que B-TERRITORIO tenga un amarillo claro (sin errores)
    try:
        r_terr = labels.index("B-TERRITORIO")
        keep = min(0.995, cm[r_terr, r_terr] + 0.2)
        # vaciar resto de la fila y concentrar en la diagonal
        row_copy = cm[r_terr].copy()
        cm[r_terr, :] = 0.0
        cm[r_terr, r_terr] = keep
        # añadir un epsilon mínimo al resto para evitar divisiones raras
        eps = 1e-6
        cm[r_terr, (r_terr + 1) % n] = eps
    except ValueError:
        pass

    # 5) Un caso oscuro fuera de la diagonal: B-PROFESION -> B-HOSPITAL con valor ~0
    try:
        r_prof = labels.index("B-PROFESION")
        c_hosp = labels.index("B-HOSPITAL")
        cm[r_prof, c_hosp] = 0.0
    except ValueError:
        pass

    # Asegurar no negatividad
    cm = np.clip(cm, 0.0, None)

    # Normalizar por fila a proporciones [0,1]
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-9
    cm_norm = cm / row_sums
    return cm_norm


def plot_matrix(cm_normalized: np.ndarray, labels: list[str], output_dir: str, filename_prefix: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(25, 18))
    im = ax.imshow(cm_normalized, cmap="viridis", aspect="equal", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(labels, ha="right", fontsize=14)

    titulo = "Matriz de confusión de ejemplo"
    plt.title(titulo, fontsize=18, pad=30, linespacing=1.5)

    # Sin barra de color (colorbar) según solicitud

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{filename_prefix}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Versión limpia sin etiquetas
    fig2, ax2 = plt.subplots(figsize=(25, 18))
    ax2.imshow(cm_normalized, cmap="viridis", aspect="equal", vmin=0, vmax=1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.tight_layout()
    out_path_clean = os.path.join(output_dir, f"{filename_prefix}_no_labels.png")
    plt.savefig(out_path_clean, bbox_inches="tight", dpi=300)
    plt.close(fig2)


def main():
    labels = build_labels_cleaned_q1()
    cm = fabricate_confusion_matrix(labels)

    # Guardar en la misma carpeta de ejemplo que se usa para "cleaned test"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "plots_example"))
    plot_matrix(cm, labels, base_dir, "confusion_matrix_example_simple")

    print("Ejemplo de matriz de confusión generado en:")
    print(os.path.join(base_dir, "confusion_matrix_example_simple.png"))
    print(os.path.join(base_dir, "confusion_matrix_example_simple_no_labels.png"))


if __name__ == "__main__":
    main()


