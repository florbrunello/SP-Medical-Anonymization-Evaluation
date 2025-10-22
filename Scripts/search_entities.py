import os
import glob

# Lista de entidades esperadas
entidades_esperadas = {
    "NOMBRE_SUJETO_ASISTENCIA", "EDAD_SUJETO_ASISTENCIA", "SEXO_SUJETO_ASISTENCIA",
    "FAMILIARES_SUJETO_ASISTENCIA", "NOMBRE_PERSONAL_SANITARIO", "FECHAS", "PROFESION",
    "HOSPITAL", "CENTRO_SALUD", "INSTITUCION", "CALLE", "TERRITORIO", "PAIS",
    "NUMERO_TELEFONO", "NUMERO_FAX", "CORREO_ELECTRONICO", "ID_SUJETO_ASISTENCIA",
    "ID_CONTACTO_ASISTENCIAL", "ID_ASEGURAMIENTO", "ID_TITULACION_PERSONAL_SANITARIO",
    "ID_EMPLEO_PERSONAL_SANITARIO", "IDENTIF_VEHICULOS_NRSERIE_PLACAS",
    "IDENTIF_DISPOSITIVOS_NRSERIE", "DIREC_PROT_INTERNET", "URL_WEB", "IDENTIF_BIOMETRICOS",
    "OTRO_NUMERO_IDENTIF", "OTROS_SUJETO_ASISTENCIA"
}

def encontrar_entidades_en_ann(archivo):
    entidades_encontradas = set()
    with open(archivo, "r", encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("T"):  # líneas de texto anotado
                partes = linea.strip().split()
                if len(partes) > 1:
                    entidad = partes[1]
                    entidades_encontradas.add(entidad)
    return entidades_encontradas

def main(carpeta):
    entidades_presentes = set()
    archivos_ann = glob.glob(os.path.join(carpeta, "*.ann"))

    if not archivos_ann:
        print("No se encontraron archivos .ann en la carpeta.")
        return

    for archivo in archivos_ann:
        entidades = encontrar_entidades_en_ann(archivo)
        entidades_presentes.update(entidades)

    entidades_faltantes = entidades_esperadas - entidades_presentes

    print("Entidades que NO aparecen en ningún archivo .ann:")
    for entidad in sorted(entidades_faltantes):
        print("-", entidad)

# Ruta de la carpeta que contiene los archivos .ann
if __name__ == "__main__":
    carpeta = "/home/usuario/Documentos/TrabajoEspecial/Datasets/MEDDOCAN/dev/brat" 
    main(carpeta)
