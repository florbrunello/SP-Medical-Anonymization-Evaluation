import pickle

# Cargar el archivo pickle
file_path = "/home/usuario/Desktop/TrabajoEspecial/Modelos/2) BiLSTM-CRF/b) MEDDOCAN 1000 casos/code/Extension2/tags.pickle"
with open(file_path, "rb") as file:
    data = pickle.load(file)

# Guardar el contenido en un archivo de texto
txt_file_path = "wordss.txt"
with open(txt_file_path, "w", encoding="utf-8") as txt_file:
    for key, values in data.items():
        txt_file.write(f"ID: {key}\n")
        for entry in values:
            txt_file.write(f"  {entry}\n")
        txt_file.write("\n" + "-"*50 + "\n\n")

print(f"Archivo guardado en: {txt_file_path}")
