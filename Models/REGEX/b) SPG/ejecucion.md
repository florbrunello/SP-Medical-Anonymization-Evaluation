Pasos para ejecutar la notebook: 

1. Configura el ambiente

Desde la carpeta REGEX/b) SPG:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install ipykernel jupyter pandas
    python -m ipykernel install --user --name=spg --display-name "Python (SPG)"

Esto crea un entorno virtual, instala las dependencias necesarias y registra el kernel para Jupyter.
Luego, abrir la notebook (.ipynb) en VS Code. Seleccionar el nuevo kernel: en la parte superior derecha de la notebook, hacer clic en el selector de kernel. Buscar y elegir: Python (SPG).

2. Cargar los datos (input)

Reemplazar las celdas existentes por: 
    INPUT_DIR = '/home/usuario/Documentos/TrabajoEspecial/Datasets/SPG/integral'
    TEXT_COLUMN = 'informes'

    # Obtener lista de archivos .txt
    txt_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")])

    # Mostrar los archivos encontrados
    print(txt_files[:5])

    data = []

    for txt_file in txt_files:
        txt_path = os.path.join(INPUT_DIR, txt_file)
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            texto = f.read()
        
        data.append({
            TEXT_COLUMN: texto
        })

    documents = pd.DataFrame(data)

    # Mostrar primeras filas
    print(documents.head(2))

3. Guarda salida 

Reemplazar las celdas existentes por: 
    documents.shape
    output_filename =  '/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/b) SPG/out.csv'
    print('Saving to:', output_filename)
    documents.to_csv(output_filename, sep=',', header=True, index=False)

Finalmente, seleccionar en run all. Se generará un archivo out.csv

4. Procesar el archivo CSV
Ejecutar los scripts: 
    python3 out_post.py
    python3 big.py
    python3 rename.py
    python3 split.py

5. Evaluación del modelo con los datos de SPG:
    cd code
    python3 evaluate.py brat ner "../../../../Datasets/SPG/dev/brat/" "../../../../Modelos/REGEX/b) SPG/brat/dev/"
    python3 evaluate.py brat ner "../../../../Datasets/SPG/test/brat/" "../../../../Modelos/REGEX/b) SPG/brat/test/"
