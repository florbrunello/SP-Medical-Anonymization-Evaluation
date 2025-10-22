```python
import transformers
from transformers import AutoTokenizer
import torch
import os
print(f"number of GPUs: torch.cuda.device_count()")
print(torch.__version__)
```

    number of GPUs: torch.cuda.device_count()
    2.6.0+rocm6.1



```python
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" 

if  torch.cuda.is_available():
    device = "cuda"
else:
    raise ValueError("No se reconoció GPU.")

pipeline = transformers.pipeline(
	"text-generation", 
	model=model_id,
	model_kwargs={"torch_dtype": torch.bfloat16},
	device=device
)
```


    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


    Device set to use cuda



```python
# Directorios de entrada y salida
input_dir = "txt/"
output_dir = "out/"
os.makedirs(output_dir, exist_ok=True)

# Instrucciones para el modelo
prompt = [
    {"role": "system", 
     "content": 
        """ 
        Quiero que identifiques entidades nombradas que requieren ser anonimizadas en el informe clínico que copio entre comillas al final de esta instrucción. Quiero que me des el resultado en formato .xml in-line, donde las entidades sean identificadas por etiquetas en el mismo texto. Quiero que etiquetes con los criterios MEDDOCAN. A continuación, te muestro un ejemplo que contiene:
        - El texto original del informe en formato plano (.txt)
        - La representación estructurada del mismo en XML con etiquetas semánticas detalladas y posiciones de texto (atributos start, end, text, TYPE, etc.).
        Tu tarea será generar un XML con las mismas reglas de estructura y etiquetado a partir de cada texto clínico. Instrucciones:
        - Conserva el formato exacto del XML del ejemplo.
        - Cada etiqueta tiene que tener el tipo de entidad (`TYPE`) del inventario de MEDDOCAN. Los tipos de entidad que puedes usar son los siguientes: 
        NOMBRE_SUJETO_ASISTENCIA
        EDAD_SUJETO_ASISTENCIA
        SEXO_SUJETO_ASISTENCIA
        FAMILIARES_SUJETO_ASISTENCIA
        NOMBRE_PERSONAL_SANITARIO
        FECHAS
        PROFESION
        HOSPITAL
        CENTRO_SALUD
        INSTITUCION
        CALLE
        TERRITORIO
        PAIS
        NUMERO_TELEFONO
        NUMERO_FAX
        CORREO_ELECTRONICO
        ID_SUJETO_ASISTENCIA
        ID_CONTACTO_ASISTENCIAL
        ID_ASEGURAMIENTO
        ID_TITULACION_PERSONAL_SANITARIO
        ID_EMPLEO_PERSONAL_SANITARIO
        IDENTIF_VEHICULOS_NRSERIE_PLACAS
        IDENTIF_DISPOSITIVOS_NRSERIE
        DIREC_PROT_INTERNET
        URL_WEB
        IDENTIF_BIOMETRICOS
        OTRO_NUMERO_IDENTIF
        OTROS_SUJETO_ASISTENCIA
          - y un campo de comentario (`comment`) vacío
        Cuando te dé un nuevo texto, responde solo con el XML, sin explicaciones adicionales.
        
        Ejemplo - Informe en formato .txt: 
        Datos del paciente.
        Nombre:  Ernesto.
        Apellidos: Rivera Bueno.
        NHC: 368503.
        NASS: 26 63514095.
        Domicilio:  Calle Miguel Benitez 90.
        Localidad/ Provincia: Madrid.
        CP: 28016.
        Datos asistenciales.
        Fecha de nacimiento: 03/03/1946.
        País: España.
        Edad: 70 años Sexo: H.
        Fecha de Ingreso: 12/12/2016.
        Médico:  Ignacio Navarro Cuéllar NºCol: 28 28 70973.
        Informe clínico del paciente: Paciente de 70 años de edad, minero jubilado, sin alergias medicamentosas conocidas, que presenta como antecedentes personales: accidente laboral antiguo con fracturas vertebrales y costales; intervenido de enfermedad de Dupuytren en mano derecha y by-pass iliofemoral izquierdo; Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; enolismo activo, fumador de 20 cigarrillos / día.
        Es derivado desde Atención Primaria por presentar hematuria macroscópica postmiccional en una ocasión y microhematuria persistente posteriormente, con micciones normales.
        En la exploración física presenta un buen estado general, con abdomen y genitales normales; tacto rectal compatible con adenoma de próstata grado I/IV.
        En la analítica de orina destaca la existencia de 4 hematíes/ campo y 0-5 leucocitos/campo; resto de sedimento normal.
        Hemograma normal; en la bioquímica destaca una glucemia de 169 mg/dl y triglicéridos de 456 mg/dl; función hepática y renal normal. PSA de 1.16 ng/ml.
        Las citologías de orina son repetidamente sospechosas de malignidad.
        En la placa simple de abdomen se valoran cambios degenerativos en columna lumbar y calcificaciones vasculares en ambos hipocondrios y en pelvis.
        La ecografía urológica pone de manifiesto la existencia de quistes corticales simples en riñón derecho, vejiga sin alteraciones con buena capacidad y próstata con un peso de 30 g.
        En la UIV se observa normofuncionalismo renal bilateral, calcificaciones sobre silueta renal derecha y uréteres arrosariados con imágenes de adición en el tercio superior de ambos uréteres, en relación a pseudodiverticulosis ureteral. El cistograma demuestra una vejiga con buena capacidad, pero paredes trabeculadas en relación a vejiga de esfuerzo. La TC abdominal es normal.
        La cistoscopia descubre la existencia de pequeñas tumoraciones vesicales, realizándose resección transuretral con el resultado anatomopatológico de carcinoma urotelial superficial de vejiga.
        Remitido por: Ignacio Navarro Cuéllar c/ del Abedul 5-7, 2º dcha 28036 Madrid, España E-mail: nnavcu@hotmail.com.
        
        Ejemplo - Informe en formato .xml: lo que debes generar
        <?xml version='1.0' encoding='UTF-8'?>
        <MEDDOCAN>
          <TEXT>
        Ejemplo - Informe en formato .txt: 
        Datos del paciente.
        Nombre:  <TAG TYPE="NOMBRE_SUJETO_ASISTENCIA">Ernesto</TAG>.
        Apellidos: <TAG TYPE="NOMBRE_SUJETO_ASISTENCIA">Rivera Bueno</TAG>.
        NHC: <TAG TYPE="ID_SUJETO_ASISTENCIA">368503</TAG>.
        NASS: <TAG TYPE="ID_ASEGURAMIENTO">26 63514095</TAG>.
        Domicilio:  <TAG TYPE="CALLE">Calle Miguel Benitez 90</TAG>.
        Localidad/ Provincia: <TAG TYPE="TERRITORIO">Madrid</TAG>.
        CP: <TAG TYPE="TERRITORIO">28016</TAG>.
        Datos asistenciales.
        Fecha de nacimiento: <TAG TYPE="FECHAS">03/03/1946</TAG>.
        País: <TAG TYPE="PAIS">España</TAG>.
        Edad: <TAG TYPE="EDAD_SUJETO_ASISTENCIA">70 años</TAG> Sexo: <TAG TYPE="SEXO_SUJETO_ASISTENCIA">H</TAG>.
        Fecha de Ingreso: <TAG TYPE="FECHAS">12/12/2016</TAG>.
        Médico:  <TAG TYPE="NOMBRE_PERSONAL_SANITARIO">Ignacio</TAG> <TAG TYPE="NOMBRE_PERSONAL_SANITARIO">Navarro Cuéllar</TAG> NºCol: <TAG TYPE="ID_TITULACION_PERSONAL_SANITARIO">28 28 70973</TAG>.
        Informe clínico del paciente: Paciente de <TAG TYPE="EDAD_SUJETO_ASISTENCIA">70 años</TAG> de edad, minero jubilado, sin alergias medicamentosas conocidas, que presenta como antecedentes personales: accidente laboral antiguo con fracturas vertebrales y costales; intervenido de enfermedad de Dupuytren en mano derecha y by-pass iliofemoral izquierdo; Diabetes Mellitus tipo II, hipercolesterolemia e hiperuricemia; enolismo activo, fumador de 20 cigarrillos / día.
        Es derivado desde Atención Primaria por presentar hematuria macroscópica postmiccional en una ocasión y microhematuria persistente posteriormente, con micciones normales.
        En la exploración física presenta un buen estado general, con abdomen y genitales normales; tacto rectal compatible con adenoma de próstata grado I/IV.
        En la analítica de orina destaca la existencia de 4 hematíes/ campo y 0-5 leucocitos/campo; resto de sedimento normal.
        Hemograma normal; en la bioquímica destaca una glucemia de 169 mg/dl y triglicéridos de 456 mg/dl; función hepática y renal normal. PSA de 1.16 ng/ml.
        Las citologías de orina son repetidamente sospechosas de malignidad.
        En la placa simple de abdomen se valoran cambios degenerativos en columna lumbar y calcificaciones vasculares en ambos hipocondrios y en pelvis.
        La ecografía urológica pone de manifiesto la existencia de quistes corticales simples en riñón derecho, vejiga sin alteraciones con buena capacidad y próstata con un peso de 30 g.
        En la UIV se observa normofuncionalismo renal bilateral, calcificaciones sobre silueta renal derecha y uréteres arrosariados con imágenes de adición en el tercio superior de ambos uréteres, en relación a pseudodiverticulosis ureteral. El cistograma demuestra una vejiga con buena capacidad, pero paredes trabeculadas en relación a vejiga de esfuerzo. La TC abdominal es normal.
        La cistoscopia descubre la existencia de pequeñas tumoraciones vesicales, realizándose resección transuretral con el resultado anatomopatológico de carcinoma urotelial superficial de vejiga.
        Remitido por: <TAG TYPE="NOMBRE_PERSONAL_SANITARIO">Ignacio</TAG> <TAG TYPE="NOMBRE_PERSONAL_SANITARIO">Navarro Cuéllar</TAG> <TAG TYPE="CALLE">c/ del Abedul 5-7, 2º dcha</TAG> <TAG TYPE="TERRITORIO" >28036</TAG> <TAG TYPE="TERRITORIO" >Madrid</TAG>, <TAG TYPE="PAIS">España</TAG> E-mail: <TAG TYPE="CORREO_ELECTRONICO">nnavcu@hotmail.com</TAG>.
        </TEXT>
        </MEDDOCAN>
        
        Recordá que en ningún caso debes incluir advertencias, explicaciones ni descripciones sobre la tarea, sobre la instrucción que te he dado o sobre cuestiones de funcionamiento del modelo de lenguaje.
        """},
     ]

# Configuración de tokens
MAX_CONTEXT_TOKENS = 8192
MAX_GENERATION_TOKENS = 4000
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - MAX_GENERATION_TOKENS

# Procesar cada archivo .txt
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            texto = f.read()

        # Crear mensaje estilo chat
        prompt_text = prompt[0]["content"]
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": texto}
        ]

        # Calcular tokens de entrada
        full_prompt = prompt_text + texto
        total_tokens = len(tokenizer.encode(full_prompt))
        print(f"{filename}: Tokens de entrada: {total_tokens}")

        # Truncar el prompt si se pasa del límite permitido
        if total_tokens > MAX_INPUT_TOKENS:
            print(f"Truncando prompt: {filename}")
            # Calcular los tokens disponibles para el prompt
            max_tokens_prompt = MAX_INPUT_TOKENS - len(tokenizer.encode(texto))
            
            # Truncar el prompt para ajustarlo al límite de tokens
            prompt_tokens = tokenizer.encode(prompt[0]["content"])
            truncated_prompt_tokens = prompt_tokens[:max_tokens_prompt]
            
            # Decodificar los tokens truncados y actualizar el prompt
            truncated_prompt = tokenizer.decode(truncated_prompt_tokens, skip_special_tokens=True)
            messages[0]["content"] = truncated_prompt

        # Generar texto
        output = pipeline(messages, max_new_tokens=MAX_GENERATION_TOKENS)

        # Extraer solo el contenido generado por el modelo
        respuesta = output[0]["generated_text"][2]["content"]

        # Guardar en .xml
        output_filename = os.path.splitext(filename)[0] + ".xml"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(respuesta)

        print(f"Procesado: {filename} → {output_filename}")

print("Proceso completado.")
```

    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    390535044.txt: Tokens de entrada: 3410


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 390535044.txt → 390535044.xml
    990720341.txt: Tokens de entrada: 3324


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 990720341.txt → 990720341.xml
    399041647.txt: Tokens de entrada: 3152


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 399041647.txt → 399041647.xml
    774011819.txt: Tokens de entrada: 2871


    You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 774011819.txt → 774011819.xml
    706134751.txt: Tokens de entrada: 3203


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 706134751.txt → 706134751.xml
    013859482.txt: Tokens de entrada: 2852


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 013859482.txt → 013859482.xml
    045870691.txt: Tokens de entrada: 2751


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 045870691.txt → 045870691.xml
    849796182.txt: Tokens de entrada: 2857


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 849796182.txt → 849796182.xml
    946340084.txt: Tokens de entrada: 3344


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 946340084.txt → 946340084.xml
    718817047.txt: Tokens de entrada: 3187


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 718817047.txt → 718817047.xml
    364282199.txt: Tokens de entrada: 3434


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 364282199.txt → 364282199.xml
    714896030.txt: Tokens de entrada: 3429


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 714896030.txt → 714896030.xml
    691741259.txt: Tokens de entrada: 2852


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 691741259.txt → 691741259.xml
    869873706.txt: Tokens de entrada: 3171


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 869873706.txt → 869873706.xml
    306009108.txt: Tokens de entrada: 3290


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 306009108.txt → 306009108.xml
    993479328.txt: Tokens de entrada: 3215


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 993479328.txt → 993479328.xml
    511664645.txt: Tokens de entrada: 3016


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 511664645.txt → 511664645.xml
    521505023.txt: Tokens de entrada: 3078


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 521505023.txt → 521505023.xml
    276549386.txt: Tokens de entrada: 3115


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 276549386.txt → 276549386.xml
    781838481.txt: Tokens de entrada: 3114


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 781838481.txt → 781838481.xml
    048699967.txt: Tokens de entrada: 3430


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 048699967.txt → 048699967.xml
    072062644.txt: Tokens de entrada: 3165


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 072062644.txt → 072062644.xml
    780612522.txt: Tokens de entrada: 3031


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 780612522.txt → 780612522.xml
    649782247.txt: Tokens de entrada: 3378


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 649782247.txt → 649782247.xml
    381610271.txt: Tokens de entrada: 3338


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 381610271.txt → 381610271.xml
    350016843.txt: Tokens de entrada: 2951


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 350016843.txt → 350016843.xml
    887826232.txt: Tokens de entrada: 3404


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 887826232.txt → 887826232.xml
    749550983.txt: Tokens de entrada: 2930


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 749550983.txt → 749550983.xml
    277906833.txt: Tokens de entrada: 3146


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 277906833.txt → 277906833.xml
    515061887.txt: Tokens de entrada: 3122


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 515061887.txt → 515061887.xml
    041397855.txt: Tokens de entrada: 3406


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 041397855.txt → 041397855.xml
    143984435.txt: Tokens de entrada: 3356


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 143984435.txt → 143984435.xml
    601035657.txt: Tokens de entrada: 3301


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 601035657.txt → 601035657.xml
    614776161.txt: Tokens de entrada: 3105


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 614776161.txt → 614776161.xml
    362766576.txt: Tokens de entrada: 3263


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 362766576.txt → 362766576.xml
    002769099.txt: Tokens de entrada: 3023


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 002769099.txt → 002769099.xml
    652947676.txt: Tokens de entrada: 3213


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 652947676.txt → 652947676.xml
    506050914.txt: Tokens de entrada: 3025


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 506050914.txt → 506050914.xml
    547282685.txt: Tokens de entrada: 3121


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 547282685.txt → 547282685.xml
    078954525.txt: Tokens de entrada: 3290


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 078954525.txt → 078954525.xml
    606063227.txt: Tokens de entrada: 3313


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 606063227.txt → 606063227.xml
    513633424.txt: Tokens de entrada: 3195


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 513633424.txt → 513633424.xml
    012135403.txt: Tokens de entrada: 3272


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 012135403.txt → 012135403.xml
    880424816.txt: Tokens de entrada: 3185


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 880424816.txt → 880424816.xml
    123376200.txt: Tokens de entrada: 3280


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 123376200.txt → 123376200.xml
    993577188.txt: Tokens de entrada: 3272


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 993577188.txt → 993577188.xml
    688634239.txt: Tokens de entrada: 3186


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 688634239.txt → 688634239.xml
    513665572.txt: Tokens de entrada: 3167


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 513665572.txt → 513665572.xml
    932528658.txt: Tokens de entrada: 3017


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 932528658.txt → 932528658.xml
    501581893.txt: Tokens de entrada: 2739


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 501581893.txt → 501581893.xml
    575005901.txt: Tokens de entrada: 3303


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 575005901.txt → 575005901.xml
    916606884.txt: Tokens de entrada: 3268


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 916606884.txt → 916606884.xml
    451450476.txt: Tokens de entrada: 3261


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 451450476.txt → 451450476.xml
    342363855.txt: Tokens de entrada: 3137


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 342363855.txt → 342363855.xml
    430059893.txt: Tokens de entrada: 3207


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 430059893.txt → 430059893.xml
    443502292.txt: Tokens de entrada: 3421


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 443502292.txt → 443502292.xml
    296551681.txt: Tokens de entrada: 3072


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 296551681.txt → 296551681.xml
    123843563.txt: Tokens de entrada: 2876


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 123843563.txt → 123843563.xml
    497801942.txt: Tokens de entrada: 3428


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 497801942.txt → 497801942.xml
    498186154.txt: Tokens de entrada: 3398


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 498186154.txt → 498186154.xml
    144874920.txt: Tokens de entrada: 3125


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 144874920.txt → 144874920.xml
    082512456.txt: Tokens de entrada: 3136


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 082512456.txt → 082512456.xml
    730086383.txt: Tokens de entrada: 3234


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 730086383.txt → 730086383.xml
    022360168.txt: Tokens de entrada: 3021


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 022360168.txt → 022360168.xml
    658320128.txt: Tokens de entrada: 3234


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 658320128.txt → 658320128.xml
    946927743.txt: Tokens de entrada: 3228


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 946927743.txt → 946927743.xml
    203147612.txt: Tokens de entrada: 3210


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 203147612.txt → 203147612.xml
    437083459.txt: Tokens de entrada: 3040


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 437083459.txt → 437083459.xml
    773071198.txt: Tokens de entrada: 3181


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 773071198.txt → 773071198.xml
    948267996.txt: Tokens de entrada: 3242


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 948267996.txt → 948267996.xml
    241960919.txt: Tokens de entrada: 3163


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 241960919.txt → 241960919.xml
    696321479.txt: Tokens de entrada: 3392


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 696321479.txt → 696321479.xml
    773787806.txt: Tokens de entrada: 3036


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 773787806.txt → 773787806.xml
    274655567.txt: Tokens de entrada: 2908


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 274655567.txt → 274655567.xml
    678788106.txt: Tokens de entrada: 2922


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 678788106.txt → 678788106.xml
    905848513.txt: Tokens de entrada: 3325


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 905848513.txt → 905848513.xml
    955704690.txt: Tokens de entrada: 3217


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 955704690.txt → 955704690.xml
    598148722.txt: Tokens de entrada: 3144


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 598148722.txt → 598148722.xml
    616361252.txt: Tokens de entrada: 3341


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 616361252.txt → 616361252.xml
    122804265.txt: Tokens de entrada: 3296


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 122804265.txt → 122804265.xml
    677503499.txt: Tokens de entrada: 3206


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 677503499.txt → 677503499.xml
    183416766.txt: Tokens de entrada: 3298


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 183416766.txt → 183416766.xml
    016764547.txt: Tokens de entrada: 3008


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 016764547.txt → 016764547.xml
    302429258.txt: Tokens de entrada: 3007


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 302429258.txt → 302429258.xml
    920707969.txt: Tokens de entrada: 3294


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 920707969.txt → 920707969.xml
    088813511.txt: Tokens de entrada: 3010


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 088813511.txt → 088813511.xml
    136615040.txt: Tokens de entrada: 2799


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 136615040.txt → 136615040.xml
    705042332.txt: Tokens de entrada: 3277


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 705042332.txt → 705042332.xml
    906070281.txt: Tokens de entrada: 2845


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 906070281.txt → 906070281.xml
    301477361.txt: Tokens de entrada: 3098


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 301477361.txt → 301477361.xml
    540108868.txt: Tokens de entrada: 3344


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 540108868.txt → 540108868.xml
    192982381.txt: Tokens de entrada: 3247


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 192982381.txt → 192982381.xml
    787493182.txt: Tokens de entrada: 3159


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 787493182.txt → 787493182.xml
    709920014.txt: Tokens de entrada: 3315


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 709920014.txt → 709920014.xml
    358938538.txt: Tokens de entrada: 3113


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 358938538.txt → 358938538.xml
    301474882.txt: Tokens de entrada: 3284


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 301474882.txt → 301474882.xml
    782959227.txt: Tokens de entrada: 3189


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 782959227.txt → 782959227.xml
    975183453.txt: Tokens de entrada: 3218


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 975183453.txt → 975183453.xml
    690073045.txt: Tokens de entrada: 3241


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 690073045.txt → 690073045.xml
    811599528.txt: Tokens de entrada: 3050


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 811599528.txt → 811599528.xml
    195936517.txt: Tokens de entrada: 2734


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 195936517.txt → 195936517.xml
    877744736.txt: Tokens de entrada: 3027


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 877744736.txt → 877744736.xml
    391785393.txt: Tokens de entrada: 3095


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 391785393.txt → 391785393.xml
    900051902.txt: Tokens de entrada: 2853


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 900051902.txt → 900051902.xml
    162397533.txt: Tokens de entrada: 3260


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 162397533.txt → 162397533.xml
    445543903.txt: Tokens de entrada: 3177


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 445543903.txt → 445543903.xml
    292849640.txt: Tokens de entrada: 3211


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 292849640.txt → 292849640.xml
    810195405.txt: Tokens de entrada: 3171


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 810195405.txt → 810195405.xml
    404964621.txt: Tokens de entrada: 3173


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 404964621.txt → 404964621.xml
    882940617.txt: Tokens de entrada: 3004


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 882940617.txt → 882940617.xml
    654170593.txt: Tokens de entrada: 2747


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 654170593.txt → 654170593.xml
    632267394.txt: Tokens de entrada: 3057


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 632267394.txt → 632267394.xml
    359968301.txt: Tokens de entrada: 2764


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 359968301.txt → 359968301.xml
    053678038.txt: Tokens de entrada: 3078


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 053678038.txt → 053678038.xml
    527978714.txt: Tokens de entrada: 3278


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 527978714.txt → 527978714.xml
    624457565.txt: Tokens de entrada: 2972


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 624457565.txt → 624457565.xml
    841824272.txt: Tokens de entrada: 2925


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 841824272.txt → 841824272.xml
    505661421.txt: Tokens de entrada: 2801


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 505661421.txt → 505661421.xml
    503605741.txt: Tokens de entrada: 3400


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 503605741.txt → 503605741.xml
    392966434.txt: Tokens de entrada: 2878


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 392966434.txt → 392966434.xml
    542501574.txt: Tokens de entrada: 3303


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 542501574.txt → 542501574.xml
    116600408.txt: Tokens de entrada: 3311


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 116600408.txt → 116600408.xml
    584744406.txt: Tokens de entrada: 3409


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 584744406.txt → 584744406.xml
    702527775.txt: Tokens de entrada: 3261


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 702527775.txt → 702527775.xml
    597185762.txt: Tokens de entrada: 3133


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 597185762.txt → 597185762.xml
    790602246.txt: Tokens de entrada: 3186


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 790602246.txt → 790602246.xml
    349992773.txt: Tokens de entrada: 3072


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 349992773.txt → 349992773.xml
    205204883.txt: Tokens de entrada: 2980


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 205204883.txt → 205204883.xml
    974586397.txt: Tokens de entrada: 2748


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 974586397.txt → 974586397.xml
    853658541.txt: Tokens de entrada: 3090


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 853658541.txt → 853658541.xml
    320458767.txt: Tokens de entrada: 2992


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 320458767.txt → 320458767.xml
    064922103.txt: Tokens de entrada: 2773


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 064922103.txt → 064922103.xml
    335648982.txt: Tokens de entrada: 3395


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 335648982.txt → 335648982.xml
    522115012.txt: Tokens de entrada: 3055


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 522115012.txt → 522115012.xml
    306437166.txt: Tokens de entrada: 3071


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 306437166.txt → 306437166.xml
    162203368.txt: Tokens de entrada: 3320


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 162203368.txt → 162203368.xml
    075747430.txt: Tokens de entrada: 3168


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 075747430.txt → 075747430.xml
    580244218.txt: Tokens de entrada: 3256


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 580244218.txt → 580244218.xml
    862010312.txt: Tokens de entrada: 3365


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 862010312.txt → 862010312.xml
    269537260.txt: Tokens de entrada: 3140


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 269537260.txt → 269537260.xml
    392836819.txt: Tokens de entrada: 3085


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 392836819.txt → 392836819.xml
    889070739.txt: Tokens de entrada: 2909


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 889070739.txt → 889070739.xml
    373806265.txt: Tokens de entrada: 3298


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 373806265.txt → 373806265.xml
    291654267.txt: Tokens de entrada: 3251


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 291654267.txt → 291654267.xml
    718221461.txt: Tokens de entrada: 3312


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 718221461.txt → 718221461.xml
    260753908.txt: Tokens de entrada: 3418


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 260753908.txt → 260753908.xml
    020364284.txt: Tokens de entrada: 2721


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 020364284.txt → 020364284.xml
    390653038.txt: Tokens de entrada: 3260


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 390653038.txt → 390653038.xml
    260913575.txt: Tokens de entrada: 3133


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 260913575.txt → 260913575.xml
    728556392.txt: Tokens de entrada: 2981


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 728556392.txt → 728556392.xml
    765766190.txt: Tokens de entrada: 3294


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 765766190.txt → 765766190.xml
    853121870.txt: Tokens de entrada: 3043


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 853121870.txt → 853121870.xml
    531776211.txt: Tokens de entrada: 3238
    Procesado: 531776211.txt → 531776211.xml


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    794026394.txt: Tokens de entrada: 3180


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 794026394.txt → 794026394.xml
    080410949.txt: Tokens de entrada: 2973


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 080410949.txt → 080410949.xml
    573276782.txt: Tokens de entrada: 3228


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 573276782.txt → 573276782.xml
    641355006.txt: Tokens de entrada: 3067


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 641355006.txt → 641355006.xml
    577234398.txt: Tokens de entrada: 3228


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 577234398.txt → 577234398.xml
    643938461.txt: Tokens de entrada: 3186


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 643938461.txt → 643938461.xml
    572730498.txt: Tokens de entrada: 3299


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 572730498.txt → 572730498.xml
    911408584.txt: Tokens de entrada: 3199


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 911408584.txt → 911408584.xml
    842115767.txt: Tokens de entrada: 3213


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 842115767.txt → 842115767.xml
    929183134.txt: Tokens de entrada: 2760


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 929183134.txt → 929183134.xml
    841606847.txt: Tokens de entrada: 3048


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 841606847.txt → 841606847.xml
    901626164.txt: Tokens de entrada: 3393


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 901626164.txt → 901626164.xml
    552845925.txt: Tokens de entrada: 3262


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 552845925.txt → 552845925.xml
    234339405.txt: Tokens de entrada: 3191


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 234339405.txt → 234339405.xml
    912232568.txt: Tokens de entrada: 3276


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 912232568.txt → 912232568.xml
    441397520.txt: Tokens de entrada: 3154


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 441397520.txt → 441397520.xml
    471629098.txt: Tokens de entrada: 3207


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 471629098.txt → 471629098.xml
    831357094.txt: Tokens de entrada: 3197


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 831357094.txt → 831357094.xml
    916946631.txt: Tokens de entrada: 3134


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 916946631.txt → 916946631.xml
    642505224.txt: Tokens de entrada: 2896


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 642505224.txt → 642505224.xml
    949223650.txt: Tokens de entrada: 3217


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 949223650.txt → 949223650.xml
    477983544.txt: Tokens de entrada: 3150


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 477983544.txt → 477983544.xml
    016215942.txt: Tokens de entrada: 3130


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 016215942.txt → 016215942.xml
    884437438.txt: Tokens de entrada: 3145


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 884437438.txt → 884437438.xml
    994427603.txt: Tokens de entrada: 3005


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 994427603.txt → 994427603.xml
    235723921.txt: Tokens de entrada: 3158


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 235723921.txt → 235723921.xml
    410163316.txt: Tokens de entrada: 3273


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 410163316.txt → 410163316.xml
    700803973.txt: Tokens de entrada: 3114


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 700803973.txt → 700803973.xml
    645257288.txt: Tokens de entrada: 3096


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 645257288.txt → 645257288.xml
    398947975.txt: Tokens de entrada: 3224


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 398947975.txt → 398947975.xml
    723982399.txt: Tokens de entrada: 2772


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 723982399.txt → 723982399.xml
    308475716.txt: Tokens de entrada: 3174


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 308475716.txt → 308475716.xml
    474430236.txt: Tokens de entrada: 3308


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 474430236.txt → 474430236.xml
    851639758.txt: Tokens de entrada: 2752


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 851639758.txt → 851639758.xml
    207286150.txt: Tokens de entrada: 2859


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 207286150.txt → 207286150.xml
    752419591.txt: Tokens de entrada: 3336


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 752419591.txt → 752419591.xml
    135072705.txt: Tokens de entrada: 3233


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 135072705.txt → 135072705.xml
    453095041.txt: Tokens de entrada: 3433


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 453095041.txt → 453095041.xml
    401708223.txt: Tokens de entrada: 3049


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 401708223.txt → 401708223.xml
    338146489.txt: Tokens de entrada: 3300


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 338146489.txt → 338146489.xml
    983746151.txt: Tokens de entrada: 3127


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 983746151.txt → 983746151.xml
    081672522.txt: Tokens de entrada: 3340


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 081672522.txt → 081672522.xml
    013047559.txt: Tokens de entrada: 3162


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 013047559.txt → 013047559.xml
    682979808.txt: Tokens de entrada: 3133


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 682979808.txt → 682979808.xml
    841382782.txt: Tokens de entrada: 3202


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 841382782.txt → 841382782.xml
    836507849.txt: Tokens de entrada: 3031


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 836507849.txt → 836507849.xml
    579940350.txt: Tokens de entrada: 3316


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 579940350.txt → 579940350.xml
    473819687.txt: Tokens de entrada: 3286


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 473819687.txt → 473819687.xml
    005766636.txt: Tokens de entrada: 2841


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 005766636.txt → 005766636.xml
    781424959.txt: Tokens de entrada: 3185


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 781424959.txt → 781424959.xml
    407044863.txt: Tokens de entrada: 3340


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 407044863.txt → 407044863.xml
    307016621.txt: Tokens de entrada: 3185


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 307016621.txt → 307016621.xml
    157910970.txt: Tokens de entrada: 3011


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 157910970.txt → 157910970.xml
    047851383.txt: Tokens de entrada: 3238


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 047851383.txt → 047851383.xml
    055796772.txt: Tokens de entrada: 3055


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 055796772.txt → 055796772.xml
    640702967.txt: Tokens de entrada: 3334


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 640702967.txt → 640702967.xml
    025375063.txt: Tokens de entrada: 3265


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 025375063.txt → 025375063.xml
    191957996.txt: Tokens de entrada: 2768


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 191957996.txt → 191957996.xml
    080935804.txt: Tokens de entrada: 3419


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 080935804.txt → 080935804.xml
    470259492.txt: Tokens de entrada: 2692


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 470259492.txt → 470259492.xml
    638583754.txt: Tokens de entrada: 3274


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 638583754.txt → 638583754.xml
    594060626.txt: Tokens de entrada: 3427


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 594060626.txt → 594060626.xml
    886849671.txt: Tokens de entrada: 2774


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 886849671.txt → 886849671.xml
    439541641.txt: Tokens de entrada: 2925


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 439541641.txt → 439541641.xml
    049582322.txt: Tokens de entrada: 2956


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 049582322.txt → 049582322.xml
    179069033.txt: Tokens de entrada: 2993


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 179069033.txt → 179069033.xml
    667643216.txt: Tokens de entrada: 3164


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 667643216.txt → 667643216.xml
    934369810.txt: Tokens de entrada: 3434


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 934369810.txt → 934369810.xml
    555112093.txt: Tokens de entrada: 3187


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 555112093.txt → 555112093.xml
    839007788.txt: Tokens de entrada: 2698


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 839007788.txt → 839007788.xml
    161193746.txt: Tokens de entrada: 3272


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 161193746.txt → 161193746.xml
    122970521.txt: Tokens de entrada: 3268


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 122970521.txt → 122970521.xml
    843054476.txt: Tokens de entrada: 3265


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 843054476.txt → 843054476.xml
    761218358.txt: Tokens de entrada: 3193


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 761218358.txt → 761218358.xml
    799030721.txt: Tokens de entrada: 3303


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 799030721.txt → 799030721.xml
    647065314.txt: Tokens de entrada: 3112


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 647065314.txt → 647065314.xml
    279380446.txt: Tokens de entrada: 3292


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 279380446.txt → 279380446.xml
    083468827.txt: Tokens de entrada: 3241


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 083468827.txt → 083468827.xml
    929817590.txt: Tokens de entrada: 3120


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 929817590.txt → 929817590.xml
    361079168.txt: Tokens de entrada: 3170


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 361079168.txt → 361079168.xml
    612890741.txt: Tokens de entrada: 3131


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 612890741.txt → 612890741.xml
    702978449.txt: Tokens de entrada: 3136


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 702978449.txt → 702978449.xml
    444300106.txt: Tokens de entrada: 3103


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 444300106.txt → 444300106.xml
    662748955.txt: Tokens de entrada: 3369


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 662748955.txt → 662748955.xml
    189308914.txt: Tokens de entrada: 3411


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 189308914.txt → 189308914.xml
    614932894.txt: Tokens de entrada: 3284


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 614932894.txt → 614932894.xml
    299988887.txt: Tokens de entrada: 3335


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 299988887.txt → 299988887.xml
    383596030.txt: Tokens de entrada: 3304


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 383596030.txt → 383596030.xml
    133046492.txt: Tokens de entrada: 3342


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 133046492.txt → 133046492.xml
    583303247.txt: Tokens de entrada: 3183


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 583303247.txt → 583303247.xml
    540437733.txt: Tokens de entrada: 3013


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 540437733.txt → 540437733.xml
    790954525.txt: Tokens de entrada: 3275


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 790954525.txt → 790954525.xml
    904999396.txt: Tokens de entrada: 3310


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 904999396.txt → 904999396.xml
    074221588.txt: Tokens de entrada: 2695


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 074221588.txt → 074221588.xml
    934901961.txt: Tokens de entrada: 3311


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 934901961.txt → 934901961.xml
    453983962.txt: Tokens de entrada: 3233


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 453983962.txt → 453983962.xml
    153970381.txt: Tokens de entrada: 3176


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 153970381.txt → 153970381.xml
    442377820.txt: Tokens de entrada: 3332


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 442377820.txt → 442377820.xml
    529088693.txt: Tokens de entrada: 3108


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 529088693.txt → 529088693.xml
    988951928.txt: Tokens de entrada: 3059


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 988951928.txt → 988951928.xml
    062333885.txt: Tokens de entrada: 3016


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 062333885.txt → 062333885.xml
    832082119.txt: Tokens de entrada: 3154


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 832082119.txt → 832082119.xml
    705162272.txt: Tokens de entrada: 3028


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 705162272.txt → 705162272.xml
    574615527.txt: Tokens de entrada: 2754


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 574615527.txt → 574615527.xml
    915623376.txt: Tokens de entrada: 3073


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 915623376.txt → 915623376.xml
    327065203.txt: Tokens de entrada: 3193


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 327065203.txt → 327065203.xml
    452906943.txt: Tokens de entrada: 3314


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 452906943.txt → 452906943.xml
    004954246.txt: Tokens de entrada: 2737


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 004954246.txt → 004954246.xml
    854340264.txt: Tokens de entrada: 3374


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 854340264.txt → 854340264.xml
    299966440.txt: Tokens de entrada: 3370


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 299966440.txt → 299966440.xml
    185331863.txt: Tokens de entrada: 3091


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 185331863.txt → 185331863.xml
    982570255.txt: Tokens de entrada: 3288


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 982570255.txt → 982570255.xml
    136867595.txt: Tokens de entrada: 2765


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 136867595.txt → 136867595.xml
    217429862.txt: Tokens de entrada: 3273


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 217429862.txt → 217429862.xml
    783699979.txt: Tokens de entrada: 3224


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 783699979.txt → 783699979.xml
    273553181.txt: Tokens de entrada: 3330


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 273553181.txt → 273553181.xml
    839579642.txt: Tokens de entrada: 2822


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 839579642.txt → 839579642.xml
    878300141.txt: Tokens de entrada: 2896


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 878300141.txt → 878300141.xml
    557096587.txt: Tokens de entrada: 3105


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 557096587.txt → 557096587.xml
    418828393.txt: Tokens de entrada: 3155


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 418828393.txt → 418828393.xml
    702273075.txt: Tokens de entrada: 3287


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 702273075.txt → 702273075.xml
    751362924.txt: Tokens de entrada: 3026


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 751362924.txt → 751362924.xml
    167310680.txt: Tokens de entrada: 3110


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 167310680.txt → 167310680.xml
    720753386.txt: Tokens de entrada: 3249


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 720753386.txt → 720753386.xml
    548395570.txt: Tokens de entrada: 3070


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 548395570.txt → 548395570.xml
    341183872.txt: Tokens de entrada: 3209


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 341183872.txt → 341183872.xml
    990575060.txt: Tokens de entrada: 3115


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 990575060.txt → 990575060.xml
    253036833.txt: Tokens de entrada: 3223


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 253036833.txt → 253036833.xml
    697572931.txt: Tokens de entrada: 3340


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 697572931.txt → 697572931.xml
    730122506.txt: Tokens de entrada: 3073


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 730122506.txt → 730122506.xml
    233248237.txt: Tokens de entrada: 3085


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 233248237.txt → 233248237.xml
    217293634.txt: Tokens de entrada: 3119


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 217293634.txt → 217293634.xml
    029198908.txt: Tokens de entrada: 3141


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 029198908.txt → 029198908.xml
    823840648.txt: Tokens de entrada: 3418


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 823840648.txt → 823840648.xml
    131026303.txt: Tokens de entrada: 3154


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 131026303.txt → 131026303.xml
    293469133.txt: Tokens de entrada: 3255


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 293469133.txt → 293469133.xml
    413614619.txt: Tokens de entrada: 3399


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 413614619.txt → 413614619.xml
    614741885.txt: Tokens de entrada: 2765


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 614741885.txt → 614741885.xml
    611045770.txt: Tokens de entrada: 3307


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 611045770.txt → 611045770.xml
    941878909.txt: Tokens de entrada: 3127


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 941878909.txt → 941878909.xml
    860703747.txt: Tokens de entrada: 3135


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 860703747.txt → 860703747.xml
    674830774.txt: Tokens de entrada: 3130


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 674830774.txt → 674830774.xml
    900212897.txt: Tokens de entrada: 3266


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 900212897.txt → 900212897.xml
    580199505.txt: Tokens de entrada: 3146


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 580199505.txt → 580199505.xml
    365565580.txt: Tokens de entrada: 3124


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 365565580.txt → 365565580.xml
    502642392.txt: Tokens de entrada: 3265


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 502642392.txt → 502642392.xml
    405635190.txt: Tokens de entrada: 3421


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 405635190.txt → 405635190.xml
    463806475.txt: Tokens de entrada: 3387


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 463806475.txt → 463806475.xml
    164247300.txt: Tokens de entrada: 3287


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 164247300.txt → 164247300.xml
    858831674.txt: Tokens de entrada: 3241


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 858831674.txt → 858831674.xml
    081813008.txt: Tokens de entrada: 3202


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 081813008.txt → 081813008.xml
    901550052.txt: Tokens de entrada: 3395


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 901550052.txt → 901550052.xml
    003666606.txt: Tokens de entrada: 3321


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 003666606.txt → 003666606.xml
    427774650.txt: Tokens de entrada: 2996


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 427774650.txt → 427774650.xml
    495194539.txt: Tokens de entrada: 3283


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 495194539.txt → 495194539.xml
    639408303.txt: Tokens de entrada: 3107


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 639408303.txt → 639408303.xml
    110423130.txt: Tokens de entrada: 3165


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 110423130.txt → 110423130.xml
    666697529.txt: Tokens de entrada: 3343


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 666697529.txt → 666697529.xml
    925033386.txt: Tokens de entrada: 3433


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 925033386.txt → 925033386.xml
    645553524.txt: Tokens de entrada: 3262


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 645553524.txt → 645553524.xml
    918485962.txt: Tokens de entrada: 2741


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 918485962.txt → 918485962.xml
    860817379.txt: Tokens de entrada: 3230


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 860817379.txt → 860817379.xml
    727575879.txt: Tokens de entrada: 3250


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 727575879.txt → 727575879.xml
    033363089.txt: Tokens de entrada: 3347


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 033363089.txt → 033363089.xml
    090527326.txt: Tokens de entrada: 3433


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 090527326.txt → 090527326.xml
    991670909.txt: Tokens de entrada: 3139


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 991670909.txt → 991670909.xml
    024574001.txt: Tokens de entrada: 3354


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 024574001.txt → 024574001.xml
    594983058.txt: Tokens de entrada: 3053


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 594983058.txt → 594983058.xml
    986986756.txt: Tokens de entrada: 3400


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 986986756.txt → 986986756.xml
    369566555.txt: Tokens de entrada: 2807


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 369566555.txt → 369566555.xml
    419923963.txt: Tokens de entrada: 2732


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 419923963.txt → 419923963.xml
    277789251.txt: Tokens de entrada: 3243


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 277789251.txt → 277789251.xml
    175955287.txt: Tokens de entrada: 3200


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 175955287.txt → 175955287.xml
    330530364.txt: Tokens de entrada: 3269


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 330530364.txt → 330530364.xml
    826007578.txt: Tokens de entrada: 2916


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 826007578.txt → 826007578.xml
    484529658.txt: Tokens de entrada: 3329


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 484529658.txt → 484529658.xml
    259670817.txt: Tokens de entrada: 3179


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 259670817.txt → 259670817.xml
    732380020.txt: Tokens de entrada: 3417


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 732380020.txt → 732380020.xml
    411597055.txt: Tokens de entrada: 3210


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 411597055.txt → 411597055.xml
    933048349.txt: Tokens de entrada: 3012


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 933048349.txt → 933048349.xml
    194407003.txt: Tokens de entrada: 3203


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 194407003.txt → 194407003.xml
    798290653.txt: Tokens de entrada: 3325


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 798290653.txt → 798290653.xml
    958411362.txt: Tokens de entrada: 3173


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 958411362.txt → 958411362.xml
    901492517.txt: Tokens de entrada: 3434


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 901492517.txt → 901492517.xml
    124641479.txt: Tokens de entrada: 3197


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 124641479.txt → 124641479.xml
    379570765.txt: Tokens de entrada: 3408


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 379570765.txt → 379570765.xml
    768892512.txt: Tokens de entrada: 3398


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 768892512.txt → 768892512.xml
    799874626.txt: Tokens de entrada: 3075


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 799874626.txt → 799874626.xml
    927909325.txt: Tokens de entrada: 3144


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 927909325.txt → 927909325.xml
    018733974.txt: Tokens de entrada: 3144


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 018733974.txt → 018733974.xml
    828999873.txt: Tokens de entrada: 3065


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 828999873.txt → 828999873.xml
    164720184.txt: Tokens de entrada: 3275


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 164720184.txt → 164720184.xml
    936813726.txt: Tokens de entrada: 3099


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 936813726.txt → 936813726.xml
    118202067.txt: Tokens de entrada: 3347


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 118202067.txt → 118202067.xml
    736336430.txt: Tokens de entrada: 3246


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 736336430.txt → 736336430.xml
    907210617.txt: Tokens de entrada: 3113


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 907210617.txt → 907210617.xml
    318417173.txt: Tokens de entrada: 3282


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 318417173.txt → 318417173.xml
    412086658.txt: Tokens de entrada: 2697


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 412086658.txt → 412086658.xml
    975828051.txt: Tokens de entrada: 3034


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 975828051.txt → 975828051.xml
    223441614.txt: Tokens de entrada: 3361


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 223441614.txt → 223441614.xml
    515993781.txt: Tokens de entrada: 3095


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 515993781.txt → 515993781.xml
    094559819.txt: Tokens de entrada: 3290
    Procesado: 094559819.txt → 094559819.xml


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    820442023.txt: Tokens de entrada: 3313


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 820442023.txt → 820442023.xml
    364448059.txt: Tokens de entrada: 3073


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 364448059.txt → 364448059.xml
    241159425.txt: Tokens de entrada: 3065


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 241159425.txt → 241159425.xml
    714352071.txt: Tokens de entrada: 2980


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 714352071.txt → 714352071.xml
    796846863.txt: Tokens de entrada: 3173


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 796846863.txt → 796846863.xml
    781034562.txt: Tokens de entrada: 2974


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 781034562.txt → 781034562.xml
    637525813.txt: Tokens de entrada: 3356


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 637525813.txt → 637525813.xml
    453908120.txt: Tokens de entrada: 3197


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 453908120.txt → 453908120.xml
    754993080.txt: Tokens de entrada: 3196


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 754993080.txt → 754993080.xml
    249444316.txt: Tokens de entrada: 3143


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 249444316.txt → 249444316.xml
    134203308.txt: Tokens de entrada: 3201


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 134203308.txt → 134203308.xml
    363672501.txt: Tokens de entrada: 3229


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 363672501.txt → 363672501.xml
    201991253.txt: Tokens de entrada: 3220


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 201991253.txt → 201991253.xml
    924985962.txt: Tokens de entrada: 3397


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 924985962.txt → 924985962.xml
    557927115.txt: Tokens de entrada: 3231


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 557927115.txt → 557927115.xml
    431321676.txt: Tokens de entrada: 3429


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 431321676.txt → 431321676.xml
    723319979.txt: Tokens de entrada: 3382


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 723319979.txt → 723319979.xml
    281224409.txt: Tokens de entrada: 3240


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 281224409.txt → 281224409.xml
    875489792.txt: Tokens de entrada: 3260


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 875489792.txt → 875489792.xml
    647918065.txt: Tokens de entrada: 3371


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 647918065.txt → 647918065.xml
    357460564.txt: Tokens de entrada: 3304


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 357460564.txt → 357460564.xml
    874430747.txt: Tokens de entrada: 2961


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 874430747.txt → 874430747.xml
    792396058.txt: Tokens de entrada: 3145


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 792396058.txt → 792396058.xml
    325406201.txt: Tokens de entrada: 3143


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 325406201.txt → 325406201.xml
    663935917.txt: Tokens de entrada: 3043


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 663935917.txt → 663935917.xml
    058835979.txt: Tokens de entrada: 3165


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 058835979.txt → 058835979.xml
    542182389.txt: Tokens de entrada: 3020


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 542182389.txt → 542182389.xml
    578297463.txt: Tokens de entrada: 3263


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 578297463.txt → 578297463.xml
    451109942.txt: Tokens de entrada: 2748


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 451109942.txt → 451109942.xml
    000096468.txt: Tokens de entrada: 3035


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 000096468.txt → 000096468.xml
    449862324.txt: Tokens de entrada: 3003


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 449862324.txt → 449862324.xml
    089427815.txt: Tokens de entrada: 3111


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 089427815.txt → 089427815.xml
    048827717.txt: Tokens de entrada: 3433


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 048827717.txt → 048827717.xml
    256143448.txt: Tokens de entrada: 3038


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 256143448.txt → 256143448.xml
    081062595.txt: Tokens de entrada: 3045


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 081062595.txt → 081062595.xml
    531425456.txt: Tokens de entrada: 3117


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 531425456.txt → 531425456.xml
    424275665.txt: Tokens de entrada: 3197


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 424275665.txt → 424275665.xml
    643530370.txt: Tokens de entrada: 3215


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 643530370.txt → 643530370.xml
    397968947.txt: Tokens de entrada: 3242


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 397968947.txt → 397968947.xml
    932402402.txt: Tokens de entrada: 3432


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 932402402.txt → 932402402.xml
    448046651.txt: Tokens de entrada: 3088


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 448046651.txt → 448046651.xml
    862537906.txt: Tokens de entrada: 3433


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 862537906.txt → 862537906.xml
    944404412.txt: Tokens de entrada: 3386


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 944404412.txt → 944404412.xml
    289159851.txt: Tokens de entrada: 3357


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 289159851.txt → 289159851.xml
    421395325.txt: Tokens de entrada: 3079


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 421395325.txt → 421395325.xml
    324465502.txt: Tokens de entrada: 2747


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 324465502.txt → 324465502.xml
    434505756.txt: Tokens de entrada: 2696


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 434505756.txt → 434505756.xml
    179617530.txt: Tokens de entrada: 2832


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 179617530.txt → 179617530.xml
    975577618.txt: Tokens de entrada: 3012


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 975577618.txt → 975577618.xml
    964191583.txt: Tokens de entrada: 3098


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 964191583.txt → 964191583.xml
    511726152.txt: Tokens de entrada: 3151


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 511726152.txt → 511726152.xml
    760419145.txt: Tokens de entrada: 3287


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 760419145.txt → 760419145.xml
    029814545.txt: Tokens de entrada: 3229


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 029814545.txt → 029814545.xml
    502064612.txt: Tokens de entrada: 3219


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 502064612.txt → 502064612.xml
    484663980.txt: Tokens de entrada: 3092


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 484663980.txt → 484663980.xml
    009022824.txt: Tokens de entrada: 3264


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 009022824.txt → 009022824.xml
    839003070.txt: Tokens de entrada: 3051


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 839003070.txt → 839003070.xml
    275193382.txt: Tokens de entrada: 3264


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 275193382.txt → 275193382.xml
    236659361.txt: Tokens de entrada: 2983


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 236659361.txt → 236659361.xml
    396147300.txt: Tokens de entrada: 2957


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 396147300.txt → 396147300.xml
    983548640.txt: Tokens de entrada: 3096


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 983548640.txt → 983548640.xml
    976493341.txt: Tokens de entrada: 3305


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 976493341.txt → 976493341.xml
    049093232.txt: Tokens de entrada: 3368


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 049093232.txt → 049093232.xml
    299335709.txt: Tokens de entrada: 2811


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 299335709.txt → 299335709.xml
    719716225.txt: Tokens de entrada: 3316


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 719716225.txt → 719716225.xml
    978387247.txt: Tokens de entrada: 3292


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 978387247.txt → 978387247.xml
    255337556.txt: Tokens de entrada: 3185


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 255337556.txt → 255337556.xml
    818261480.txt: Tokens de entrada: 3268


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 818261480.txt → 818261480.xml
    934277412.txt: Tokens de entrada: 3144


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 934277412.txt → 934277412.xml
    053688516.txt: Tokens de entrada: 3289


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 053688516.txt → 053688516.xml
    023074037.txt: Tokens de entrada: 2905


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 023074037.txt → 023074037.xml
    989036296.txt: Tokens de entrada: 3247


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 989036296.txt → 989036296.xml
    948537830.txt: Tokens de entrada: 3272


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 948537830.txt → 948537830.xml
    607259377.txt: Tokens de entrada: 3331


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 607259377.txt → 607259377.xml
    740144638.txt: Tokens de entrada: 3144


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 740144638.txt → 740144638.xml
    876682700.txt: Tokens de entrada: 3201


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 876682700.txt → 876682700.xml
    425391768.txt: Tokens de entrada: 3222


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 425391768.txt → 425391768.xml
    103395131.txt: Tokens de entrada: 3057


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 103395131.txt → 103395131.xml
    810385877.txt: Tokens de entrada: 3091


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 810385877.txt → 810385877.xml
    521039011.txt: Tokens de entrada: 3181


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 521039011.txt → 521039011.xml
    949789988.txt: Tokens de entrada: 3140


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 949789988.txt → 949789988.xml
    804953670.txt: Tokens de entrada: 3065


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 804953670.txt → 804953670.xml
    462185820.txt: Tokens de entrada: 3026


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 462185820.txt → 462185820.xml
    371953953.txt: Tokens de entrada: 3302


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 371953953.txt → 371953953.xml
    644180013.txt: Tokens de entrada: 3240


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 644180013.txt → 644180013.xml
    852054481.txt: Tokens de entrada: 3260


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 852054481.txt → 852054481.xml
    364114062.txt: Tokens de entrada: 2842


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 364114062.txt → 364114062.xml
    504163138.txt: Tokens de entrada: 3429


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 504163138.txt → 504163138.xml
    248559323.txt: Tokens de entrada: 3339


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 248559323.txt → 248559323.xml
    827278311.txt: Tokens de entrada: 3137


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 827278311.txt → 827278311.xml
    262107453.txt: Tokens de entrada: 3276


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 262107453.txt → 262107453.xml
    262291314.txt: Tokens de entrada: 3155


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Procesado: 262291314.txt → 262291314.xml
    743560436.txt: Tokens de entrada: 2754
    Procesado: 743560436.txt → 743560436.xml
    Proceso completado.



```python

```
