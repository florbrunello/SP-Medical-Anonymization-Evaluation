1. Depuración:
    Orden: 
    1. Sacar los de menos de 5 lineas 1error
    2. Buscar "Datos del Paciente. Nombre:" y hacer correcciones 2error
    3. Si eso NO esta en el archivo -> eliminarlo viendo los logs (algunos archivos no tenian esto) 3error
    4. Corregir los archivos restantes: 4error
        i) 975577618 Cosas que empiezan con Antecedentes, luego Datos del Paciente, se borra lo anterior
        ii) Quitar asteriscos: elimina tanto * como **.
        iii) Recorte inicial: arranca desde la primera vez que aparece “Datos del paciente”.
        iv) Elimina notas: corta todo desde “Nota:” inclusive.
        v) Elimina “Por favor…”: borra cualquier texto que empiece con esa frase hasta el final.
        vi) Formatea el encabezado paciente/nombre en dos líneas.
        vii) Añade punto a “Datos asistenciales”.
        viii) Descarta líneas vacías para dejar el archivo compacto.
    5. Modificacion a mano 982570255
    Datos del paciente.
    Nombre: Adam Simon Villanueva
    DNI: 42817526J
    Fecha de nacimiento: 06/11/1992
    Género: M
    Domicilio: Calle de Santa Ana 99
    Ciudad: Torrelodones, Vizcaya, Pais Vasco
    Código postal: 48999
    Email: adamvillanueva278@uhu.es
    Teléfono fijo: +34 946 76 82 09
    Teléfono móvil: +34 646 98 90 78
    NHC: 2523300
    NASS: 022204004785
    Condición de riesgo: Médico
    Datos asistenciales.
    Médico: Dra. Gema Hurtado Merino. NC 871218741. Investigadora Principal en Optometría Clínica. Instituto Universitario de Oftalmobiología Aplicada (IOBA). Avenida Ramón y Cajal, 7. 47011. Valladolid. España.
    Fecha de ingreso: 17/11/2024
    Episodio: 55960149
    Hospital: Hospital Universitario Severo Ochoa
    Matrícula del coche: 0773CHG
    Modelo: Tesla Model 3
    VIN: VSTD3QYPF9I684069
    Informe clínico del paciente:
    Paciente vegano de 32 años de edad, acompañado de su madre.
    (Está dentro del rango de 100-1200 palabras)
    Paciente vegano de 32 años de edad, acompañado de su madre, quien refiere que su hijo ha presentado una serie de síntomas en los últimos días, incluyendo dolor de cabeza, mareos y visión borrosa. El paciente tiene una historia de alergia a los alimentos y ha sido vegano durante varios años.
    La exploración física revela: Tª 37,5 C; T.A: 120/80 mmHg; Fc: 80 lpm. Se encuentra consciente, orientado, sudoroso, eupneico, con buen estado de nutrición e hidratación. En cabeza y cuello no se palpan adenopatías, ni bocio ni ingurgitación de vena yugular, con pulsos carotídeos simétricos. Auscultación cardíaca rítmica, sin soplos, roces ni extratonos. Auscultación pulmonar con conservación del murmullo vesicular. Abdomen blando, depresible, sin masas ni megalias. En la exploración neurológica no se detectan signos meníngeos ni datos de focalidad. Extremidades sin varices ni edemas. Pulsos periféricos presentes y simétricos.
    Los datos analíticos muestran los siguientes resultados: Hemograma: Hb 14,5 g/dl; leucocitos 8.500/mm3 (neutrófilos 70%); plaquetas 250.000/ mm3. VSG: 30 mm 1ª hora. Coagulación: TQ 95%; TTPA 25,8 seg. Bioquímica: Glucosa 90 mg/dl; urea 15 mg/dl; creatinina 0,8 mg/dl; sodio 140 mEq/l; potasio 3,5 mEq/l; GOT 12 U/l; GPT 20 U/l; GGT 25 U/l; fosfatasa alcalina 120 U/l; calcio 9,2 mg/dl. Orina: sedimento normal.
    Se solicitan pruebas de imagen (Ecografía abdominal, TAC craneal, Ecocardiograma transtorácico) para evaluar la posible causa de los síntomas del paciente. Los resultados de las pruebas muestran una serie de hallazgos que sugieren una posible causa de los síntomas del paciente.
    Con el diagnóstico de una posible causa
    
2. Generar .ann: 
    La carpeta 5ann_spg contiene los .ann generador son SPG -> para ello se hicieron modificaciones en SPG. 
        Dados los txt existentes (no se debieron generar nuevamente) -> se creo la carpeta existing_txt y 
        en el archivo main.py se anñadio la funcion generate_brat_from_existing_txt. 
    La carpeta 6ann_llama_1 contiene los .ann generados con 6generar_ann_p1 -> prompt en prompts.md. 
    La carpeta 6ann_llama_2 contiene los .ann generados con 6generar_ann_p2 -> prompt en prompts.md. 
    
3. Corregir ann: 
    La carpeta 8ann_corregidos tiene las correcciones junto a 8corregir_ann.py
    
4. Rehacer train / dev / test usando 4txt y 8ann_corregidos.

5. En vez de hacer (2) (3) (4) se hicieron xml con 9xml y 9generar_xml 
Nota 1: se ajustaron los tokens según la cantidad máxima de Llama y los tokens de input (3400 aprox). Sino se cortaba el output. Ejemplo de como se cortaban los outputs en 9_048699967.xml. 
Nota 2: luego de correr generar_xml y obtener los 448 xml 453095041.xml le faltaba cerrar con </TEXT></MEDDOCAN> por lo que se le agregó a mano ambas etiquetas. 
En 10xml esta la version final en xml

6. En 11_ann estan los ann donde se quitaron etiquetas que NO estén en el listado y se generaron los .ann usando el formato y estándar esperados por el modelo. 
Extrae todas las etiquetas <tag> del XML cuyo atributo type esté en la lista ALLOWED_TAGS.
Devuelve una lista de tuplas tipo: [("FECHAS", "12/04/2023"), ("HOSPITAL", "Hospital Central"), ...]
Para cada entidad extraída, busca todas sus apariciones exactas dentro del texto plano usando la función find_all_occurrences.

Versiones finales (4)txt (10)xml (11)ann