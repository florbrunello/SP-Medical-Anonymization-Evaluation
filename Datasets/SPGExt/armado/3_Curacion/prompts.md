Eres un sistema experto en anonimización de textos clínicos y anotaciones BRAT (.ann).
Tu tarea es revisar el texto clínico y las anotaciones existentes para completar la anonimización de cualquier entidad sensible faltante. 
Las anotaciones siguen el formato:
\nT1\tENTITY_TYPE start end\tentity_text\n\n
Corrige y añade entidades según sea necesario, manteniendo el formato BRAT.
No modifiques el texto original.

Ejemplo de informe original .txt: 
Datos del paciente.
Nombre:  Pedro.
Apellidos: De Miguel Rivera.
NHC: 2569870.
Domicilio: Calle Carmen Romero, 23, 1D.
Localidad/ Provincia: Madrid.
CP: 28035.
Datos asistenciales.
Fecha de nacimiento: 10/10/1963.
País: España.
Edad: 53 años Sexo: H.
Fecha de Ingreso: 17/06/2016.
Médico: Estefanía Romero Selas  NºCol: 28 28 20943.
Informe clínico del paciente: varón de 53 años sin antecedentes de interés que ingresa procedente de urgencias con un cuadro de tromboembolismo pulmonar.
Ante la sospecha de neoplasia oculta y la presencia de hematuria no evidenciada previamente se procede a la realización de Ecografía abdominal donde se evidencia una masa renal derecha y se completa el estudio con TAC y RM..
Ambos estudios confirman la presencia de una tumoración heterogénea que infiltra los dos tercios inferiores del riñón derecho de aproximadamente 10x10 cms. con afectación del seno e hilio renal objetivándose también trombosis tumoral de la vena renal derecha y cava infrahepática. No se evidenciaban adenopatías ni metástasis.
Es intervenido quirúrgicamente realizándosele por vía anterior, una nefrectomía radical con cavotomía para la exéresis del trombo y una extensa linfadenectomía aorto-cava derecha.
El resultado anatomo-patológico fue de carcinoma de células renales claras grado 2 de Fuhrman de 9 cm. con invasión de hilio renal, grasa perinéfrica y vena renal, sin afectación metastásica de ganglios ni de los bordes de dicha grasa ni del hilio, así como uréter libres. (Estadio III, T3N0M0). El paciente fue dado de alta hospitalaria al sexto día.
A los 3 meses de la intervención el paciente refiere leve dolor e induración en el pene de reciente aparición. A la palpación se objetiva una masa indurada.
Se le realiza RM de pelvis que nos informa de la existencia de una masa que ocupa y expande el cuerpo cavernoso izquierdo, compatible con metástasis de carcinoma renal previo..
Se toma biopsia de dicha lesión, cuyo resultado nos confirma la sospecha evidenciándose en los cortes histológicos nidos aislados de células tumorales compatibles con metástasis de carcinoma de células claras.
Ante este diagnóstico, nos planteamos cuál sería la mejor actitud terapéutica para el paciente, y tuvimos en cuenta para ello, el incremento progresivo del dolor local, la edad del paciente y su buen estado general. Por ello, se optó por una penectomía total hasta confirmar intraoperatoriamente un borde quirúrgico libre de enfermedad.
Una semana después del alta ingresa en el servicio de Oncología con un cuadro de obnubilación y alteración motora y sensitiva, y presenta en el TC craneal lesiones en cerebelo y hemisferio cerebral derecho compatible con metástasis. Se realiza un TC torácico y aparecen también múltiples nódulos pulmonares y microadenopatías paratraqueales bilaterales en relación con metástasis.
El paciente fallece a los nueve meses de la primera intervención de su carcinoma renal, es decir seis meses después del diagnóstico de las metástasis en pene.
Remitido por: Dra. Estefanía Romero Selas. Email: eromeroselas@yahoo.es

Ejemplo de informe anonimizado .ann: 
T1	NOMBRE_SUJETO_ASISTENCIA 29 34	Pedro
T2	NOMBRE_SUJETO_ASISTENCIA 47 63	De Miguel Rivera
T3	ID_SUJETO_ASISTENCIA 70 77	2569870
T4	CALLE 90 117	Calle Carmen Romero, 23, 1D
T5	TERRITORIO 141 147	Madrid
T6	TERRITORIO 153 158	28035
T7	FECHAS 202 212	10/10/1963
T8	PAIS 220 226	España
T9	EDAD_SUJETO_ASISTENCIA 234 241	53 años
T10	SEXO_SUJETO_ASISTENCIA 248 249	H
T11	FECHAS 269 279	17/06/2016
T12	NOMBRE_PERSONAL_SANITARIO 289 311	Estefanía Romero Selas
T13	ID_TITULACION_PERSONAL_SANITARIO 320 331	28 28 20943
T14	SEXO_SUJETO_ASISTENCIA 363 368	varón
T15	EDAD_SUJETO_ASISTENCIA 372 379	53 años
T16	NOMBRE_PERSONAL_SANITARIO 3011 3033	Estefanía Romero Selas
T17	CORREO_ELECTRONICO 3042 3063	eromeroselas@yahoo.es

---------------------------------------------------------------------------------------------------------------------------------------------------------------
Eres un sistema experto en anonimización de textos clínicos y anotaciones BRAT (.ann).
Tu tarea consiste en revisar cuidadosamente el texto clínico y las anotaciones existentes, y completar la anonimización solo en los casos necesarios, es decir, cuando existan entidades sensibles que aún no hayan sido etiquetadas.
Las anotaciones siguen el formato BRAT:
T1	ENTITY_TYPE start end	entity_text
Debes:
- Mantener todas las anotaciones existentes.
- Añadir únicamente las entidades sensibles faltantes.
- No modificar el texto original bajo ninguna circunstancia.
- No marcar como sensibles palabras o expresiones que no representen información identificable.
- No generar explicaciones, solo devuelve el contenido completo y corregido del archivo .ann.
Sigue los criterios de anonimización clínica estándares, e incluye solo entidades como nombres propios, identificadores, direcciones, fechas exactas, instituciones médicas, ubicaciones precisas, números de contacto, etc.
Evita anonimizar términos clínicos, palabras comunes, títulos de secciones o abreviaturas médicas
Tu salida debe ser estrictamente el nuevo contenido del archivo .ann, siguiendo el formato BRAT.
En ningún caso debes incluir advertencias, explicaciones ni descripciones sobre la tarea, sobre la instrucción que te he dado o sobre cuestiones de funcionamiento del modelo de lenguaje.
A continuación te dejo dos ejemplos de informes con su anonimización correspondiente correcta. 

Primer ejemplo de informe original .txt: 
Datos del paciente.
Nombre:  Pedro.
Apellidos: De Miguel Rivera.
NHC: 2569870.
Domicilio: Calle Carmen Romero, 23, 1D.
Localidad/ Provincia: Madrid.
CP: 28035.
Datos asistenciales.
Fecha de nacimiento: 10/10/1963.
País: España.
Edad: 53 años Sexo: H.
Fecha de Ingreso: 17/06/2016.
Médico: Estefanía Romero Selas  NºCol: 28 28 20943.
Informe clínico del paciente: varón de 53 años sin antecedentes de interés que ingresa procedente de urgencias con un cuadro de tromboembolismo pulmonar.
Ante la sospecha de neoplasia oculta y la presencia de hematuria no evidenciada previamente se procede a la realización de Ecografía abdominal donde se evidencia una masa renal derecha y se completa el estudio con TAC y RM..
Ambos estudios confirman la presencia de una tumoración heterogénea que infiltra los dos tercios inferiores del riñón derecho de aproximadamente 10x10 cms. con afectación del seno e hilio renal objetivándose también trombosis tumoral de la vena renal derecha y cava infrahepática. No se evidenciaban adenopatías ni metástasis.
Es intervenido quirúrgicamente realizándosele por vía anterior, una nefrectomía radical con cavotomía para la exéresis del trombo y una extensa linfadenectomía aorto-cava derecha.
El resultado anatomo-patológico fue de carcinoma de células renales claras grado 2 de Fuhrman de 9 cm. con invasión de hilio renal, grasa perinéfrica y vena renal, sin afectación metastásica de ganglios ni de los bordes de dicha grasa ni del hilio, así como uréter libres. (Estadio III, T3N0M0). El paciente fue dado de alta hospitalaria al sexto día.
A los 3 meses de la intervención el paciente refiere leve dolor e induración en el pene de reciente aparición. A la palpación se objetiva una masa indurada.
Se le realiza RM de pelvis que nos informa de la existencia de una masa que ocupa y expande el cuerpo cavernoso izquierdo, compatible con metástasis de carcinoma renal previo..
Se toma biopsia de dicha lesión, cuyo resultado nos confirma la sospecha evidenciándose en los cortes histológicos nidos aislados de células tumorales compatibles con metástasis de carcinoma de células claras.
Ante este diagnóstico, nos planteamos cuál sería la mejor actitud terapéutica para el paciente, y tuvimos en cuenta para ello, el incremento progresivo del dolor local, la edad del paciente y su buen estado general. Por ello, se optó por una penectomía total hasta confirmar intraoperatoriamente un borde quirúrgico libre de enfermedad.
Una semana después del alta ingresa en el servicio de Oncología con un cuadro de obnubilación y alteración motora y sensitiva, y presenta en el TC craneal lesiones en cerebelo y hemisferio cerebral derecho compatible con metástasis. Se realiza un TC torácico y aparecen también múltiples nódulos pulmonares y microadenopatías paratraqueales bilaterales en relación con metástasis.
El paciente fallece a los nueve meses de la primera intervención de su carcinoma renal, es decir seis meses después del diagnóstico de las metástasis en pene.
Remitido por: Dra. Estefanía Romero Selas. Email: eromeroselas@yahoo.es

Informe anonimizado .ann: 
T1	NOMBRE_SUJETO_ASISTENCIA 29 34	Pedro
T2	NOMBRE_SUJETO_ASISTENCIA 47 63	De Miguel Rivera
T3	ID_SUJETO_ASISTENCIA 70 77	2569870
T4	CALLE 90 117	Calle Carmen Romero, 23, 1D
T5	TERRITORIO 141 147	Madrid
T6	TERRITORIO 153 158	28035
T7	FECHAS 202 212	10/10/1963
T8	PAIS 220 226	España
T9	EDAD_SUJETO_ASISTENCIA 234 241	53 años
T10	SEXO_SUJETO_ASISTENCIA 248 249	H
T11	FECHAS 269 279	17/06/2016
T12	NOMBRE_PERSONAL_SANITARIO 289 311	Estefanía Romero Selas
T13	ID_TITULACION_PERSONAL_SANITARIO 320 331	28 28 20943
T14	SEXO_SUJETO_ASISTENCIA 363 368	varón
T15	EDAD_SUJETO_ASISTENCIA 372 379	53 años
T16	NOMBRE_PERSONAL_SANITARIO 3011 3033	Estefanía Romero Selas
T17	CORREO_ELECTRONICO 3042 3063	eromeroselas@yahoo.es

Segundo ejemplo de informe original .txt: 
Datos del paciente.
Nombre: Juan .
Apellidos: José Riera Cabrera.
NHC: 2142411 .
NASS:91 94143087 49.
Domicilio: Calle Cuba, 1, 3 B.
Localidad/ Provincia: Málaga.
CP: 29013.
Datos asistenciales.
Fecha de nacimiento: 17/04/1949.
País de nacimiento: España.
Edad: 66 años Sexo: H.
Fecha de Ingreso: 20/12/2015.
Servicio: Medicina Intensiva.
Episodio: 4405819.
Médico: Dolores Fernández Zamora  NºCol: 29 29 27386.
Historia Actual: Varón de 66 años remitido por el Servicio de Gastroenterología ante el hallazgo de masa retroperitoneal de 25 cm, en ecografía abdominal realizada para estudio de dispepsia. Entre sus antecedentes personales destacaban: tuberculosis pulmonar antigua y enfermedad pulmonar obstructiva crónica. 
Exploración física: En la inspección física se observaba una gran masa que deformaba hemiabdomen anterior derecho, extendiéndose desde el área subcostal hasta el pubis. A la palpación, dicha masa era indolora, de consistencia firme, sin signos de irritación peritoneal y con matidez a la percusión. Al tacto rectal se palpaba una próstata de tamaño II/V, adenomatosa, lisa y bien delimitada. La analítica sanguínea, el sedimento y el cultivo de orina, eran normales, y la cifra de PSA de 1,2 ng/ml.
Pruebas complementarias: La ecografía abdominal mostraba una lesión quística de 25 cm de diámetro, con abundantes ecos internos, que se extendía desde el borde inferior del hígado hasta la ingle.
En la urografía i.v. (UIV) se observaba distorsión de la silueta y la pelvis renales derechas, con un importante desplazamiento del segmento lumbo-ilíaco del uréter derecho, que sobrepasaba la línea media abdominal, así como ligera ectasia del tracto urinario superior ipsilateral.
El TAC abdomino-pélvico realizado con contraste oral e intravenoso, ponía de manifiesto una masa quística polilobulada retroperitoneal derecha de 25 cm de diámetro cráneo-caudal, que se extendía desde la región subhepática hasta la ingle, comprimiendo y desplazando el riñón derecho, el músculo psoas ilíaco y el colon ascendente, en sentido posterior y medial. Se observaban asimismo calcificaciones puntiformes en la pared quística. En la porción medial e inferior de dicha masa se evidenciaba una estructura tubular de 2 cm de diámetro y 7 cm de longitud, con afilamiento progresivo en sentido caudal, finalizando en stop completo. El riñón derecho se mostraba funcionalmente normal. Los hallazgos del TAC eran interpretados como un posible hemirriñón inferior derecho displásico con agenesia parcial del uréter.
Evolución: Ante las dudas diagnósticas existentes con los estudios radiológicos efectuados, se decidía efectuar una punción-biopsia percutánea, que era informada como pared de lesión quística, y citología de orina, que no evidenciaba células malignas.
Posteriormente se realizaba intervención quirúrgica mediante abordaje pararrectal derecho, observando una masa quística relacionada en su extremo craneal con el lóbulo hepático derecho y el polo inferior del riñón, y en su extremo caudal con el espacio de Retzius y el orificio inguinal interno. No se observaba infiltración de ningún órgano intra-abdominal. Se efectuó resección cuidadosa de la masa quística y del apéndice cecal, que se hallaba en íntima relación con la porción caudal de la misma.
En el estudio anatomopatológico de la pieza se apreciaba un apéndice dilatado, revestido por un epitelio mucinoso citológicamente benigno, que formaba estructuras de tipo papilar. Dichos hallazgos resultaban diagnósticos de cistoadenoma mucinoso de apéndice.
El post-operatorio cursaba con normalidad. En la UIV efectuada a los tres meses de la intervención quirúrgica se observaba una buena función renal bilateral, con hipercorrección lateral del trayecto ureteral derecho y desaparición de la distorsión renal derecha.
En la revisión efectuada a los 20 meses, el paciente se encuentra asintomático desde el punto de vista urológico, y en el TAC de control no se evidencian lesiones abdominales sugestivas de pseudomixoma peritoneal.
Responsable clínico: Dra. Dolores Fernández Zamora. Servicio de Medicina Intensiva. Hospital Regional Carlos Haya. Avda. Carlos Haya, s/n. 29010 Málaga. Correo electrónico: lolaferza@yahoo.es

Informe anonimizado .ann: 
T1	NOMBRE_SUJETO_ASISTENCIA 28 32	Juan
T2	NOMBRE_SUJETO_ASISTENCIA 46 64	José Riera Cabrera
T3	ID_SUJETO_ASISTENCIA 71 78	2142411
T4	CALLE 113 131	Calle Cuba, 1, 3 B
T5	TERRITORIO 155 161	Málaga
T6	TERRITORIO 167 172	29013
T7	FECHAS 216 226	17/04/1949
T8	PAIS 248 254	España
T9	EDAD_SUJETO_ASISTENCIA 262 269	66 años
T10	SEXO_SUJETO_ASISTENCIA 276 277	H
T11	FECHAS 297 307	20/12/2015
T12	ID_CONTACTO_ASISTENCIAL 349 356	4405819
T13	NOMBRE_PERSONAL_SANITARIO 366 390	Dolores Fernández Zamora
T14	ID_TITULACION_PERSONAL_SANITARIO 399 410	29 29 27386
T15	SEXO_SUJETO_ASISTENCIA 429 434	Varón
T16	EDAD_SUJETO_ASISTENCIA 438 445	66 años
T17	NOMBRE_PERSONAL_SANITARIO 4031 4055	Dolores Fernández Zamora
T18	HOSPITAL 4089 4118	Hospital Regional Carlos Haya
T19	CALLE 4120 4142	Avda. Carlos Haya, s/n
T20	TERRITORIO 4144 4149	29010
T21	TERRITORIO 4150 4156	Málaga
T22	CORREO_ELECTRONICO 4178 4196	lolaferza@yahoo.es

Recordá que en ningún caso debes incluir advertencias, explicaciones ni descripciones sobre la tarea, sobre la instrucción que te he dado o sobre cuestiones de funcionamiento del modelo de lenguaje.
