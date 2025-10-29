A partir de "generar_xml.py" (de cuando se extendió el Dataset) saltaba el siguiente error:

---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[7], line 133
    131 # Calcular tokens de entrada
    132 full_prompt = prompt_text + texto
--> 133 total_tokens = len(tokenizer.encode(full_prompt))
    134 print(f"{filename}: Tokens de entrada: {total_tokens}")
    136 # Truncar el prompt si se pasa del límite permitido

NameError: name 'tokenizer' is not defined 

Se añadió la línea de código -> 
# Tokenizer necesario para contar tokens
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

El resto todo igual (salvo que saqué la etiqueta NUMERO_BENEF_PLAN_SALUD). 

¿Por qué no usar los xml que ya me da el MEDDOCAN para el ejemplo del prompt? Ejemplo: 
    <NAME id="T17" start="29" end="34" text="Pedro" TYPE="NOMBRE_SUJETO_ASISTENCIA" comment=""/>
Vimos que le cuesta bastante hacer notaciones stand-off, por eso lo hacemos in-line. 

-----------------------------------------------------------------------------------------------------

Generación de xml para dev:

Registro cuánto demoró: 5 hs - 250 archivos
Prompts truncados:

    Procesado: S0212-71992004000600005-1.txt → S0212-71992004000600005-1.xml
    S0212-16112010000600022-1.txt: Tokens de entrada: 4369
    Truncando prompt: S0212-16112010000600022-1.txt

    Procesado: S1699-65852007000300003-1.txt → S1699-65852007000300003-1.xml
    S1134-80462015000200004-1.txt: Tokens de entrada: 4267
    Truncando prompt: S1134-80462015000200004-1.txt

    Procesado: S0211-57352011000100008-1.txt → S0211-57352011000100008-1.xml
    S0211-57352013000300012-1.txt: Tokens de entrada: 4319
    Truncando prompt: S0211-57352013000300012-1.txt

-----------------------------------------------------------------------------------------------------

Generación de xml para test:

Registro cuánto demoró: 6 hs - 250 archivos
Prompts truncados:

    Procesado: S0210-56912006000300007-2.txt → S0210-56912006000300007-2.xml
    S0210-48062003001000009-1.txt: Tokens de entrada: 4238
    Truncando prompt: S0210-48062003001000009-1.txt

    Procesado: S1134-80462008000200008-1.txt → S1134-80462008000200008-1.xml
    S0376-78922015000100011-1.txt: Tokens de entrada: 4332
    Truncando prompt: S0376-78922015000100011-1.txt

    Procesado: S1889-836X2015000100003-1.txt → S1889-836X2015000100003-1.xml
    S0212-71992007000600008-1.txt: Tokens de entrada: 4221
    Truncando prompt: S0212-71992007000600008-1.txt

    Procesado: S0212-71992007000600008-1.txt → S0212-71992007000600008-1.xml
    S1130-63432014000100012-1.txt: Tokens de entrada: 4487
    Truncando prompt: S1130-63432014000100012-1.txt

    Procesado: S1135-76062014000100006-1.txt → S1135-76062014000100006-1.xml
    S0376-78922009000100011-1.txt: Tokens de entrada: 4313
    Truncando prompt: S0376-78922009000100011-1.txt

    Procesado: S0365-66912012000500005-2.txt → S0365-66912012000500005-2.xml
    S0213-12852016000500002-1.txt: Tokens de entrada: 4271
    Truncando prompt: S0213-12852016000500002-1.txt
