"""
Script para corregir espacios extra antes de XDIRECCIONX en la columna anonymized.
Reemplaza "  XDIRECCIONX" (dos espacios) y "   XDIRECCIONX" (tres espacios) por " XDIRECCIONX" (un espacio).
"""

import pandas as pd
import re
import sys
from pathlib import Path

def fix_xdireccionx_spaces(input_file, output_file=None):
    """
    Corrige espacios extra antes de XDIRECCIONX en la columna anonymized.
    
    Args:
        input_file (str): Ruta al archivo CSV de entrada
        output_file (str): Ruta al archivo CSV de salida (opcional)
    """
    
    # Si no se especifica archivo de salida, crear uno con sufijo _fixed
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
    
    print(f"Leyendo archivo: {input_file}")
    
    try:
        # Leer el CSV
        df = pd.read_csv(input_file)
        
        # Verificar que existe la columna anonymized
        if 'anonymized' not in df.columns:
            print("Error: No se encontró la columna 'anonymized' en el archivo CSV")
            print(f"Columnas disponibles: {list(df.columns)}")
            return False
        
        print(f"Archivo leído correctamente. Filas: {len(df)}")
        print(f"Columnas: {list(df.columns)}")
        
        # Contadores para estadísticas
        total_fixes_2spaces = 0
        total_fixes_3spaces = 0
        
        # Aplicar la corrección a la columna anonymized
        def fix_spaces_before_xdireccionx(text):
            if pd.isna(text) or not isinstance(text, str):
                return text
            
            # Contar cuántas veces se aplica cada corrección
            original_text = text
            
            # Contar ocurrencias antes de reemplazar
            fixes_2spaces = original_text.count('  XDIRECCIONX')
            fixes_3spaces = original_text.count('   XDIRECCIONX')
            
            # Aplicar las correcciones
            # Primero reemplazar tres espacios (para evitar conflictos)
            fixed_text = re.sub(r'   XDIRECCIONX', ' XDIRECCIONX', text)
            # Luego reemplazar dos espacios
            fixed_text = re.sub(r'  XDIRECCIONX', ' XDIRECCIONX', fixed_text)
            
            # Actualizar contadores globales
            nonlocal total_fixes_2spaces, total_fixes_3spaces
            total_fixes_2spaces += fixes_2spaces
            total_fixes_3spaces += fixes_3spaces
            
            return fixed_text
        
        # Aplicar la función a la columna anonymized
        df['anonymized'] = df['anonymized'].apply(fix_spaces_before_xdireccionx)
        
        # Guardar el archivo corregido
        df.to_csv(output_file, index=False)
        
        print(f"Archivo corregido guardado como: {output_file}")
        print(f"Total de correcciones aplicadas:")
        print(f"  - Dos espacios antes de XDIRECCIONX: {total_fixes_2spaces}")
        print(f"  - Tres espacios antes de XDIRECCIONX: {total_fixes_3spaces}")
        print(f"  - Total general: {total_fixes_2spaces + total_fixes_3spaces}")
        
        # Mostrar algunos ejemplos de las correcciones
        total_fixes = total_fixes_2spaces + total_fixes_3spaces
        if total_fixes > 0:
            print("\nEjemplos de correcciones:")
            examples = 0
            for idx, row in df.iterrows():
                anonymized_text = str(row.get('anonymized', ''))
                if '  XDIRECCIONX' in anonymized_text or '   XDIRECCIONX' in anonymized_text:
                    # Mostrar el tipo de corrección
                    if '   XDIRECCIONX' in anonymized_text:
                        print(f"Fila {idx}: '   XDIRECCIONX' -> ' XDIRECCIONX' (3 espacios -> 1 espacio)")
                    elif '  XDIRECCIONX' in anonymized_text:
                        print(f"Fila {idx}: '  XDIRECCIONX' -> ' XDIRECCIONX' (2 espacios -> 1 espacio)")
                    examples += 1
                    if examples >= 5:  # Mostrar máximo 5 ejemplos
                        break
        
        return True
        
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        return False

def main():
    """Función principal del script."""
    
    # Ruta del archivo de entrada
    input_file = "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/b) SPG/out.csv"
    
    # Verificar que el archivo existe
    if not Path(input_file).exists():
        print(f"Error: El archivo {input_file} no existe")
        sys.exit(1)
    
    # Ruta del archivo de salida
    output_file = "/home/usuario/Documentos/TrabajoEspecial/Modelos/REGEX/b) SPG/out_post.csv"
    
    print("=== Script de Corrección de Espacios XDIRECCIONX ===")
    print(f"Archivo de entrada: {input_file}")
    print(f"Archivo de salida: {output_file}")
    print()
    
    # Ejecutar la corrección
    success = fix_xdireccionx_spaces(input_file, output_file)
    
    if success:
        print("\n✅ Proceso completado exitosamente")
    else:
        print("\n❌ Error en el proceso")
        sys.exit(1)

if __name__ == "__main__":
    main() 