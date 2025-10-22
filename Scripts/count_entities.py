"""
Script to analyze entity occurrences in medical datasets and generate Excel report.
Analyzes MEDDOCAN, SPG, SPGExt, and CARMEN-I datasets using .ann files.
"""

import os
import pandas as pd
from collections import defaultdict
import glob

# Define the entities to analyze (from the provided list)
ENTITIES = [
    'NOMBRE_SUJETO_ASISTENCIA',
    'EDAD_SUJETO_ASISTENCIA',
    'SEXO_SUJETO_ASISTENCIA',
    'FAMILIARES_SUJETO_ASISTENCIA',
    'NOMBRE_PERSONAL_SANITARIO',
    'FECHAS',
    'PROFESION',
    'HOSPITAL',
    'CENTRO_SALUD',
    'INSTITUCION',
    'CALLE',
    'TERRITORIO',
    'PAIS',
    'NUMERO_TELEFONO',
    'NUMERO_FAX',
    'CORREO_ELECTRONICO',
    'ID_SUJETO_ASISTENCIA',
    'ID_CONTACTO_ASISTENCIAL',
    'ID_ASEGURAMIENTO',
    'ID_TITULACION_PERSONAL_SANITARIO',
    'ID_EMPLEO_PERSONAL_SANITARIO',
    'IDENTIF_VEHICULOS_NRSERIE_PLACAS',
    'IDENTIF_DISPOSITIVOS_NRSERIE',
    'DIREC_PROT_INTERNET',
    'URL_WEB',
    'IDENTIF_BIOMETRICOS',
    'OTRO_NUMERO_IDENTIF',
    'OTROS_SUJETO_ASISTENCIA'
]

# Define datasets to analyze
DATASETS = ['MEDDOCAN', 'SPG', 'SPGExt', 'CARMEN-I']

# Define splits to analyze
SPLITS = ['train', 'dev', 'test']

def parse_ann_file(file_path):
    """
    Parse a .ann file and extract entity counts.
    
    Args:
        file_path (str): Path to the .ann file
        
    Returns:
        dict: Dictionary with entity type as key and count as value
    """
    entity_counts = defaultdict(int)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse BRAT annotation format: ID\tENTITY_TYPE start_pos end_pos\ttext
                parts = line.split('\t')
                if len(parts) >= 2:
                    # Get the entity type from the second part
                    entity_info = parts[1].split()
                    if len(entity_info) >= 1:
                        entity_type = entity_info[0]
                        if entity_type in ENTITIES:
                            entity_counts[entity_type] += 1
                    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return dict(entity_counts)

def analyze_dataset(dataset_name, base_path="Datasets"):
    """
    Analyze a specific dataset across all splits.
    
    Args:
        dataset_name (str): Name of the dataset
        base_path (str): Base path to datasets directory
        
    Returns:
        dict: Dictionary with split as key and entity counts as value
    """
    dataset_results = {}
    
    for split in SPLITS:
        ann_path = os.path.join(base_path, dataset_name, split, 'brat')
        
        if not os.path.exists(ann_path):
            print(f"Warning: {ann_path} does not exist for {dataset_name}")
            dataset_results[split] = {entity: 0 for entity in ENTITIES}
            continue
        
        # Find all .ann files
        ann_files = glob.glob(os.path.join(ann_path, '*.ann'))
        
        if not ann_files:
            print(f"Warning: No .ann files found in {ann_path}")
            dataset_results[split] = {entity: 0 for entity in ENTITIES}
            continue
        
        # Initialize entity counts for this split
        split_counts = defaultdict(int)
        
        print(f"Processing {len(ann_files)} files in {dataset_name}/{split}...")
        
        # Process each .ann file
        for ann_file in ann_files:
            file_counts = parse_ann_file(ann_file)
            for entity, count in file_counts.items():
                split_counts[entity] += count
        
        # Convert to regular dict and ensure all entities are present
        split_results = {}
        for entity in ENTITIES:
            split_results[entity] = split_counts[entity]
        
        dataset_results[split] = split_results
    
    return dataset_results

def create_excel_report(all_results, output_file="/home/usuario/Documentos/TrabajoEspecial/Modelos/Scripts/entity_analysis_report.xlsx"):
    """
    Create an Excel report with entity counts for all datasets and splits.
    
    Args:
        all_results (dict): Results from all datasets
        output_file (str): Output Excel file name
    """
    
    # Create a list to store all data for DataFrame
    data_rows = []
    
    for dataset_name, dataset_data in all_results.items():
        for split_name, split_data in dataset_data.items():
            for entity_name, count in split_data.items():
                data_rows.append({
                    'Dataset': dataset_name,
                    'Split': split_name,
                    'Entity': entity_name,
                    'Count': count
                })
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Create comparative table with proper structure
        comparative_data = []
        
        for entity in ENTITIES:
            row_data = {'Entity': entity}
            
            for dataset in DATASETS:
                for split in SPLITS:
                    if dataset in all_results and split in all_results[dataset]:
                        count = all_results[dataset][split].get(entity, 0)
                    else:
                        count = 0
                    row_data[f'{dataset}_{split}'] = count
            
            comparative_data.append(row_data)
        
        comparative_df = pd.DataFrame(comparative_data)
        
        # Write comparative table to Excel
        comparative_df.to_excel(writer, sheet_name='Comparative_Table', index=False)
        
        # Get the workbook and worksheet to modify headers
        workbook = writer.book
        worksheet = writer.sheets['Comparative_Table']
        
        # Clear the first row and insert our custom headers
        worksheet.delete_rows(1)
        worksheet.insert_rows(1, 2)
        
        # Set the custom headers
        # First row: Dataset names
        worksheet.cell(row=1, column=1, value='Entity')
        col = 2
        for dataset in DATASETS:
            for split in SPLITS:
                worksheet.cell(row=1, column=col, value=dataset)
                col += 1
        
        # Second row: train/dev/test repeated
        worksheet.cell(row=2, column=1, value='Entity')
        col = 2
        for dataset in DATASETS:
            for split in SPLITS:
                worksheet.cell(row=2, column=col, value=split)
                col += 1
        
        # Merge cells for dataset names (each dataset spans 3 columns)
        for i, dataset in enumerate(DATASETS):
            start_col = 2 + (i * 3)  # Start from column 2, each dataset takes 3 columns
            end_col = start_col + 2   # End at start_col + 2 (3 columns total)
            worksheet.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=end_col)
        
        # Create summary by dataset (total counts per dataset/split)
        summary_data = []
        for dataset in DATASETS:
            for split in SPLITS:
                if dataset in all_results and split in all_results[dataset]:
                    total_count = sum(all_results[dataset][split].values())
                else:
                    total_count = 0
                summary_data.append({
                    'Dataset': dataset,
                    'Split': split,
                    'Total_Entities': total_count
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Dataset_Summary', index=False)
        
        # Create pivot table for detailed view
        pivot_df = df.pivot_table(
            values='Count',
            index='Entity',
            columns=['Dataset', 'Split'],
            fill_value=0,
            aggfunc='sum'
        )
        pivot_df.to_excel(writer, sheet_name='Detailed_Pivot')
        
        # Create summary by entity (total counts per entity across all datasets)
        entity_totals = []
        for entity in ENTITIES:
            total_count = 0
            for dataset in DATASETS:
                for split in SPLITS:
                    if dataset in all_results and split in all_results[dataset]:
                        total_count += all_results[dataset][split].get(entity, 0)
            
            entity_totals.append({
                'Entity': entity,
                'Total_Count': total_count
            })
        
        entity_totals_df = pd.DataFrame(entity_totals)
        entity_totals_df = entity_totals_df.sort_values('Total_Count', ascending=False)
        entity_totals_df.to_excel(writer, sheet_name='Entity_Totals', index=False)
    
    print(f"Excel report saved as: {output_file}")
    print("Sheets created:")
    print("  - Comparative_Table: Entities as rows, custom headers (Dataset/Split)")
    print("  - Dataset_Summary: Total entities per dataset/split")
    print("  - Detailed_Pivot: Detailed pivot table view")
    print("  - Entity_Totals: Total counts per entity across all datasets")

def main():
    """
    Main function to run the entity analysis.
    """
    print("Starting entity analysis across all datasets...")
    print(f"Analyzing entities: {', '.join(ENTITIES)}")
    print(f"Analyzing datasets: {', '.join(DATASETS)}")
    print(f"Analyzing splits: {', '.join(SPLITS)}")
    print("-" * 80)
    
    all_results = {}
    
    # Analyze each dataset
    for dataset in DATASETS:
        print(f"\nAnalyzing dataset: {dataset}")
        dataset_results = analyze_dataset(dataset)
        all_results[dataset] = dataset_results
        
        # Print summary for this dataset
        total_entities = 0
        for split, counts in dataset_results.items():
            split_total = sum(counts.values())
            total_entities += split_total
            print(f"  {split}: {split_total} entities")
        print(f"  Total: {total_entities} entities")
    
    # Create Excel report
    print("\n" + "=" * 80)
    print("Creating Excel report...")
    create_excel_report(all_results)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY:")
    print("=" * 80)
    
    for dataset, dataset_data in all_results.items():
        print(f"\n{dataset}:")
        for split, split_data in dataset_data.items():
            total = sum(split_data.values())
            print(f"  {split}: {total} entities")
    
    print(f"\nExcel report generated successfully!")
    print("Check 'entity_analysis_report.xlsx' for detailed results.")

if __name__ == "__main__":
    main() 