import re
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

# ---------- USER CONFIG ----------
mappings_tsv_path = "/home/usuario/Documentos/CARMEN-I/CARMEN1_mappings.tsv"
txt_replaced_dir = "/home/usuario/Documentos/CARMEN-I/txt/replaced"
ann_anon_dir = "/home/usuario/Documentos/CARMEN-I/ann/replaced/anon"
output_dir = "/home/usuario/Documentos/TrabajoEspecial/Datasets/CARMEN-I"
random_seed = 42
# ----------------------------------

os.makedirs(output_dir, exist_ok=True)

# Create train, dev, test directories
train_dir = os.path.join(output_dir, "train")
dev_dir = os.path.join(output_dir, "dev")
test_dir = os.path.join(output_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def extract_report_type(filename: str) -> str:
    """
    Extrae el tipo de informe del nombre de archivo.
    Ejemplo esperado: CARMEN-I_IA_ANTECEDENTES_2.txt -> IA
    Ajustá la regex si tu naming difiere.
    """
    m = re.match(r"^CARMEN-I_([A-Z]+)_", filename)
    if m:
        return m.group(1)
    parts = filename.split("_")
    return parts[1] if len(parts) > 1 else "UNKNOWN"

def copy_files_to_split(df_split, target_dir, split_name):
    """
    Copia los archivos .txt y .ann correspondientes a la carpeta del split.
    """
    txt_copied = 0
    ann_copied = 0
    
    for _, row in df_split.iterrows():
        filename = row['filename']
        
        # Handle both .txt and .ann files
        txt_filename = filename if filename.endswith('.txt') else f"{filename}.txt"
        ann_filename = filename.replace('.txt', '.ann') if filename.endswith('.txt') else f"{filename}.ann"
        
        # Source paths
        txt_source = os.path.join(txt_replaced_dir, txt_filename)
        ann_source = os.path.join(ann_anon_dir, ann_filename)
        
        # Destination paths
        txt_dest = os.path.join(target_dir, txt_filename)
        ann_dest = os.path.join(target_dir, ann_filename)
        
        # Copy txt file
        if os.path.exists(txt_source):
            shutil.copy2(txt_source, txt_dest)
            txt_copied += 1
        else:
            print(f"Warning: TXT file not found: {txt_source}")
        
        # Copy ann file
        if os.path.exists(ann_source):
            shutil.copy2(ann_source, ann_dest)
            ann_copied += 1
        else:
            print(f"Warning: ANN file not found: {ann_source}")
    
    print(f"{split_name}: Copied {txt_copied} TXT files and {ann_copied} ANN files to {target_dir}")
    return txt_copied, ann_copied

# Load mappings TSV
df = pd.read_csv(mappings_tsv_path, sep="\t", dtype=str)  # file has columns like: filename, language, ner_annotations
# Normalize column names
df.columns = [c.strip() for c in df.columns]

# Filter only Spanish documents
if "language" not in df.columns:
    raise ValueError("Expected a 'language' column in the mappings TSV.")
df_es = df[df["language"].str.lower().isin(["es", "español", "es-ES"])].copy()
print(f"Total documents in mappings file: {len(df)}")
print(f"Spanish documents after filtering: {len(df_es)}")

# Ensure filename column exists
if "filename" not in df_es.columns:
    raise ValueError("Expected a 'filename' column in the mappings TSV.")

# Extract or infer report/document type
if "report_type" not in df_es.columns:
    df_es["report_type"] = df_es["filename"].apply(extract_report_type)
else:
    # if present but possibly different naming, you may want to normalize it
    df_es["report_type"] = df_es["report_type"].astype(str).str.strip()

# Show counts per report_type
counts = df_es["report_type"].value_counts().to_dict()
print("Counts by report_type (Spanish only):")
for k, v in counts.items():
    print(f"  {k}: {v}")

# Safety check: number of documents
n_total = len(df_es)
if n_total == 0:
    raise SystemExit("No Spanish documents found. Check the mappings TSV and language codes.")

# Basic parameters
test_frac = 0.25   # final test = 25%
train_frac = 0.50  # final train = 50%
dev_frac = 0.25    # final dev = 25%

# We'll first split: temp (train+dev) = 75%, test = 25%
temp_frac = 1.0 - test_frac  # 0.75
# For the second step, train relative to temp should be train_frac / temp_frac = 0.50/0.75 = 2/3
train_relative_to_temp = train_frac / temp_frac  # 0.666666...

y = df_es["report_type"].values

# Pre-check: identify classes with very small counts
class_counts = df_es["report_type"].value_counts()
min_count = class_counts.min()
print(f"Minimum samples in a class: {min_count}")

# If any class has fewer than 3 examples, it's impossible to have it in all three splits.
# The CARMEN-I paper guarantees >=5 for the smallest classes, but we include a fallback.
min_required_for_all_three = 3
if (class_counts < min_required_for_all_three).any():
    small = class_counts[class_counts < min_required_for_all_three]
    print("Warning: the following classes have <3 samples and cannot be present in all 3 splits:")
    print(small)
    print("We'll proceed with a special manual distribution for those classes (one per split where possible).")

# Do the first stratified split: temp (75%) / test (25%)
# If a class has less than 2 samples, sklearn stratify may fail; but with >=3 it should be okay.
try:
    df_temp, df_test = train_test_split(
        df_es,
        test_size=test_frac,
        stratify=y,
        random_state=random_seed,
        shuffle=True,
    )
except ValueError as e:
    # Fall back to a manual split per-class if stratified split fails
    print("Stratified split failed (likely due to very small classes). Falling back to manual per-class allocation.")
    temp_rows = []
    test_rows = []
    rng = pd.np.random.RandomState(random_seed)  # slight deprecation but works; otherwise use numpy directly
    for rpt, group in df_es.groupby("report_type"):
        n = len(group)
        n_test = int(round(n * test_frac))
        n_test = max(1, n_test) if n >= min_required_for_all_three else 1  # ensure at least 1 for reasonably sized groups
        perm = group.sample(frac=1.0, random_state=random_seed)
        test_rows.append(perm.iloc[:n_test])
        temp_rows.append(perm.iloc[n_test:])
    df_test = pd.concat(test_rows).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_temp = pd.concat(temp_rows).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

# Second split: df_temp -> train and dev with stratify on report_type
y_temp = df_temp["report_type"].values
try:
    df_train, df_dev = train_test_split(
        df_temp,
        test_size=(1.0 - train_relative_to_temp),  # 1/3 of temp -> dev
        stratify=y_temp,
        random_state=random_seed,
        shuffle=True,
    )
except ValueError as e:
    # If stratified split fails here (rare for counts >=5), do manual per-class distribution from df_temp
    print("Second stratified split failed. Performing manual per-class distribution inside train+dev.")
    train_parts = []
    dev_parts = []
    for rpt, group in df_temp.groupby("report_type"):
        n = len(group)
        # allocate at least one in dev and remainder to train, but keep proportions roughly
        if n >= 3:
            n_dev = int(round(n * (dev_frac / temp_frac)))  # dev relative to temp (should be ~1/3)
            n_dev = max(1, n_dev)  # at least one to dev
        else:
            # if n==1 or 2, put one in dev and rest in train
            n_dev = 1
        perm = group.sample(frac=1.0, random_state=random_seed)
        dev_parts.append(perm.iloc[:n_dev])
        train_parts.append(perm.iloc[n_dev:])
    df_dev = pd.concat(dev_parts).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_train = pd.concat(train_parts).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

# Final sanity checks: ensure each report_type appears in each split
def ensure_presence(train_df, dev_df, test_df, key="report_type"):
    train_set = set(train_df[key].unique())
    dev_set = set(dev_df[key].unique())
    test_set = set(test_df[key].unique())
    all_types = set(df_es[key].unique())
    missing = {}
    for t in all_types:
        missing_in = []
        if t not in train_set:
            missing_in.append("train")
        if t not in dev_set:
            missing_in.append("dev")
        if t not in test_set:
            missing_in.append("test")
        if missing_in:
            missing[t] = missing_in
    return missing

missing = ensure_presence(df_train, df_dev, df_test)
if missing:
    print("Post-split, some report types are missing from splits. Attempting to fix by moving samples where possible.")
    # For each missing class in a split, try to move one sample from a split where that class has >1 sample
    for rpt, missing_in in missing.items():
        for target in missing_in:
            moved = False
            # Try to move from train -> target, else dev -> target, else test -> target
            for source_df_name in ["df_train", "df_dev", "df_test"]:
                if source_df_name == f"df_{target}":
                    continue
                source_df = locals()[source_df_name]
                # find candidates of that rpt in source
                candidates = source_df[source_df["report_type"] == rpt]
                if len(candidates) > 1:
                    # move one
                    row_to_move = candidates.sample(n=1, random_state=random_seed)
                    # remove from source
                    locals()[source_df_name] = source_df.drop(index=row_to_move.index)
                    # add to target
                    locals()[f"df_{target}"] = pd.concat([locals()[f"df_{target}"], row_to_move], ignore_index=True)
                    moved = True
                    print(f"Moved one sample of {rpt} from {source_df_name} to df_{target}")
                    break
            if not moved:
                # As last resort, if some split has zero but another has exactly 1, move that 1 (this will make source 0)
                for source_df_name in ["df_train", "df_dev", "df_test"]:
                    if source_df_name == f"df_{target}":
                        continue
                    source_df = locals()[source_df_name]
                    candidates = source_df[source_df["report_type"] == rpt]
                    if len(candidates) == 1:
                        row_to_move = candidates
                        locals()[source_df_name] = source_df.drop(index=row_to_move.index)
                        locals()[f"df_{target}"] = pd.concat([locals()[f"df_{target}"], row_to_move], ignore_index=True)
                        moved = True
                        print(f"Moved sole sample of {rpt} from {source_df_name} to df_{target} (source may now lack the class).")
                        break
            if not moved:
                print(f"Could not move any sample to fill {rpt} in {target}. You may need to adjust manually.")
    # recompute missing
    missing = ensure_presence(locals()["df_train"], locals()["df_dev"], locals()["df_test"])
    if missing:
        print("After attempting fixes, these types are still missing in some splits (manual intervention needed):")
        print(missing)
    else:
        print("All report types are now present in every split.")

# Print final counts
def print_counts(d, name):
    c = d["report_type"].value_counts().to_dict()
    print(f"\n{name} size: {len(d)}")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")

print_counts(df_train, "TRAIN")
print_counts(df_dev, "DEV")
print_counts(df_test, "TEST")

# Copy files to respective directories
print("\nCopying files to split directories...")
copy_files_to_split(df_train, train_dir, "TRAIN")
copy_files_to_split(df_dev, dev_dir, "DEV")
copy_files_to_split(df_test, test_dir, "TEST")

print(f"\nRandom seed used: {random_seed}")
print(f"Files copied to: {train_dir}, {dev_dir}, {test_dir}")
