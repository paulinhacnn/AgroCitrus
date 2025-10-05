#!/usr/bin/env python3
# prepare_dataset.py
"""
Prepara dataset multimodal a partir de um CSV existente (dados.csv).
Gera:
 - dataset.csv (image_path, text, label)  [multiclass]
 - train.csv, val.csv, test.csv (estratificados)
 - label_map.json
 - dataset_multilabel.csv (se detectar multilabels) com colunas binárias por classe
 - missing_images.txt (lista de imagens não encontradas)
 
Uso:
    python prepare_dataset.py
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# ---------- CONFIG ----------
DEFAULT_INPUT = "dados.csv"    # seu arquivo já existente
IMG_ROOT = "images"            # pasta onde estão as imagens (relativa ao script)
OUTPUT_DIR = "."               # onde salvar os arquivos gerados
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
# ----------------------------

np.random.seed(RANDOM_SEED)

POSSIBLE_IMAGE_COLS = ["image_path", "path", "filepath", "filename", "file", "img", "img_path"]
POSSIBLE_TEXT_COLS = ["text", "description", "descricao", "desc", "annot", "annotation", "metadata"]
POSSIBLE_LABEL_COLS = ["label", "labels", "class", "classe", "target", "y"]

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive
    cols_lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def detect_multilabel(series):
    # returns (is_multilabel, sep) - detects separators ; or ,
    if series.dtype == object:
        sample = series.dropna().astype(str)
        sample = sample[sample.str.len() > 0]
        if len(sample) == 0:
            return False, None
        # if any row has ';' or ',' assume multilabel
        has_semicolon = sample.str.contains(";").any()
        has_comma = sample.str.contains(",").any()
        if has_semicolon:
            return True, ";"
        if has_comma:
            # but may be normal comma in numeric? we assume multilabel if commas present and not numeric
            # check if entries are numeric lists or single numbers with commas suspicious -> treat as multilabel
            return True, ","
    return False, None

def normalize_path(p):
    if pd.isna(p):
        return ""
    p = str(p).strip()
    # remove leading ./
    if p.startswith("./"):
        p = p[2:]
    return p

def main(args):
    input_csv = args.input
    img_root = args.img_root
    out_dir = args.output_dir

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        print(f"ERRO: arquivo '{input_csv}' não encontrado. Coloque o arquivo no diretório ou passe --input.")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    print(f"Leu {len(df)} linhas de {input_csv}")

    # Detectar colunas
    img_col = find_column(df, POSSIBLE_IMAGE_COLS)
    text_col = find_column(df, POSSIBLE_TEXT_COLS)
    label_col = find_column(df, POSSIBLE_LABEL_COLS)

    print("Detecção de colunas:")
    print("  imagem:", img_col)
    print("  texto:", text_col)
    print("  label:", label_col)

    # Se nada detectado para image, tentar inferir de outras colunas (primeira coluna com extensão jpg/png)
    if img_col is None:
        for col in df.columns:
            # look for values that look like filenames
            sample_vals = df[col].dropna().astype(str).head(50).tolist()
            if any(v.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) for v in sample_vals):
                img_col = col
                print(f"Inferi coluna de imagem como '{img_col}'")
                break

    if img_col is None:
        print("ERRO: não localizei coluna com caminhos de imagem. Nomes esperados:", POSSIBLE_IMAGE_COLS)
        sys.exit(1)

    # Normalize image paths
    df["__image_path_raw__"] = df[img_col].apply(normalize_path)

    # Text column fallback: criar texto a partir do filename se vazio
    if text_col is None:
        df["__text__"] = df["__image_path_raw__"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
        print("Nenhuma coluna text encontrada — o campo 'text' será preenchido com o nome do arquivo.")
    else:
        df["__text__"] = df[text_col].fillna("").astype(str).apply(lambda s: s.strip() if str(s).strip() != "" else None)
        # fill empty with filename
        df["__text__"] = df.apply(lambda r: r["__text__"] if r["__text__"] not in (None, "None", "") else os.path.splitext(os.path.basename(r["__image_path_raw__"]))[0], axis=1)

    # Labels
    is_multilabel = False
    multilabel_sep = None
    if label_col is None:
        print("AVISO: não foi encontrada coluna de rótulo. Vou criar uma coluna 'label' vazia (-1).")
        df["__label_raw__"] = -1
        label_map = {}
    else:
        df["__label_raw__"] = df[label_col]
        # detect multilabel
        is_multilabel, multilabel_sep = detect_multilabel(df["__label_raw__"])
        if is_multilabel:
            print("Multilabel detectado com separador:", repr(multilabel_sep))

    # Normalize and check image files exist
    missing = []
    for p in df["__image_path_raw__"].values:
        fullp = os.path.join(img_root, p)
        if not os.path.exists(fullp):
            missing.append(p)
    if missing:
        print(f"Aviso: {len(missing)} imagens aparentemente ausentes (ver {os.path.join(out_dir, 'missing_images.txt')}).")
        with open(os.path.join(out_dir, "missing_images.txt"), "w", encoding="utf-8") as f:
            for m in missing:
                f.write(m + "\n")
    else:
        print("Todas as imagens referenciadas foram encontradas em", img_root)

    # Process labels
    if is_multilabel:
        # explode labels, build label map
        def split_labels(x):
            if pd.isna(x):
                return []
            s = str(x).strip()
            if s == "":
                return []
            parts = [t.strip() for t in s.split(multilabel_sep) if t.strip() != ""]
            return parts
        df["__labels_list__"] = df["__label_raw__"].apply(split_labels)
        # all unique label names
        all_labels = sorted({lbl for lst in df["__labels_list__"] for lbl in lst})
        label_map = {str(lbl): i for i, lbl in enumerate(all_labels)}
        print("Mapeamento de classes multilabel:", label_map)
        # create binary columns
        for lbl_name, idx in label_map.items():
            colname = f"lbl_{idx}_{lbl_name}"
            df[colname] = df["__labels_list__"].apply(lambda lst: 1 if lbl_name in lst else 0)
        # save dataset_multilabel.csv with binary cols
        cols_to_save = ["__image_path_raw__", "__text__"] + [c for c in df.columns if c.startswith("lbl_")]
        out_ml = os.path.join(out_dir, "dataset_multilabel.csv")
        df[cols_to_save].rename(columns={"__image_path_raw__": "image_path", "__text__": "text"}).to_csv(out_ml, index=False)
        print("Salvo multilabel em:", out_ml)
    else:
        # multiclass or no labels
        # if label is numeric keep; if string map to ints
        raw = df["__label_raw__"]
        if raw.dtype == object or raw.dtype.name.startswith("str"):
            # map unique strings to ints, but preserve numeric-like strings
            unique_vals = pd.Series(raw.dropna().astype(str).unique()).tolist()
            # remove empty
            unique_vals = [u for u in unique_vals if u not in ("", "nan", "None")]
            label_map = {str(v): i for i, v in enumerate(sorted(unique_vals, key=lambda x: str(x)))}
            # apply map; missing -> -1
            def map_fn(x):
                if pd.isna(x):
                    return -1
                s = str(x).strip()
                if s == "":
                    return -1
                return label_map.get(s, -1)
            df["__label_int__"] = df["__label_raw__"].apply(map_fn)
            print("Mapeamento de labels (texto->int) salvo em label_map.json")
        else:
            # numeric
            # cast to int (safely)
            try:
                df["__label_int__"] = df["__label_raw__"].astype(int)
                unique_vals = sorted(df["__label_int__"].unique().tolist())
                label_map = {str(v): int(v) for v in unique_vals}
            except Exception:
                # fallback mapping
                unique_vals = pd.Series(raw.dropna().unique()).tolist()
                label_map = {str(v): i for i, v in enumerate(sorted(unique_vals, key=lambda x: str(x)))}
                df["__label_int__"] = df["__label_raw__"].astype(str).apply(lambda s: label_map.get(s, -1))

        # save dataset.csv with image_path, text, label
        out_ds = os.path.join(out_dir, "dataset.csv")
        df[["__image_path_raw__", "__text__", "__label_int__"]].rename(columns={"__image_path_raw__": "image_path", "__text__": "text", "__label_int__": "label"}).to_csv(out_ds, index=False)
        print("Salvo dataset multiclass em:", out_ds)
        # save label map
        with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

    # Create splits
    if not is_multilabel:
        # read dataset saved
        ds = pd.read_csv(os.path.join(out_dir, "dataset.csv"))
        # If all labels are -1 (no labels), skip splitting
        if ds["label"].nunique() <= 1 and (ds["label"].unique()[0] == -1):
            print("Sem rótulos válidos — pulando criação de splits.")
        else:
            # Stratified splits: first train / temp, then val/test split of temp
            stratify_col = ds["label"]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=(1-TRAIN_FRAC), random_state=RANDOM_SEED)
            train_idx, temp_idx = next(sss.split(ds, stratify_col))
            df_train = ds.iloc[train_idx].reset_index(drop=True)
            df_temp = ds.iloc[temp_idx].reset_index(drop=True)

            # split temp into val/test equally
            test_size = TEST_FRAC / (TEST_FRAC + VAL_FRAC)  # proportion of temp that becomes test
            df_val, df_test = train_test_split(df_temp, test_size=test_size, stratify=df_temp["label"], random_state=RANDOM_SEED)

            df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
            df_val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
            df_test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

            print("Splits criados:")
            print("  train:", len(df_train))
            print("  val:  ", len(df_val))
            print("  test: ", len(df_test))
            # show label distribution
            print("\nDistribuição de labels em cada split (value_counts):")
            print("Train:\n", df_train["label"].value_counts().sort_index().to_dict())
            print("Val:\n", df_val["label"].value_counts().sort_index().to_dict())
            print("Test:\n", df_test["label"].value_counts().sort_index().to_dict())
    else:
        # Multilabel splits: attempt to preserve distribution.
        # We'll use a simple heuristic: stratify by number of labels (n_labels per sample) to keep similar label-cardinality distribution.
        df_ml = pd.read_csv(os.path.join(out_dir, "dataset_multilabel.csv"))
        lbl_cols = [c for c in df_ml.columns if c.startswith("lbl_")]
        if len(df_ml) < 10:
            print("Poucas amostras para multilabel splitting robusto; farei split aleatório com seed.")
            df_train, df_temp = train_test_split(df_ml, train_size=TRAIN_FRAC, random_state=RANDOM_SEED, shuffle=True)
            df_val, df_test = train_test_split(df_temp, test_size=TEST_FRAC/(TEST_FRAC+VAL_FRAC), random_state=RANDOM_SEED, shuffle=True)
        else:
            df_ml["__n_labels__"] = df_ml[lbl_cols].sum(axis=1).astype(int)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=(1-TRAIN_FRAC), random_state=RANDOM_SEED)
            train_idx, temp_idx = next(sss.split(df_ml, df_ml["__n_labels__"]))
            df_train = df_ml.iloc[train_idx].reset_index(drop=True)
            df_temp = df_ml.iloc[temp_idx].reset_index(drop=True)
            df_val, df_test = train_test_split(df_temp, test_size=TEST_FRAC/(TEST_FRAC+VAL_FRAC), stratify=df_temp["__n_labels__"], random_state=RANDOM_SEED)
        df_train.to_csv(os.path.join(out_dir, "train_multilabel.csv"), index=False)
        df_val.to_csv(os.path.join(out_dir, "val_multilabel.csv"), index=False)
        df_test.to_csv(os.path.join(out_dir, "test_multilabel.csv"), index=False)
        print("Splits multilabel criados (heurística por número de labels):")
        print("  train:", len(df_train), " val:", len(df_val), " test:", len(df_test))

    print("\nTudo pronto. Arquivos gerados no diretório:", out_dir)
    print("Principais arquivos:")
    for name in ["dataset.csv", "dataset_multilabel.csv", "train.csv", "val.csv", "test.csv", "label_map.json", "missing_images.txt"]:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            print(" -", name)
    print("\nSe quiser, posso adaptar o script para:")
    print(" - concatenar colunas de metadados ao campo 'text'")
    print(" - gerar automaticamente descrições textualizadas a partir de colunas (ex: pH, cor, local)")
    print(" - mover/copiar imagens faltantes para outra pasta")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset multimodal a partir de dados.csv")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="CSV de entrada (padrão: dados.csv)")
    parser.add_argument("--img_root", type=str, default=IMG_ROOT, help="pasta com imagens (padrão: images/)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="onde salvar os outputs")
    args = parser.parse_args()
    main(args)

