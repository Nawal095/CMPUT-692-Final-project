#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import sqlite3
import re
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ====================== CONFIG ======================
XML_DIR = "/data/camll_model/mabdalla_students_data/models/CTPI/extracted_trials/labeled_trials_data"
DB_PATH = "eligibility.db"
CSV_OUT = "db_export_csv/trials_criteria_final.csv"
BATCH_SIZE = 100

Path("db_export_csv").mkdir(exist_ok=True)

# ====================== SAFE TEXT EXTRACTION ======================
def get_text_safe(elem):
    return (elem.text or "").strip() if elem is not None else ""

# ====================== CRITERIA PARSER ======================
def parse_criteria_block(textblock):
    if not textblock or not textblock.strip():
        return [], []

    text = re.sub(r'\s+', ' ', textblock).strip()
    if not text:
        return [], []

    inclusion = []
    exclusion = []
    current_type = None
    current_num = None
    current_text = []

    def flush():
        nonlocal current_num, current_text
        if current_num is not None and current_text:
            txt = " ".join(current_text).strip()
            if txt:
                if current_type == "inclusion":
                    inclusion.append((current_num, txt))
                elif current_type == "exclusion":
                    exclusion.append((current_num, txt))
        current_num = None
        current_text = []

    parts = re.split(r'(Inclusion\s*Criteria\s*[:\-]?\s*|Exclusion\s*Criteria\s*[:\-]?\s*|[-•*]?\s*([0-9a-zA-Z][\.\)])\s*)', text, flags=re.IGNORECASE)
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue

        if re.match(r'^Inclusion\s*Criteria', part, re.I):
            flush()
            current_type = "inclusion"
            i += 1
            continue
        if re.match(r'^Exclusion\s*Criteria', part, re.I):
            flush()
            current_type = "exclusion"
            i += 1
            continue

        if i + 1 < len(parts):
            num_part = parts[i + 1].strip()
            if re.match(r'^[0-9a-zA-Z][\.\)]$', num_part):
                flush()
                current_num = num_part.strip(".)")
                i += 2
                if i < len(parts):
                    current_text = [parts[i].strip()]
                i += 1
                continue

        if current_text is not None:
            current_text.append(part)
        i += 1

    flush()
    return inclusion, exclusion

# ====================== XML PARSING ======================
def parse_single_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        trial_id = get_text_safe(root.find('id_info/nct_id'))
        if not trial_id:
            return None

        trial_title = get_text_safe(root.find('official_title')) or get_text_safe(root.find('brief_title')) or ""

        textblock_elem = root.find('eligibility/criteria/textblock')
        textblock = get_text_safe(textblock_elem)

        inclusion, exclusion = parse_criteria_block(textblock)

        criteria = [
            (trial_id, 'inclusion', num or "", txt) for num, txt in inclusion
        ] + [
            (trial_id, 'exclusion', num or "", txt) for num, txt in exclusion
        ]

        return {
            "trial": (trial_id, trial_title),
            "criteria": criteria
        }
    except Exception as e:
        print(f"ERROR {xml_path}: {e}")
        return None

# ====================== MAIN ======================
def main():
    xml_files = glob.glob(f"{XML_DIR}/*.xml")
    print(f"Found {len(xml_files)} XML files")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executescript("""
        PRAGMA foreign_keys = ON;
        DROP TABLE IF EXISTS trial_criterion;
        DROP TABLE IF EXISTS trial;

        CREATE TABLE trial (
            trial_id TEXT PRIMARY KEY,
            trial_title TEXT
        );

        CREATE TABLE trial_criterion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id TEXT,
            criterion_type TEXT,
            criterion_number TEXT,
            criterion_text TEXT,
            FOREIGN KEY (trial_id) REFERENCES trial(trial_id)
        );
    """)

    batch = []
    all_rows = []

    for xml_file in tqdm(xml_files, desc="Parsing"):
        data = parse_single_xml(xml_file)
        if not data:
            continue

        batch.append(data)
        trial_id, trial_title = data["trial"]

        for crit in data["criteria"]:
            all_rows.append({
                "trial_id": trial_id,
                "trial_title": trial_title,
                "criterion_type": crit[1],
                "criterion_number": crit[2] if crit[2] else None,
                "criterion_text": crit[3]
            })

        if len(batch) >= BATCH_SIZE:
            for item in batch:
                c.execute("INSERT OR REPLACE INTO trial VALUES (?, ?)", item["trial"])
                if item["criteria"]:
                    c.executemany(
                        "INSERT INTO trial_criterion (trial_id, criterion_type, criterion_number, criterion_text) VALUES (?, ?, ?, ?)",
                        item["criteria"]
                    )
            conn.commit()
            batch = []

    if batch:
        for item in batch:
            c.execute("INSERT OR REPLACE INTO trial VALUES (?, ?)", item["trial"])
            if item["criteria"]:
                c.executemany(
                    "INSERT INTO trial_criterion (trial_id, criterion_type, criterion_number, criterion_text) VALUES (?, ?, ?, ?)",
                    item["criteria"]
                )
        conn.commit()

    conn.close()

    if not all_rows:
        print("No criteria found. Creating empty CSV.")
        df = pd.DataFrame(columns=["trial_id", "trial_title", "criterion_type", "criterion_number", "criterion_text"])
    else:
        df = pd.DataFrame(all_rows)
        df = df[["trial_id", "trial_title", "criterion_type", "criterion_number", "criterion_text"]]

    df.to_csv(CSV_OUT, index=False, encoding='utf-8')
    print(f"\nFINAL CSV: {CSV_OUT} → {len(df)} rows")

    if 'NCT02264392' in df['trial_id'].values:
        print("\nSAMPLE (NCT02264392):")
        print(df[df['trial_id'] == 'NCT02264392'].head(10).to_string(index=False))

    if 'NCT02284984' in df['trial_id'].values:
        print("\nSAMPLE (NCT02284984):")
        print(df[df['trial_id'] == 'NCT02284984'].head(5).to_string(index=False))

if __name__ == "__main__":
    main()