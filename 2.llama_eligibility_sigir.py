#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# ============================= CONFIG =============================
DB_PATH = "/data/camll_model/mabdalla_students_data/models/CTPI/extracted_trials/db_export_csv_sigir/eligibility.db"
MODEL_PATH = "/data/camll_model/mabdalla_students_data/models/llama3/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
OFFLOAD_DIR = "/tmp/offload"

OUTPUT_CSV = "results/all_rows_results.csv"
DEBUG_LOG = "results/debug_all_rows.txt"
os.makedirs("results", exist_ok=True)

# ============================= LOAD DATA =============================
print("Loading data from database...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM mapping", conn)
conn.close()

print(f"Loaded {len(df):,} rows")
df.columns = [c.strip().lower() for c in df.columns]
df['trial_id'] = df['trial_id'].astype(str)
df['patient_id'] = df['patient_id'].astype(str)

# ============================= MODEL SETUP =============================
print("\nLoading Llama-3.3-70B-Instruct...")
accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder=OFFLOAD_DIR,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = accelerator.prepare(model)
print("Model loaded successfully\n")

# ============================= PREP =============================
nltk.download('punkt', quiet=True)

def build_prompt(patient_note: str, criterion_text: str, is_inclusion: bool, trial_info: dict) -> str:
    crit_type = "Inclusion Criteria" if is_inclusion else "Exclusion Criteria"
    labels = (
        ['not applicable', 'not enough information', 'included', 'not included']
        if is_inclusion else
        ['not applicable', 'not enough information', 'excluded', 'not excluded']
    )

    prompt = f"""You are a clinical trial eligibility expert. Compare the patient note with the following {crit_type} and determine eligibility for this single criterion.

Reasoning guidelines:
- If the criterion is not applicable to this patient/trial, label 'not applicable'.
- If applicable but no direct evidence in the note:
  - For inclusion: assume met unless contradicted → 'not enough information' only if unclear.
  - For exclusion: assume not met unless explicitly present → 'not enough information' if unclear.
- Use direct evidence when available.

Output format — only a Python dict after "Plain output:":
{{"0": [reasoning, [relevant_sentence_ids], eligibility_label]}}

Labels: {labels}

Patient note (sentence_id: sentence):
{patient_note}

Trial:
NCTID: {trial_info['trial_id']}
Title: {trial_info['trial_title']}
{crit_type}:
{criterion_text}

Plain output:"""
    return prompt

def extract_dict_from_output(text: str) -> dict:
    if "Plain output:" in text:
        text = text.split("Plain output:", 1)[-1].strip()
    try:
        start = text.rfind("{")
        end = text.rfind("}") + 1
        return eval(text[start:end], {"__builtins__": {}})
    except Exception:
        return {}

# ============================= PROCESSING =============================
results = []
with open(DEBUG_LOG, "w", buffering=1) as debug_log:
    for idx, row in df.iterrows():
        patient_id = row['patient_id']
        trial_id = row['trial_id']
        criterion_type = row['criterion_type'].lower()
        is_inclusion = criterion_type == "inclusion"

        note = row['note']
        trial_title = row['trial_title']
        criterion_text = row['criterion_text']

        # Sentence numbering
        sentences = sent_tokenize(note)
        sentences.append("The patient will provide informed consent and comply with the trial protocol.")
        numbered_note = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))

        trial_info = {"trial_id": trial_id, "trial_title": trial_title}
        prompt = build_prompt(numbered_note, criterion_text, is_inclusion, trial_info)

        # Inference
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )[0]

        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)[len(prompt):].strip()

        # Debug logging
        debug_log.write(f"\n=== ROW {idx} | Patient {patient_id} | Trial {trial_id} ===\n")
        debug_log.write(f"PROMPT:\n{prompt}\n\nRAW OUTPUT:\n{raw_output}\n\n")

        parsed = extract_dict_from_output(raw_output)
        criterion_result = parsed.get("0", ["Model failed to return criterion", [], "not enough information"])

        result_row = {
            **row.to_dict(),
            "llama_eligibility": criterion_result[2],
            "llama_sentences": str(criterion_result[1]),
            "llama_explanation": criterion_result[0],
        }
        results.append(result_row)

        print(f"Processed {idx+1}/{len(df)} | Patient {patient_id} | Trial {trial_id} | {criterion_type.capitalize()} | → {criterion_result[2]}")

# ============================= SAVE RESULTS =============================
final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nProcessing complete! Results saved to {OUTPUT_CSV} ({len(final_df)} rows)")