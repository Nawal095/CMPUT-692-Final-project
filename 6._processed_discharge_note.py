#!/usr/bin/env python3
import re
import json
import os
import torch
import nltk
import hashlib
import time
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from accelerate import Accelerator
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline

# ============================= SETUP =============================
nltk.download('punkt', quiet=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

INPUT_CSV = Path("/data/camll_model/mabdalla_students_data/models/llama3/MIMIC_2nd_batch_patient/output/ot_new.csv")
OUTPUT_CSV = Path("/data/camll_model/mabdalla_students_data/models/llama3/MIMIC_2nd_batch_patient/output/exp_3nd_batch_patient_result.csv")
MODEL_PATH = "/data/camll_model/mabdalla_students_data/models/llama3/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
CACHE_DIR = Path("/data/camll_model/mabdalla_students_data/cache")
CACHE_DIR.mkdir(exist_ok=True)

# ============================= MODEL LOADING =============================
print("Loading Llama-3.3-70B-Instruct...")
accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
quant_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quant_config,
    local_files_only=True
)
model = accelerator.prepare(model)

hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    return_full_text=False,
    temperature=0.05,
    top_p=0.8,
    do_sample=False,
    truncation=True
)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# ============================= DATA PREP =============================
df = pd.read_csv(INPUT_CSV, encoding='latin1')
df = df.dropna(subset=["patient_id"])
df["patient_id"] = df["patient_id"].astype(int)
df["headline"] = df["headline"].astype(str)
df["content"] = df["content"].fillna("").astype(str)

def build_patient_notes(df: pd.DataFrame) -> dict:
    notes = {}
    for patient_id, p_group in df.groupby("patient_id"):
        notes[patient_id] = {}
        for ds_num, ds_group in p_group.groupby("ds_number"):
            notes[patient_id][ds_num] = ds_group.set_index("headline")["content"].to_dict()
    return notes

patient_notes = build_patient_notes(df)
print(f"Preprocessed notes for {len(patient_notes)} patients")

# ============================= CRITERIA =============================
CRITERIA = [
    {"study": "CONVERGE-HF", "text": "Adult patients", "type": "Inclusion criteria"},
    {"study": "CONVERGE-HF", "text": "Established chronic heart failure (>= 6 months)", "type": "Inclusion criteria"},
    {"study": "CONVERGE-HF", "text": "Mild-to-moderate cognitive impairment (as per the diagnosis of cognitive impairment or a Montreal Cognitive Assessment (MoCA) score 10-25)", "type": "Inclusion criteria"},
    {"study": "CONVERGE-HF", "text": "Patients who have contraindications for sGC stimulator and vericiguat therapy (i.e. use of long-acting nitrates, other soluble guanylate cyclase stimulators (e.g., riociguat), or phosphodiesterase type 5 (PDE-5), severe anemia, pregnancy or breast-feeding)", "type": "Exclusion criteria"},
    {"study": "CONVERGE-HF", "text": "Unable to undergo CMR imaging or brain MRI", "type": "Exclusion criteria"},
    {"study": "CONVERGE-HF", "text": "CMR exclusions: renal failure [a glomerular filtration rate <30 mL/min)], implantable cardiac device (ICD, PPM or CRT), uncontrolled atrial fibrillation or recurrent ventricular arrhythmias)", "type": "Exclusion criteria"},
    {"study": "CONVERGE-HF", "text": "General medical conditions: uncontrolled thyroid disorders, hepatic failure, or myocardial revascularization procedures [coronary angioplasty and/or surgical revascularization in the previous 3 months], cancer/malignancy, or with moderate-severe dementia)", "type": "Exclusion criteria"},
    {"study": "CONVERGE-HF", "text": "Allergies to study products", "type": "Exclusion criteria"},
]

# Headline mapping for each criterion
HEADLINE_MAP = {
    "Adult patients": [],
    "Established chronic heart failure (>= 6 months)": ["History of Present Illness", "Past Medical History", "Discharge Diagnosis"],
    "Mild-to-moderate cognitive impairment (as per the diagnosis of cognitive impairment or a Montreal Cognitive Assessment (MoCA) score 10-25)": ["Brief Hospital Course", "Past Medical History", "Discharge Diagnosis"],
    "Patients who have contraindications for sGC stimulator and vericiguat therapy (i.e. use of long-acting nitrates, other soluble guanylate cyclase stimulators (e.g., riociguat), or phosphodiesterase type 5 (PDE-5), severe anemia, pregnancy or breast-feeding)": ["Discharge Medications", "Brief Hospital Course", "Discharge Diagnosis"],
    "Unable to undergo CMR imaging or brain MRI": ["Past Cardiac Procedures", "Brief Hospital Course", "Discharge Diagnosis", "Past Medical History"],
    "CMR exclusions: renal failure [a glomerular filtration rate <30 mL/min)], implantable cardiac device (ICD, PPM or CRT), uncontrolled atrial fibrillation or recurrent ventricular arrhythmias)": ["Past Cardiac Procedures", "Brief Hospital Course", "Discharge Diagnosis", "Past Medical History"],
    "General medical conditions: uncontrolled thyroid disorders, hepatic failure, or myocardial revascularization procedures [coronary angioplasty and/or surgical revascularization in the previous 3 months], cancer/malignancy, or with moderate-severe dementia)": ["Brief Hospital Course", "Past Medical History", "Discharge Diagnosis"],
    "Allergies to study products": ["History of Present Illness", "Past Medical History"],
}

# ============================= HELPERS =============================
def get_gender(patient_id: int, notes: dict) -> str:
    ds_numbers = sorted(notes.get(patient_id, {}).keys(), reverse=True)
    for ds in ds_numbers:
        history = notes[patient_id][ds].get("History of Present Illness", "")
        if re.search(r'\b(male|gentleman|man)\b', history, re.IGNORECASE):
            return "male"
        if re.search(r'\b(female|lady|woman)\b', history, re.IGNORECASE):
            return "female"
    return "unknown"

def extract_relevant_text(patient_id: int, ds_number: str, headlines: list) -> tuple[str, list]:
    ds_data = patient_notes.get(patient_id, {}).get(ds_number, {})
    content_parts = []
    matched = []
    for headline in headlines:
        for key, text in ds_data.items():
            if re.match(rf'^{re.escape(headline)}$', key, re.IGNORECASE):
                if text.strip():
                    content_parts.append(text.strip())
                    matched.append(key)
    text = " ".join(content_parts)
    return text if text else "No relevant content found.", matched

def build_prompt(patient_id: int, ds_number: str, text: str, matched: list, criterion: str, is_inclusion: bool, gender: str) -> str:
    sentences = nltk.sent_tokenize(text)[:30]
    numbered = [f"{i}: {s.strip()}" for i, s in enumerate(sentences, 1) if s.strip()]
    note_text = "\n".join(numbered) if numbered else "No relevant content."

    crit_type = "inclusion" if is_inclusion else "exclusion"

    return f"""
You are a medical expert evaluating eligibility for the CONVERGE-HF trial.
Determine if the patient meets this {crit_type} criterion using the note below.

Scoring:
- 1: Meets inclusion / not excluded
- 0: Does not meet inclusion / excluded
- 2: Insufficient information
- 3: Not applicable (e.g., pregnancy for male)

Patient: {patient_id}-DS-{ds_number} | Gender: {gender}
Matched sections: {', '.join(matched) or 'None'}

Note (sentence_id: sentence):
{note_text}

Criterion: {criterion}

Return ONLY valid JSON:
{{
  "score": "2",
  "reasoning": "Brief explanation (3-5 sentences, <500 chars) referencing evidence and relevance.",
  "sentence_ids": []
}}
"""

chain = ChatPromptTemplate.from_template("{prompt}") | llm | JsonOutputParser()

def get_cache_path(patient_id: int, criterion: str) -> Path:
    hash_val = hashlib.md5(criterion.encode()).hexdigest()
    return CACHE_DIR / f"{patient_id}_{hash_val}.json"

def evaluate_criterion(patient_id: int, criterion: str, is_inclusion: bool) -> tuple[str, str, list]:
    cache_path = get_cache_path(patient_id, criterion)
    # Caching disabled for consistency with original
    # if cache_path.exists():
    #     data = json.loads(cache_path.read_text())
    #     return data["score"], data["reasoning"], data["sentence_ids"]

    # Special fast path
    if "Adult patients" in criterion:
        result = {"score": "1", "reasoning": "All patients are adults. Criterion met.", "sentence_ids": []}
        cache_path.write_text(json.dumps(result))
        return "1", result["reasoning"], []

    gender = get_gender(patient_id, patient_notes)
    headlines = HEADLINE_MAP.get(criterion, ["History of Present Illness", "Past Medical History", "Brief Hospital Course", "Discharge Diagnosis"])
    ds_numbers = sorted(patient_notes.get(patient_id, {}).keys(), reverse=True)
    ds_numbers = [ds_numbers[0]] if "within" in criterion.lower() else ds_numbers

    for ds in ds_numbers:
        text, matched = extract_relevant_text(patient_id, ds, headlines)
        if text == "No relevant content found.":
            continue

        prompt = build_prompt(patient_id, ds, text, matched, criterion, is_inclusion, gender)
        try:
            raw = chain.invoke({"prompt": prompt})
            score = str(raw.get("score", "2"))
            reasoning = raw.get("reasoning", "No reasoning provided.")[:500]
            sentence_ids = raw.get("sentence_ids", [])
            if score not in {"0", "1", "2", "3"}:
                score = "2"
                reasoning = "Invalid score from model."

            result = {"score": score, "reasoning": reasoning, "sentence_ids": sentence_ids}
            cache_path.write_text(json.dumps(result))
            return score, reasoning, sentence_ids
        except Exception as e:
            print(f"LLM error for {patient_id} | {criterion}: {e}")

    # Fallback
    fallback = {"score": "2", "reasoning": f"Insufficient information for criterion '{criterion}'.", "sentence_ids": []}
    cache_path.write_text(json.dumps(fallback))
    return "2", fallback["reasoning"], []

# ============================= MAIN =============================
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(columns=["Patient number", "Study", "Criteria", "Criteria Type", "Llama Eligibility", "Llama Explanation"]).to_csv(OUTPUT_CSV, index=False)

rows = []
for patient_id in patient_notes:
    for crit in CRITERIA:
        score, reasoning, _ = evaluate_criterion(patient_id, crit["text"], crit["type"] == "Inclusion criteria")
        rows.append({
            "Patient number": patient_id,
            "Study": crit["study"],
            "Criteria": crit["text"],
            "Criteria Type": crit["type"],
            "Llama Eligibility": score,
            "Llama Explanation": reasoning
        })

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
    print(f"Processed patient {patient_id} → {len(rows)} total rows")

print(f"\nDone! Results saved to {OUTPUT_CSV} ({len(rows)} rows)")