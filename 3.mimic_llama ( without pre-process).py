#!/usr/bin/env python3
import os
import re
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# ============================= CONFIG =============================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

MODEL_PATH = "/data/camll_model/mabdalla_students_data/models/llama3/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
DATA_DIR = "/data/camll_model/mabdalla_students_data/models/llama3/patient_info/"
OUTPUT_FILE = "sigir_prompt_mimic_results_20k.csv"
TEMP_OUTPUT = "sigir_partial_20k.csv"
TEMP_SAVE_EVERY = 5

MAX_PROMPT_TOKENS = 15000
GENERATION_KWARGS = {
    "max_new_tokens": 128,
    "temperature": 0.3,
    "do_sample": True,
}

# ============================= MODEL SETUP =============================
accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)
model = accelerator.prepare(model)

# ============================= HELPERS =============================
def normalize(text: str) -> str:
    return text.replace("1i", "1").lower() if isinstance(text, str) else ""

def rule_based_decision(note: str, criterion: str, crit_type: str) -> dict | None:
    note_n = normalize(note)
    crit_n = criterion.lower()

    if "adult patients" in crit_n:
        return {"score": "1", "reasoning": "All patients assumed adults; criterion met.", "sentence_ids": []}

    if any(term in crit_n for term in ["chronic heart failure", "≥ 6 months", ">= 6 months"]):
        hf_terms = ["heart failure", "chf", "congestive heart failure", "hf", "hfre", "hfr ef", "hfref", "hfpef"]
        if not any(t in note_n for t in hf_terms):
            return {"score": "0", "reasoning": "No evidence of chronic heart failure.", "sentence_ids": []}

        match = re.search(r"(\d+)\s*months?", note_n)
        if match and int(match.group(1)) >= 6:
            return {"score": "1", "reasoning": "Heart failure duration ≥ 6 months.", "sentence_ids": []}
        if match:
            return {"score": "0", "reasoning": "Heart failure duration < 6 months.", "sentence_ids": []}
        return {"score": "2", "reasoning": "Heart failure present but duration unknown.", "sentence_ids": []}

    return None

def count_tokens(text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())

def trim_to_tokens(text: str, max_tokens: int) -> str:
    """Keep the most recent (last) tokens — typically more clinically relevant."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    trimmed = tokens[-max_tokens:]
    return tokenizer.decode(trimmed, clean_up_tokenization_spaces=True)

def build_prompt(patient_text: str, criterion: str, crit_type: str) -> str:
    display_type = "inclusion criteria" if "inclusion" in crit_type.lower() else "exclusion criteria"

    header = (
        f"You are an expert physician matching patients to clinical trials. "
        f"Evaluate the patient note against the following {display_type}.\n\n"
        "Reason step-by-step:\n"
        "1. Is the criterion applicable?\n"
        "2. Is there direct evidence?\n"
        "3. If unclear, label as 'not enough information'.\n\n"
        "Return ONLY a JSON object with keys: \"score\", \"reasoning\", \"sentence_ids\".\n"
        "score must be \"0\", \"1\", \"2\", or \"3\".\n\n"
    )
    footer = f"\nCriterion: {criterion}\n\nJSON:"

    header_tokens = count_tokens(header + footer)
    safety_buffer = 256
    max_patient_tokens = max(MAX_PROMPT_TOKENS - header_tokens - safety_buffer, 128)

    patient_trimmed = trim_to_tokens(patient_text, max_patient_tokens)
    prompt = header + "Patient note (sentence_id: sentence):\n" + patient_trimmed + footer

    # Final safety trim
    if count_tokens(prompt) > MAX_PROMPT_TOKENS:
        patient_trimmed = trim_to_tokens(patient_trimmed, max_patient_tokens // 2)
        prompt = header + "Patient note (sentence_id: sentence):\n" + patient_trimmed + footer

    return prompt

def run_llm(prompt: str) -> dict:
    if count_tokens(prompt) > MAX_PROMPT_TOKENS:
        return {
            "score": "2",
            "reasoning": f"Not enough information: prompt exceeded token limit ({count_tokens(prompt)} > {MAX_PROMPT_TOKENS}).",
            "sentence_ids": []
        }

    try:
        torch.cuda.empty_cache()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, **GENERATION_KWARGS)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = text[len(prompt):].strip()

        # Extract JSON
        match = re.search(r"\{.*\}", generated, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {"score": "2", "reasoning": "Not enough information: invalid or missing JSON.", "sentence_ids": []}

        result.setdefault("score", "2")
        result.setdefault("reasoning", "Not enough information.")
        result.setdefault("sentence_ids", [])

        if str(result["score"]) == "2" and "not enough information" not in result["reasoning"].lower():
            result["reasoning"] = "Not enough information: " + result["reasoning"]

        return result

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"score": "2", "reasoning": "Not enough information: CUDA OOM.", "sentence_ids": []}
    except Exception:
        torch.cuda.empty_cache()
        return {"score": "2", "reasoning": "Not enough information: generation error.", "sentence_ids": []}

def map_to_numeric(label: str, crit_type: str) -> int:
    label = str(label).lower().strip()
    if label in {"2", "not enough information"}:
        return 2
    if "inclusion" in crit_type.lower():
        return 1 if label in {"1", "included"} else 0
    else:  # exclusion
        return 0 if label in {"0", "excluded"} else 1

# ============================= TRIAL CRITERIA =============================
trial_criteria = {
    "CONVERGE-HF": {
        "inclusion": [
            "Adult patients",
            "Established chronic heart failure (≥ 6 months)",
            "Mild-to-moderate cognitive impairment (MoCA score 10–25)"
        ],
        "exclusion": [
            "Patients who have contraindications for sGC stimulator or vericiguat therapy",
            "Unable to undergo CMR imaging or brain MRI",
            "CMR exclusions: renal failure (GFR <30), ICD/CRT/PPM, uncontrolled AF or recurrent VT",
            "General medical conditions: thyroid disorders, hepatic failure, recent revascularization, cancer, moderate–severe dementia",
            "Allergies to study products"
        ]
    }
}

# ============================= RESUME SUPPORT =============================
processed_keys = set()
rows = []

if os.path.exists(TEMP_OUTPUT):
    print(f"[RESUME] Loading partial results from {TEMP_OUTPUT}")
    partial_df = pd.read_csv(TEMP_OUTPUT, dtype=str)
    rows = partial_df.values.tolist()
    processed_keys = {f"{r[0]}|{r[2]}|{r[3]}" for r in rows}

# ============================= MAIN LOOP =============================
patient_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
print(f"Found {len(patient_files)} patient files")

columns = ["Patient number", "Study", "Criteria", "Criteria Type", "LLAMA Label", "LLAMA Explanation"]

for i, filename in enumerate(patient_files, 1):
    patient_id = filename.removesuffix(".txt")
    note_path = os.path.join(DATA_DIR, filename)

    try:
        with open(note_path, "r", encoding="utf-8") as f:
            patient_note = f.read()
    except Exception as e:
        print(f"[SKIP] Failed to read {filename}: {e}")
        continue

    # Sentence numbering
    sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', patient_note) if s.strip()]
    formatted_note = "\n".join(f"{idx}: {s}" for idx, s in enumerate(sentences, start=1))

    for study, criteria in trial_criteria.items():
        for crit_type, crit_list in [("Inclusion criteria", criteria["inclusion"]),
                                     ("Exclusion criteria", criteria["exclusion"])]:
            for criterion in crit_list:
                key = f"{patient_id}|{criterion}|{crit_type}"
                if key in processed_keys:
                    continue

                rule_result = rule_based_decision(patient_note, criterion, crit_type)
                if rule_result is None:
                    prompt = build_prompt(formatted_note, criterion, crit_type)
                    result = run_llm(prompt)
                else:
                    result = rule_result

                numeric_label = map_to_numeric(result["score"], crit_type)
                rows.append([patient_id, study, criterion, crit_type, numeric_label, result["reasoning"]])
                processed_keys.add(key)

                if len(rows) % TEMP_SAVE_EVERY == 0:
                    pd.DataFrame(rows, columns=columns).to_csv(TEMP_OUTPUT, index=False, encoding="utf-8")
                    torch.cuda.empty_cache()

    # Save after each patient
    pd.DataFrame(rows, columns=columns).to_csv(TEMP_OUTPUT, index=False, encoding="utf-8")
    print(f"Processed {i}/{len(patient_files)} patients → {len(rows)} total evaluations")

# ============================= FINAL SAVE =============================
pd.DataFrame(rows, columns=columns).to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
if os.path.exists(TEMP_OUTPUT):
    os.remove(TEMP_OUTPUT)

print(f"\nDone! Final results saved to {OUTPUT_FILE} ({len(rows)} rows)")