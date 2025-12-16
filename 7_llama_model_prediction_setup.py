import os
import re
import sys
import json
import time
import hashlib
from pathlib import Path

import pandas as pd
import torch
import nltk

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

nltk.download("punkt", quiet=True)

# ----------------------------
# Config (edit as needed)
# ----------------------------

INPUT_NOTES_CSV = os.getenv("INPUT_NOTES_CSV", "notes.csv")
CRITERIA_CSV = os.getenv("CRITERIA_CSV", "criteria.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "results.csv")

MODEL_PATH = os.getenv("MODEL_PATH")  # required
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))

if MODEL_PATH is None:
    raise ValueError("MODEL_PATH must be set via environment variable")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def preprocess_notes(df: pd.DataFrame) -> dict:
    """Group discharge notes by patient -> discharge summary -> section."""
    patient_notes = {}
    for patient_id, p_df in df.groupby("patient_id"):
        patient_notes[patient_id] = {}
        for ds_number, ds_df in p_df.groupby("ds_number"):
            patient_notes[patient_id][ds_number] = (
                ds_df.set_index("headline")["content"].to_dict()
            )
    return patient_notes


def infer_gender(patient_notes: dict, patient_id, ds_number: str) -> str:
    """Lightweight heuristic gender inference from HPI."""
    text = patient_notes.get(patient_id, {}).get(ds_number, {}).get(
        "History of Present Illness", ""
    )
    if re.search(r"\b(male|gentleman|man)\b", text, re.I):
        return "male"
    if re.search(r"\b(female|lady|woman)\b", text, re.I):
        return "female"
    return "unknown"


def search_notes(patient_notes, patient_id, ds_number, headlines):
    """Concatenate content from matching section headers."""
    ds_data = patient_notes.get(patient_id, {}).get(ds_number, {})
    content, matched = [], []
    for h in headlines:
        for key, value in ds_data.items():
            if re.fullmatch(re.escape(h), key, flags=re.I):
                content.append(value)
                matched.append(key)
    return " ".join(content).strip(), matched


def cache_path(patient_id, criterion: str) -> Path:
    h = hashlib.md5(criterion.encode()).hexdigest()
    return CACHE_DIR / f"{patient_id}_{h}.json"

# -----------------------------------------------------------------------------
# Prompting
# -----------------------------------------------------------------------------

def build_prompt(
    patient_id,
    ds_number,
    content,
    matched_headlines,
    trial_name,
    criterion,
    is_inclusion,
    gender,
):
    sentences = nltk.sent_tokenize(content)[:30]
    numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))

    criteria_type = "inclusion" if is_inclusion else "exclusion"

    return f"""
You are a medical expert evaluating eligibility for the {trial_name} trial.

Task:
Determine whether the patient meets the following {criteria_type} criterion.

Scoring:
- 1: Included / Not excluded
- 0: Not included / Excluded
- 2: Insufficient information
- 3: Not applicable

Patient ID: {patient_id}
Discharge Summary: {ds_number}
Gender: {gender}
Matched Sections: {', '.join(matched_headlines) if matched_headlines else 'None'}

Note Content (sentence_id: sentence):
{numbered if numbered else 'No relevant content found.'}

Criterion:
{criterion}

Instructions:
- Return ONLY valid JSON with keys: score, reasoning, sentence_ids
- score must be one of: "0", "1", "2", "3"
- reasoning: 3–5 sentences, max 200 characters
- sentence_ids: list of integers

JSON format:
{{"score": "2", "reasoning": "...", "sentence_ids": []}}
"""

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    local_files_only=True,
)
model = accelerator.prepare(model)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.05,
    top_p=0.8,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt_tmpl = ChatPromptTemplate.from_template("{prompt}")
parser = JsonOutputParser()
chain = prompt_tmpl | llm | parser

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_criterion(patient_notes, patient_id, trial, criterion, is_inclusion):
    ds_numbers = sorted(patient_notes.get(patient_id, {}), reverse=True)

    gender = "unknown"
    for ds in ds_numbers:
        gender = infer_gender(patient_notes, patient_id, ds)
        if gender != "unknown":
            break

    # Shortcut rule
    if criterion == "Adult patients":
        return "1", "All patients are adults.", []

    section_map = {
        "Allergies to study products": ["History of Present Illness", "Past Medical History"],
    }

    headlines = section_map.get(
        criterion,
        [
            "History of Present Illness",
            "Past Medical History",
            "Brief Hospital Course",
            "Discharge Diagnosis",
            "Discharge Medications",
            "Past Cardiac Procedures",
        ],
    )

    for ds in ds_numbers:
        content, matched = search_notes(patient_notes, patient_id, ds, headlines)
        if not content:
            continue

        prompt = build_prompt(
            patient_id,
            ds,
            content,
            matched,
            trial,
            criterion,
            is_inclusion,
            gender,
        )

        try:
            response = chain.invoke({"prompt": prompt})
            score = response.get("score", "2")
            reasoning = response.get("reasoning", "")
            sentence_ids = response.get("sentence_ids", [])
            return score, reasoning, sentence_ids
        except Exception:
            continue

    return "2", f"Insufficient data for criterion '{criterion}'.", []

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    notes_df = pd.read_csv(INPUT_NOTES_CSV)
    criteria_df = pd.read_csv(CRITERIA_CSV)

    patient_notes = preprocess_notes(notes_df)

    rows = []
    for patient_id in patient_notes:
        for _, row in criteria_df.iterrows():
            score, reasoning, _ = evaluate_criterion(
                patient_notes,
                patient_id,
                row["Study"],
                row["Criteria"],
                row["Criteria Type"].lower().startswith("inclusion"),
            )

            rows.append(
                {
                    "Patient number": patient_id,
                    "Study": row["Study"],
                    "Criteria": row["Criteria"],
                    "Criteria Type": row["Criteria Type"],
                    "Llama Eligibility": score,
                    "Llama Explanation": reasoning,
                }
            )

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
