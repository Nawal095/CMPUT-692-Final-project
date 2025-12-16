#!/usr/bin/env python3
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRIALS_DIR = "/data/camll_model/mabdalla_students_data/models/CTPI/extracted_trials"
DATA_DIR = "/data/camll_model/mabdalla_students_data/models/CTPI/extracted_trials/data"
MODEL_PATH = "/data/camll_model/mabdalla_students_data/models/llama3/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbc386b"
SUMMARIES_FILE = os.path.join(DATA_DIR, "patient_summaries.txt")

def load_patient_summaries(file_path):
    logger.info(f"Loading patient summaries from {file_path}")
    try:
        with open(file_path, "r") as f:
            content = f.read()
        logger.info(f"First 100 characters:\n{content[:100]}")
        top_entries = re.split(r'</TOP>\s*', content)
        summaries = []
        for entry in top_entries:
            entry = entry.strip()
            if not entry or '<TOP>' not in entry:
                continue
            num_match = re.search(r'<NUM>(\d+)</NUM>', entry)
            title_match = re.search(r'<TITLE>(.*?)(?:</TITLE>|(?=$|\n))', entry, re.DOTALL)
            if num_match and title_match:
                num = num_match.group(1).strip()
                title = title_match.group(1).strip()
                if title:
                    summaries.append({"num": num, "title": title})
                else:
                    logger.warning(f"Skipping entry with empty title: NUM={num}")
            else:
                logger.warning(f"Skipping malformed entry: {entry[:50]}...")
        logger.info(f"Loaded {len(summaries)} patient summaries")
        return summaries
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return []

def extract_trial_info(xml_file):
    logger.info(f"Parsing trial XML: {xml_file}")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        def get_text(path, default="Unknown"):
            elem = root.find(path)
            return elem.text.strip() if elem is not None and elem.text else default

        interventions = [
            t.find("intervention_name").text.strip()
            for t in root.findall(".//intervention")
            if t.find("intervention_name") is not None and t.find("intervention_name").text
        ]

        criteria_elem = root.find(".//eligibility/criteria/textblock")
        criteria_text = criteria_elem.text.strip() if criteria_elem is not None and criteria_elem.text else "Not provided"

        trial_info = {
            "nct_id": get_text(".//nct_id"),
            "condition": get_text(".//condition"),
            "interventions": interventions,
            "inclusion_criteria": criteria_text,
            "min_age": get_text(".//eligibility/minimum_age", "Not specified")
        }
        return trial_info
    except ET.ParseError as e:
        logger.error(f"XML parse error {xml_file}: {e}")
        return None

def create_prompt(trial_info, patient_summary):
    return f"""
You are a medical expert analyzing patient summaries for relevance to a clinical trial. Below is the trial information and a patient summary. Determine if the patient is likely eligible based on the trial's condition, inclusion criteria, and interventions. Provide a brief explanation (2-3 sentences) and conclude with "Eligible: Yes" or "Eligible: No".

**Clinical Trial Info:**
- Trial ID: {trial_info['nct_id']}
- Condition: {trial_info['condition']}
- Interventions: {', '.join(trial_info['interventions']) if trial_info['interventions'] else 'None'}
- Inclusion Criteria: {trial_info['inclusion_criteria']}
- Minimum Age: {trial_info['min_age']}

**Patient Summary:**
{patient_summary['title']}

**Analysis:**
"""

def initialize_model():
    logger.info(f"Loading Llama-3.3-70B from {MODEL_PATH}")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def query_llama(model, tokenizer, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        return f"Error: {e}"

def detect_relevant_summaries():
    patient_summaries = load_patient_summaries(SUMMARIES_FILE)
    if not patient_summaries:
        logger.error("No patient summaries loaded. Exiting.")
        return None

    model, tokenizer = initialize_model()
    if not model or not tokenizer:
        logger.error("Model initialization failed. Exiting.")
        return None

    results = []
    xml_files = list(Path(TRIALS_DIR).glob("*.xml"))
    logger.info(f"Found {len(xml_files)} trial XML files")

    for xml_file in xml_files:
        trial_info = extract_trial_info(xml_file)
        if not trial_info:
            continue

        logger.info(f"Processing trial {trial_info['nct_id']}")
        trial_results = {"trial_id": trial_info["nct_id"], "summaries": []}

        for summary in patient_summaries:
            prompt = create_prompt(trial_info, summary)
            response = query_llama(model, tokenizer, prompt)

            eligibility = "Unknown"
            if "Eligible: Yes" in response:
                eligibility = "Yes"
            elif "Eligible: No" in response:
                eligibility = "No"

            trial_results["summaries"].append({
                "trial_id": trial_info["nct_id"],
                "summary_num": summary["num"],
                "summary_title": summary["title"],
                "eligibility": eligibility,
                "model_response": response
            })

        results.extend(trial_results["summaries"])

    output_path = os.path.join(DATA_DIR, "relevance_results.csv")
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")

    return results

if __name__ == "__main__":
    logger.info("Starting relevance detection")
    results = detect_relevant_summaries()
    if results:
        for r in results[:20]:
            print(f"\nTrial: {r['trial_id']} | Summary {r['summary_num']}")
            print(f"Eligibility: {r['eligibility']}")
            print(f"Response: {r['model_response'][:200]}...")
            print("-" * 80)
    logger.info("Processing complete")