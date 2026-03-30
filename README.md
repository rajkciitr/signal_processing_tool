# signal_processing_tool
Codes to learn basic application of signal (speech, image etc.) processing techniques for real applications

Useful links:
https://github.com/SuperKogito/spafe

# Resnet

# source for Android app running onnxruntime-genai model
https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/README.md

# Code for onnx-runtime-genai benchmarking
import time
import json
import numpy as np

import onnxruntime_genai as og

from datasets import load_dataset
from tqdm import tqdm

# =========================
# CONFIG
# =========================

MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/apollo2-onnx/onnx-out/'

EVAL_SAMPLES = 500
MAX_LENGTH = 512

OUTPUT_FILE = "onnx_benchmark_results.json"

# =========================
# LOAD MODEL
# =========================

print("Loading ONNX model...")

model = og.Model(MODEL_PATH)

tokenizer = og.Tokenizer(model)

# =========================
# PROMPT FORMAT
# =========================

def format_medmcqa(sample):

    options = [
        sample["opa"],
        sample["opb"],
        sample["opc"],
        sample["opd"]
    ]

    prompt = f"""
Question:
{sample['question']}

Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Answer:
"""

    return prompt


def extract_letter(text):

    for letter in ["A", "B", "C", "D"]:
        if letter in text:
            return letter

    return "A"

# =========================
# GENERATION FUNCTION
# =========================

def generate_answer(prompt):

    tokens = tokenizer.encode(prompt)
    input_length = len(tokens)

    # Create fresh params per request
    params = og.GeneratorParams(model)

    params.set_search_options(
        max_length=input_length + MAX_LENGTH,
        temperature=0.0,
        do_sample=False
    )

    generator = og.Generator(model, params)


    start = time.time()
    generator.append_tokens(tokens)

    while not generator.is_done():
            generator.generate_next_token()

    end = time.time()

    output_tokens = generator.get_sequence(0)

    text = tokenizer.decode(output_tokens)

    answer = text[len(prompt):].strip()

    latency = end - start

    token_count = len(output_tokens)

    return answer, latency, token_count


# =========================
# MEDMCQA BENCHMARK
# =========================

print("\nRunning MedMCQA...")

dataset = load_dataset(
    "medmcqa",
    split="validation"
)

correct = 0

latencies = []
tokens_total = 0

for sample in tqdm(dataset.select(range(EVAL_SAMPLES))):

    prompt = format_medmcqa(sample)

    answer_text, latency, tokens = generate_answer(prompt)

    pred_letter = extract_letter(answer_text)

    correct_idx = sample["cop"]

    correct_letter = ["A","B","C","D"][correct_idx]

    if pred_letter == correct_letter:
        correct += 1

    latencies.append(latency)
    tokens_total += tokens


medmcqa_accuracy = correct / EVAL_SAMPLES

# =========================
# PUBMEDQA BENCHMARK
# =========================

print("\nRunning PubMedQA...")

pubmedqa = load_dataset(
    "pubmed_qa",
    "pqa_labeled",
    split="train"
)

correct = 0

for sample in tqdm(pubmedqa.select(range(EVAL_SAMPLES))):

    context = " ".join(
        sample["context"]["contexts"]
    )

    question = sample["question"]

    prompt = f"""
Context:
{context}

Question:
{question}

Answer yes, no, or maybe.

Answer:
"""

    answer_text, latency, tokens = generate_answer(prompt)

    prediction = answer_text.lower()

    gold = sample["final_decision"].lower()

    if gold in prediction:
        correct += 1


pubmedqa_accuracy = correct / EVAL_SAMPLES

# =========================
# PERFORMANCE METRICS
# =========================

avg_latency = np.mean(latencies)

tokens_per_sec = tokens_total / np.sum(latencies)

# =========================
# SAVE RESULTS
# =========================

results = {

    "model_type": "onnxruntime-genai",

    "model_path": MODEL_PATH,

    "medmcqa_accuracy": medmcqa_accuracy,

    "pubmedqa_accuracy": pubmedqa_accuracy,

    "avg_latency_sec": float(avg_latency),

    "tokens_per_sec": float(tokens_per_sec)
}

with open(OUTPUT_FILE, "w") as f:

    json.dump(results, f, indent=4)

print("\nFinal Results:")

print(json.dumps(results, indent=4))
