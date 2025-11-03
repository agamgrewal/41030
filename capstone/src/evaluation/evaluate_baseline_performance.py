# evaluate_baseline_performance.py
# Agam Grewal – Capstone Project
# Evaluates the BLIP-VQA baseline model (without captions) and generates plots

import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text


def get_baseline_accuracy(predictions, ground_truth):
    correct, total = 0, 0
    for pred in predictions:
        qid = pred["question_id"]
        total += 1
        if clean_text(pred["answer"]) == ground_truth.get(qid, ""):
            correct += 1
    return (correct / total * 100) if total > 0 else 0


def compute_accuracy_by_question_type(preds, gt, qdata):
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})
    qtext = {q["question_id"]: q["question"].lower() for q in qdata}

    for p in preds:
        qid = p["question_id"]
        question = qtext.get(qid, "")
        prefix = question.split(" ")[0] if question else "other"
        grouped[prefix]["total"] += 1
        if clean_text(p["answer"]) == gt.get(qid, ""):
            grouped[prefix]["correct"] += 1

    acc_by_type = {k: (v["correct"] / v["total"]) * 100 for k, v in grouped.items() if v["total"] > 0}
    return acc_by_type


def show_baseline_results(summary):
    print("\n===== Baseline Model Evaluation =====")
    print(f"Questions evaluated: {summary['evaluated']}")
    print(f"Correct answers:     {summary['correct']}")
    print(f"Accuracy:            {summary['accuracy']}%")
    print("=====================================")


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def run_baseline_check():
    pred_path = "results/baseline_predictions.json"
    annotation_path = "src/data/annotations/vqa_train_sample5000_annotations.json"
    question_path = "src/data/annotations/vqa_train_sample5000_questions.json"

    with open(pred_path) as f:
        preds = json.load(f)
    with open(annotation_path) as f:
        data = json.load(f)["annotations"]
    with open(question_path) as f:
        qdata = json.load(f)["questions"]

    gt = {a["question_id"]: clean_text(a["multiple_choice_answer"]) for a in data}

    accuracy = get_baseline_accuracy(preds, gt)
    summary = {
        "evaluated": len(preds),
        "correct": int(accuracy * len(preds) / 100),
        "accuracy": round(accuracy, 2),
    }
    with open("results/baseline_accuracy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Accuracy by question type
    acc_by_type = compute_accuracy_by_question_type(preds, gt, qdata)
    types, accs = zip(*sorted(acc_by_type.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(10, 5))
    plt.bar(types, accs, color="#2980b9", width=0.6)
    plt.title("Baseline BLIP-VQA Accuracy by Question Type", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Question Type")
    plt.xticks(rotation=40, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/baseline_accuracy_by_type.png", dpi=300)
    plt.close()

    show_baseline_results(summary)
    print("Saved bar chart to results/baseline_accuracy_by_type.png")


if __name__ == "__main__":
    run_baseline_check()