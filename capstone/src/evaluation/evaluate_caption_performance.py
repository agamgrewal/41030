# ================== IMPORTS ==================
import json, time, psutil, GPUtil, matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from bert_score import score as bertscore

# ================== CONFIG ==================
RUN_TYPE = "caption"
PRED_FILE = Path(f"results/{RUN_TYPE}_predictions.json")
ANNOTATIONS_FILE = Path("src/data/annotations/vqa_train_sample5000_annotations.json")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ================== HELPERS ==================
def compute_vqa_soft_accuracy(pred, gt_answers):
    pred = pred.strip().lower()
    match_count = sum(pred == ans for ans in gt_answers)
    return min(1.0, match_count / 3.0)

# ================== LOAD DATA ==================
start_time = time.time()
with open(PRED_FILE, "r") as f:
    predictions = json.load(f)
with open(ANNOTATIONS_FILE, "r") as f:
    annotations = json.load(f)["annotations"]
gt_map = {a["question_id"]: [ans["answer"].lower() for ans in a["answers"]] for a in annotations}
qtype_map = {a["question_id"]: a.get("question_type", "Other") for a in annotations}

# ================== EVALUATION ==================
total = len(predictions)
strict_correct = 0
soft_sum = 0.0
refs, hyps = [], []
type_counter, correct_counter = Counter(), Counter()
for p in predictions:
    qid = p.get("question_id")
    pred = p.get("answer", "").strip().lower()
    gt_answers = gt_map.get(qid, [])
    q_type = qtype_map.get(qid, "Other")
    if any(pred == ans for ans in gt_answers):
        strict_correct += 1
        correct_counter[q_type] += 1
    acc = compute_vqa_soft_accuracy(pred, gt_answers)
    soft_sum += acc
    type_counter[q_type] += 1
    if gt_answers:
        refs.append(gt_answers[0])
        hyps.append(pred)
    else:
        refs.append("")
        hyps.append(pred)
strict_acc = (strict_correct / total) * 100 if total > 0 else 0
soft_acc = (soft_sum / total) * 100 if total > 0 else 0

# ================== BERTSCORE ==================
P, R, F1 = bertscore(hyps, refs, lang="en", rescale_with_baseline=True)
bert_mean = float(F1.mean() * 100)

# ================== SYSTEM METRICS ==================
elapsed = time.time() - start_time
avg_inference_time = (elapsed / total) if total > 0 else 0
throughput = total / elapsed if elapsed > 0 else 0
gpus = GPUtil.getGPUs()
gpu_util = gpus[0].load * 100 if gpus else 0
gpu_mem = gpus[0].memoryUsed if gpus else 0
cpu_util = psutil.cpu_percent(interval=1)

# ================== PRINT ==================
print(f"\n=== {RUN_TYPE.upper()} PERFORMANCE METRICS ===")
print(f"Total predictions: {total:,}")
print(f"Strict accuracy: {strict_acc:.2f}%")
print(f"VQA Soft accuracy: {soft_acc:.2f}%")
print(f"BERTScore (semantic similarity): {bert_mean:.2f}%")
print(f"Average evaluation time per sample: {avg_inference_time:.4f} sec")
print(f"Throughput: {throughput:.2f} samples/sec")
print(f"GPU utilisation: {gpu_util:.1f}% | Memory used: {gpu_mem:.1f} MB")
print(f"CPU utilisation: {cpu_util:.1f}%")

# ================== VISUALS ==================
types = list(type_counter.keys())
type_acc = [100 * (correct_counter[t] / type_counter[t]) if type_counter[t] else 0 for t in types]
plt.figure(figsize=(7,4))
plt.bar(types, type_acc, color=("#4c72b0" if RUN_TYPE=="baseline" else "#f28e2b"))
plt.xlabel("Question Type")
plt.ylabel("Accuracy (%)")
plt.title(f"{RUN_TYPE.capitalize()} BLIP-VQA Accuracy by Question Type")
plt.tight_layout()
plt.savefig(OUT_DIR / f"{RUN_TYPE}_accuracy_by_type.png", dpi=300)
plt.close()

summary_data = [
    ["Total Predictions", f"{total:,}"],
    ["Strict Accuracy (%)", f"{strict_acc:.2f}"],
    ["VQA Soft Accuracy (%)", f"{soft_acc:.2f}"],
    ["BERTScore (F1 %)", f"{bert_mean:.2f}"],
    ["Avg Inference Time (s)", f"{avg_inference_time:.4f}"],
    ["Throughput (samples/s)", f"{throughput:.2f}"],
    ["GPU Utilisation (%)", f"{gpu_util:.1f}"],
    ["GPU Memory (MB)", f"{gpu_mem:.1f}"],
    ["CPU Utilisation (%)", f"{cpu_util:.1f}"]
]
fig, ax = plt.subplots(figsize=(6,3))
ax.axis("off")
table = ax.table(cellText=summary_data, colLabels=["Metric","Value"], loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1,1.2)
plt.title(f"{RUN_TYPE.capitalize()} BLIP-VQA System and Accuracy Summary", fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(OUT_DIR / f"{RUN_TYPE}_system_metrics.png", dpi=300)
plt.close()

# ================== SAVE JSON ==================
summary = {
    "run_type": RUN_TYPE,
    "total_predictions": total,
    "strict_accuracy": round(strict_acc,2),
    "soft_accuracy": round(soft_acc,2),
    "bertscore_F1": round(bert_mean,2),
    "avg_inference_time_sec": round(avg_inference_time,4),
    "throughput_per_sec": round(throughput,2),
    "gpu_utilisation_pct": round(gpu_util,1),
    "gpu_memory_used_mb": round(gpu_mem,1),
    "cpu_utilisation_pct": round(cpu_util,1),
    "accuracy_by_type": {t: round(a,2) for t,a in zip(types,type_acc)}
}
with open(OUT_DIR / f"{RUN_TYPE}_eval_summary.json","w") as f:
    json.dump(summary,f,indent=4)

print(f"Saved: {RUN_TYPE}_accuracy_by_type.png, {RUN_TYPE}_system_metrics.png, {RUN_TYPE}_eval_summary.json")