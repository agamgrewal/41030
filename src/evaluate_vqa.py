import json
from collections import defaultdict

# Paths
PRED_PATH = "/data/akgrewal/pycharm_projects/capstone/results/baseline_results.json"
GT_PATH = "/data/akgrewal/pycharm_projects/capstone/src/data/annotations/vqa_train_sample_questions.json"

# Load predictions
with open(PRED_PATH, "r") as f:
    preds = json.load(f)
pred_dict = {p["question_id"]: p["answer"].strip().lower() for p in preds}

# Load ground-truth
with open(GT_PATH, "r") as f:
    raw = json.load(f)

if isinstance(raw, dict) and "questions" in raw:
    data = raw["questions"]
else:
    data = raw

# Evaluate
total = 0
correct = 0
no_answer = 0

for item in data:
    qid = item["question_id"]
    answers = [a.lower() for a in item.get("answers", [])]
    pred = pred_dict.get(qid, None)

    if pred is None:
        no_answer += 1
        continue

    total += 1
    # Standard VQA soft-accuracy: if prediction appears in human answers
    if pred in answers:
        correct += 1

# Accuracy
if total > 0:
    accuracy = (correct / total) * 100
else:
    accuracy = 0

print("=====================================")
print(f"✅ Evaluated {total} questions")
print(f"✅ Correct answers: {correct}")
print(f"⚠️ Missing predictions: {no_answer}")
print(f"🎯 Accuracy: {accuracy:.2f}%")
print("=====================================")
