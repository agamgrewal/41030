import json, re
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

SAMPLE_IMG_DIR = Path("../data/sample5000")
QUESTIONS_JSON = Path("../data/annotations/vqa_train_sample5000_questions.json")
ANNOTATIONS_JSON = Path("../data/annotations/vqa_train_sample5000_annotations.json")
BLIP_CAPTIONS_PATH = Path("../data/annotations/captions_sample5000.jsonl")
COCO_CAPTIONS_JSON = None

def extract_image_id_from_filename(fname):
    m = re.search(r"(\d{6,12})(?=\.jpe?g$)", fname)
    return int(m.group(1)) if m else None

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

img_files = [p.name for p in SAMPLE_IMG_DIR.glob("*.jpg")]
image_ids = set(extract_image_id_from_filename(n) for n in img_files if extract_image_id_from_filename(n))
print(f"Images in sample: {len(image_ids):,}")

Q = load_json(QUESTIONS_JSON)
A = load_json(ANNOTATIONS_JSON)
questions = Q.get("questions", Q)
annotations = A.get("annotations", A)

qs_by_image = defaultdict(list)
for q in questions:
    if q.get("image_id") in image_ids:
        qs_by_image[q["image_id"]].append(q)

anns_by_qid = {ann["question_id"]: ann for ann in annotations}
filtered_questions = [q for iid in image_ids for q in qs_by_image[iid]]

print(f"Questions for sample images: {len(filtered_questions):,}")

total_answers = 0
qt_counter = Counter()
for q in filtered_questions:
    qid = q["question_id"]
    ann = anns_by_qid.get(qid)
    if not ann:
        continue
    answers = ann.get("answers", [])
    total_answers += len(answers)
    qt_counter[ann.get("question_type", "Unknown")] += 1

print(f"Answer annotations: {total_answers:,}")

# --- Question prefix extraction ---
def get_prefix(q):
    text = q["question"].strip().lower()
    match = re.match(r"([a-z']+\s[a-z']+|[a-z']+)", text)
    return match.group(1) if match else "other"

prefix_counts = Counter(get_prefix(q) for q in filtered_questions)
top_prefixes = prefix_counts.most_common(15)

# --- Summary printout ---
print("\n=== SUMMARY (paste in report) ===")
print(f"- Sample images: {len(image_ids):,}")
print(f"- Questions linked to sample: {len(filtered_questions):,}")
avg_q_per_img = len(filtered_questions)/len(image_ids) if image_ids else 0
avg_ans_per_q = total_answers/len(filtered_questions) if filtered_questions else 0
print(f"- Avg questions per image: {avg_q_per_img:.2f}")
print(f"- Avg answers per question: {avg_ans_per_q:.2f}")

# --- 1. Bar chart of top question prefixes ---
if top_prefixes:
    labels, values = zip(*top_prefixes)
    plt.figure(figsize=(10,5))
    plt.barh(labels[::-1], values[::-1], color='steelblue')
    plt.xlabel("Count")
    plt.ylabel("Question Prefix")
    plt.title("Top 15 Question Prefixes in Sample5000")
    plt.tight_layout()
    plt.savefig("question_prefix_distribution.png", dpi=300)
    plt.close()
    print("Saved: question_prefix_distribution.png")

# --- 2. Pie chart of question types ---
# --- 2. Simplified category grouping ---
def classify_question(qtext):
    qtext = qtext.lower()
    if qtext.startswith(("is", "are", "does", "do", "has", "have", "was", "were", "can", "could")):
        return "Yes/No"
    elif "how many" in qtext or re.search(r"\d", qtext):
        return "Number"
    else:
        return "Other"

category_counter = Counter(classify_question(q["question"]) for q in filtered_questions)

# --- Pie chart (clean version) ---
labels, sizes = zip(*category_counter.items())
colors = ["#69b3a2", "#f5a623", "#4c72b0"]

plt.figure(figsize=(5,5))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    textprops={"fontsize": 10},
)
plt.title("Simplified Question Type Distribution", fontsize=12)
plt.tight_layout()
plt.savefig("question_type_clean.png", dpi=300)
plt.close()
print("Saved: question_type_clean.png")

# --- 3. Table summary for dataset counts (as PNG) ---
table_data = [
    ["Images in sample", f"{len(image_ids):,}"],
    ["Total questions", f"{len(filtered_questions):,}"],
    ["Total answers (annotations)", f"{total_answers:,}"],
    ["Average questions per image", f"{avg_q_per_img:.2f}"],
    ["Average answers per question", f"{avg_ans_per_q:.2f}"],
    ["Yes/No questions", f"{category_counter.get('Yes/No', 0):,}"],
    ["Number questions", f"{category_counter.get('Number', 0):,}"],
    ["Other questions", f"{category_counter.get('Other', 0):,}"]
]

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=["Metric", "Count"], loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#E6E6E6")
        cell.set_text_props(weight="bold")
plt.title("Dataset Summary (Sample5000)", fontsize=12, pad=10)
plt.tight_layout()
plt.savefig("dataset_summary_table.png", dpi=300)
plt.close()
print("Saved: dataset_summary_table.png")

print("\n Graphs generated successfully.")
