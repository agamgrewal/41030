import os
import json

def create_vqa_subset(
    image_folder: str,
    questions_path: str,
    annotations_path: str,
    output_questions_path: str,
    output_annotations_path: str
):
    image_ids = {
        int(os.path.splitext(fname)[0])
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    }

    with open(questions_path, "r") as f:
        questions_data = json.load(f)

    with open(annotations_path, "r") as f:
        annotations_data = json.load(f)

    filtered_questions = [
        q for q in questions_data["questions"] if q["image_id"] in image_ids
    ]
    trimmed_questions = {
        **{k: v for k, v in questions_data.items() if k != "questions"},
        "questions": filtered_questions
    }

    filtered_annotations = [
        a for a in annotations_data["annotations"] if a["image_id"] in image_ids
    ]
    trimmed_annotations = {
        **{k: v for k, v in annotations_data.items() if k != "annotations"},
        "annotations": filtered_annotations
    }

    os.makedirs(os.path.dirname(output_questions_path), exist_ok=True)
    with open(output_questions_path, "w") as fq:
        json.dump(trimmed_questions, fq)
    with open(output_annotations_path, "w") as fa:
        json.dump(trimmed_annotations, fa)

    question_ids = {q["image_id"] for q in trimmed_questions["questions"]}
    annotation_ids = {a["image_id"] for a in trimmed_annotations["annotations"]}
    unmatched = question_ids.symmetric_difference(annotation_ids)

    print(f"Trimmed questions: {len(filtered_questions)}")
    print(f"Trimmed annotations: {len(filtered_annotations)}")
    print(f"Unmatched image IDs: {len(unmatched)}")
    print(f"Saved to: {output_questions_path}")
    print(f"          {output_annotations_path}")


if __name__ == "__main__":
    create_vqa_subset(
        image_folder="data/sample5000",
        questions_path="data/v2_OpenEnded_mscoco_train2014_questions.json",
        annotations_path="data/v2_mscoco_train2014_annotations.json",
        output_questions_path="data/annotations/vqa_train_sample_questions.json",
        output_annotations_path="data/annotations/vqa_train_sample_annotations.json"
    )
