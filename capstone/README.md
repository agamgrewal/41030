# Image Captioning as a Data Augmentation Strategy for Visual Question Answering  

## Research Question  
**Does incorporating automatically generated image captions enhance the performance of pre-trained Visual Question Answering (VQA) models?**  

---

## Abstract  
This project investigates whether automatically generated image captions can serve as an effective form of semantic data augmentation for Visual Question Answering (VQA) systems. Instead of modifying images through conventional augmentations such as flipping or cropping, this study explores enriching the text input by pairing each image with a generated caption. By comparing a baseline (image + question) setup against an augmented (caption + image + question) configuration, the research evaluates how additional linguistic context affects the inference accuracy of a pre-trained VQA model.  

---

## Background and Motivation  
VQA tasks require models to integrate information from both visual and textual modalities to answer natural-language questions about images. Although transformer-based architectures have achieved strong performance on large-scale datasets, they often rely heavily on training data and remain sensitive to bias.  

Image captioning provides an auxiliary textual description of visual content and may improve multimodal alignment between vision and language. Supplying captions to a pre-trained VQA model at inference time offers a low-cost strategy to test whether semantic augmentation can improve reasoning and answer accuracy without any retraining.  

---

## Research Objectives  
- Construct a reproducible evaluation pipeline for comparing caption-augmented and baseline VQA inference.  
- Quantitatively assess whether caption-based augmentation improves model accuracy on a curated subset of the VQA v2 dataset.  
- Analyse qualitative differences in generated answers across both settings.  
- Provide insights into the effectiveness of linguistic data augmentation in multimodal understanding tasks.  

---

## Relevant Literature  
- **Agrawal et al. (2017)** – Identified dataset bias in VQA and the need for stronger generalisation.  
- **Anderson et al. (2018)** – Demonstrated the potential of image captioning for producing meaningful visual representations.  
- **Tan & Bansal (2019)** – Introduced LXMERT, highlighting cross-modality pre-training benefits.  
- **Zhou et al. (2020)** – Showed that large-scale pre-training improves robustness in vision–language tasks.  

These works motivate testing captions as an additional semantic cue for VQA.  

---

## Methodology  

### 1. Datasets  
- **VQA v2 (train2014 subset):** Primary benchmark for evaluation.  
- **MS COCO (train2014 images):** Source of corresponding visual data.  
- **Generated captions:** Created using the BLIP image-captioning model.  

### 2. Experimental Configurations  
| Configuration | Input | Description |
|---------------|--------|-------------|
| **Baseline** | *Image + Question* | Standard pre-trained BLIP-VQA inference. |
| **Caption-Augmented** | *Caption + Image + Question* | Adds BLIP-generated captions to enrich context. |

### 3. Model  
- **Salesforce BLIP-VQA-Base:** A pre-trained transformer model for Visual Question Answering, evaluated without fine-tuning.  

### 4. Evaluation Metrics  
- Overall accuracy against ground-truth VQA answers.  
- Comparative performance between baseline and caption-augmented runs.  
- Qualitative examples illustrating semantic improvements or errors.  

---

## Current Progress  
- Environment configured (Python, Conda, PyTorch, Transformers).  
- COCO and VQA v2 datasets prepared and aligned to 5 000 valid image–question pairs.  
- Baseline and caption-augmented inference pipelines implemented.  
- Caption generation completed using the BLIP image-captioning model.  

---

## Expected Outcomes  
- Empirical evidence on whether captions improve the reasoning capability of pre-trained VQA systems.  
- Quantitative comparison of baseline and caption-augmented accuracies.  
- Discussion on the potential of text-based augmentation for multimodal learning efficiency.  

---

## Repository Structure
```
capstone/
├── README.md
├── requirements.txt
├── results/
│ ├── baseline_predictions.json
│ ├── baseline_accuracy_summary.json
│ ├── caption_predictions.json
│ └── caption_accuracy_summary.json
└── src/
├── data/
│ ├── annotations/
│ │ ├── vqa_train_sample5000_annotations.json
│ │ └── vqa_train_sample5000_questions.json
│ ├── captions_sample5000.jsonl
│ └── images_sample5000/
├── preprocessing/
│ ├── dataset_sampler_5000.py
│ └── generate_image_captions.py
├── models/
│ ├── run_vqa_baseline_inference.py
│ └── run_vqa_caption_inference.py
└── evaluation/
├── evaluate_baseline_performance.py
└── evaluate_caption_performance.py
```
