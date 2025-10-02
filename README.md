# Image Captioning as a Data Augmentation Strategy for Visual Question Answering  

## Research Question  
**Can image captioning be used as an effective data augmentation strategy to improve Visual Question Answering (VQA) performance?**  

## Abstract  
This research project investigates whether automatically generated image captions can serve as an effective form of data augmentation to enhance the performance of Visual Question Answering (VQA) systems. The study explores the integration of captions alongside traditional augmentation methods such as image flipping, cropping, and inversion. By embedding richer semantic information into the training process, the aim is to improve model generalisation, robustness, and accuracy on benchmark VQA tasks.  

## Background and Motivation  
VQA is a multimodal problem that requires understanding both visual and textual information to answer natural language questions about images. While recent advances in deep learning, particularly transformer-based architectures, have significantly improved VQA performance, these models remain highly data-dependent and prone to dataset biases.  

Image captioning, which generates descriptive natural language statements from images, offers an additional perspective on visual content. Integrating such descriptions during training may help VQA systems develop stronger alignment between vision and language modalities. This research seeks to examine whether this approach leads to measurable improvements in performance.  

## Research Objectives  
- Design a VQA training pipeline that incorporates image captions as data augmentation.  
- Evaluate the effectiveness of caption-based augmentation in comparison to conventional augmentation methods (e.g., flipping, cropping, inversion).  
- Analyse whether semantic information derived from captions improves model accuracy and generalisation.  
- Contribute insights on multimodal augmentation strategies for future VQA research.  

## Literature Review Snapshot  
- **Agrawal et al. (2017)** – highlighted dataset biases in VQA, motivating the need for stronger generalisation strategies.  
- **Zhou et al. (2020)** – explored large-scale pretraining for vision-language tasks, showing improved robustness.  
- **Tan & Bansal (2019)** – introduced LXMERT, a cross-modality encoder that inspired further augmentation experiments.  
- **Anderson et al. (2018)** – showed the importance of image captioning models and their role in generating descriptive features.  

These studies provide the foundation for testing caption-based augmentation as a method for enhancing VQA performance.  

## Methodology (Planned)  

### 1. Datasets  
- **VQA v2 dataset**: benchmark dataset for training and evaluation.  
- **MS COCO Captions dataset** or captions generated from a pretrained captioning model.  

### 2. Augmentation Strategies  
- **Traditional**: image flipping, cropping, inversion, colour jitter.  
- **Proposed**: auto-generated captions as supplementary input features.  

### 3. Models  
- **Baseline**: pretrained VQA model without augmentation.  
- **Experimental**: VQA model trained with caption-based augmentation.  

### 4. Evaluation Metrics  
- VQA accuracy (standard evaluation protocol).  
- Qualitative analysis of model responses.  
- Comparative analysis between augmentation strategies.  

## Current Progress  
- Research proposal completed and research question defined.  
- Development environment configured (Conda + PyCharm).  
- Preliminary design of augmentation strategies (traditional and caption-based).  
- Preparation for baseline model training.  

## Anticipated Outcomes  
- Determination of whether caption-based augmentation provides measurable performance improvements in VQA tasks.  
- Identification of the strengths and limitations of multimodal augmentation.  
- Contribution to the broader discussion on data efficiency and robustness in multimodal learning.  

## Repository Structure (Draft)
in progress
## Submodules  
This repository uses the [LXMERT](https://github.com/airsplay/lxmert) repo as a **submodule**.  

After cloning, initialise the submodule with:  
```bash
git submodule update --init --recursive
