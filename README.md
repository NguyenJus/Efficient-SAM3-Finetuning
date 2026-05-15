# Efficient-SAM3-Finetuning

This open source package is designed to help facilitate rapid and scalable finetuning of SAM3 on any niche dataset a user might need. The intent is for low compute users to cater the massive foundation model to meet their niche needs. 

## Key Libraries 

- https://github.com/facebookresearch/sam3
- https://github.com/huggingface/peft
- https://github.com/ray-project/ray
- https://github.com/argoproj/argoproj

## Input Modalities 

Package supports any input prompt made available by SAM. This includes:

- Text
- Bounding Boxes
- Points
- Masks
- Others if available

## Output Modalities 

Package supports outputting either instance segmentation masks or semantic segmentation masks depending on users' needs. Package supports both static image datasets and videos.

## Models

Project supports both SAM3 and SAM 3.1

- https://huggingface.co/facebook/sam3.1
- https://huggingface.co/facebook/sam3

## Licenses 

This package is fully open source and is free for commercial use. As such, all requirements must conform.