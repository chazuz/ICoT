# I-CoT: Implicit Chain-of-Thought via Visual Cue Injection

This repository contains code for a small experimental study investigating whether implicit visual cue injection (I-CoT) can influence physical reasoning performance in Vision-Language Models.

## Idea

The project tests whether adding simple perceptual cues to prompts can improve reasoning without changing model weights.

The cues are short keyword lists describing relevant visual properties such as motion, structure, or object interaction.

Examples include:
- motion direction cues (vanishing point, spatial continuity)
- manipulation cues (hand position, contact surface, object displacement)
- property cues (texture, rigidity, sharpness)

## Dataset

- PhysBench (50-sample stratified subset)
- Task types: dynamics, relationships, property, scene

## Model

- Qwen2.5-VL-3B-Instruct
- CPU inference

## Method

Two conditions are compared:

- Baseline prompt
- I-CoT prompt (baseline plus visual cue injection)

No training or fine-tuning is used. Only prompt structure is modified.

## Results

- Baseline accuracy: 38.0 percent
- I-CoT accuracy: 38.0 percent
- Contested samples: split evenly (6 to 6)

The results suggest that simple cue injection does not consistently improve performance, but may have task-dependent effects.

## Repository structure

scripts/
  inference/        evaluation scripts

data/
  processed/        dataset subsets
  raw/              images

experiments/
  logs/             evaluation outputs

## Run evaluation

python scripts/inference/run_physbench_terminal_compare.py --num_samples 50

## Requirements

pip install -r requirements.txt

## Model

Qwen2.5-VL-3B-Instruct via HuggingFace Transformers

## Paper

I-CoT: Implicit Chain-of-Thought Reasoning via Visual Cue Injection

## License

Academic use only
