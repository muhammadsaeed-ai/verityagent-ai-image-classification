<div align="center">

# VerityAgent AI Image Classification

**A modular ImageNet classifier using PyTorch, Torchvision, and pretrained ResNet-50.**

*VerityAgent Python AI Mini Projects*

</div>

---

## Overview

Classify a single image with Torchvision's pretrained **ResNet-50** model. The CLI prints the top-k ImageNet categories and their softmax probability scores, saves a side-by-side visualization, and can optionally export predictions to JSON.

No training job, API key, or cloud inference service is required. The first real inference run may download the pretrained checkpoint if it is not cached locally.

## Features

- Uses the modern `ResNet50_Weights.DEFAULT` API.
- Uses the selected weights' official preprocessing transforms and category metadata.
- Runs inference with `torch.inference_mode()`, applies softmax, and ranks predictions with `torch.topk`.
- Defaults to five predictions and validates top-k against the available labels.
- Automatically selects CUDA when available, with CPU fallback and an explicit device option.
- Saves a readable horizontal probability chart alongside the input image.
- Optional JSON export with rank, ImageNet class index, label, probability, and percentage.
- Automated tests use a mock network and do not download pretrained model weights.

## Inference pipeline

~~~mermaid
flowchart TD
    A[Input image] --> B[Validate and decode with Pillow]
    B --> C[Selected weights preprocessing]
    C --> D[Move tensor to CPU or CUDA]
    D --> E[Pretrained ResNet-50]
    E --> F[Softmax class probabilities]
    F --> G[Rank top-k categories]
    G --> H[Terminal predictions]
    G --> I[Image + chart output]
    G --> J[Optional JSON export]
~~~

## Project structure

~~~text
verityagent-ai-image-classification/
├── app.py
├── classifier.py
├── visualizer.py
├── config.py
├── requirements.txt
├── .gitignore
├── README.md
└── tests/
    ├── test_classifier.py
    └── test_visualizer.py
~~~

## Requirements

- Python **3.10–3.14**
- PyTorch `2.10.0`
- Torchvision `0.25.0`
- Pillow `12.3.0`
- Matplotlib `3.10.8`
- pytest `9.0.2` (tests)

The pinned PyTorch and Torchvision versions are a matching pair. Installation can differ for CPU-only and CUDA environments. For the correct wheel command for your operating system and accelerator, consult the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) and [Torchvision installation notes](https://github.com/pytorch/vision#installation).

## Installation

### 1. Create a virtual environment

**Windows PowerShell**

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
~~~

**macOS / Linux**

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~

### 2. Install dependencies

~~~bash
python -m pip install -r requirements.txt
~~~

If your environment requires a specific CPU/CUDA PyTorch build, use the official PyTorch selector to install the matching `torch==2.10.0` and `torchvision==0.25.0` packages first. Then install the remaining pinned packages:

~~~bash
python -m pip install Pillow==12.3.0 matplotlib==3.10.8 pytest==9.0.2
~~~

Do not mix incompatible PyTorch and Torchvision builds.

### 3. Run tests

~~~bash
python -m pytest -q
~~~

The test suite injects deterministic mock outputs and temporary local images, so it runs without downloading ResNet-50 weights.

## Usage

### Basic classification

~~~bash
python app.py ./example.jpg
~~~

By default, the application saves `classification_results.png` and prints the top five predictions.

### Set visualization and JSON output paths

~~~bash
python app.py ./images/cat.jpg \
  --output ./results/cat-classification.png \
  --json-output ./results/cat-predictions.json
~~~

Output directories are created automatically.

### Change the number of predictions

~~~bash
python app.py ./images/bird.png --top-k 3
~~~

Valid values are integers from 1 through the number of model output classes (normally 1,000 for ImageNet-1K). Invalid values are rejected before model loading.

### Choose an inference device

~~~bash
python app.py ./images/object.webp --device auto
python app.py ./images/object.webp --device cpu
python app.py ./images/object.webp --device cuda
~~~

`auto` is the default and chooses CUDA when available, otherwise CPU. If CUDA is explicitly selected but unavailable, the CLI reports an error.

### CLI reference

~~~text
python app.py IMAGE [--output PATH] [--json-output PATH]
               [--top-k N] [--device {auto,cpu,cuda}]
~~~

| Argument | Default | Purpose |
|---|---|---|
| `IMAGE` | Required | Input image path |
| `--output` | `classification_results.png` | Visualization path |
| `--json-output` | Disabled | Optional JSON export path |
| `--top-k` | `5` | Number of ranked predictions |
| `--device` | `auto` | Select `auto`, `cpu`, or `cuda` |

## Output

Console output follows this format. The predicted labels and values are produced by inference on the supplied image; the examples below are placeholders, not claimed results.

~~~text
VerityAgent AI Image Classification
Input: images/example.jpg
Device: cpu

Top 5 predictions:
1. <predicted ImageNet category> — <probability>%
2. <predicted ImageNet category> — <probability>%
...

Visualization saved to: classification_results.png
JSON results saved to: predictions.json
~~~

The chart shows the input image beside a horizontal bar chart with category labels and probability percentages. The optional JSON file records the model name, selected weight enum, each prediction's rank, category index, category name, probability (0–1), and percentage.

## Model architecture and preprocessing

- **Architecture:** ResNet-50, a residual convolutional neural network.
- **Pretrained weights:** `ResNet50_Weights.DEFAULT`, Torchvision's default ImageNet-1K weights for the installed version. The `DEFAULT` alias can change between Torchvision releases.
- **Preprocessing:** `weights.transforms()` supplies the preprocessing associated with the selected weights.
- **Category labels:** `weights.meta["categories"]` provides the corresponding ImageNet labels.
- **Inference:** the model runs in evaluation mode; the input tensor and model share the selected device; inference uses `torch.inference_mode()`; softmax converts logits to class probabilities; top-k selects the highest scores.

Official references: [ResNet-50 model and weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) and [Torchvision model weights guide](https://docs.pytorch.org/vision/stable/models.html).

## Supported image formats

Pillow commonly supports JPEG/JPG, PNG, BMP, GIF, WebP, and TIFF; exact format support may depend on the Pillow build. Images are converted to RGB before preprocessing. An animated image is treated as a single image, typically its first frame.

## Important limitation: classification is not object detection

This is **whole-image classification**. The model predicts categories learned during ImageNet training; it does not locate individual objects, generate bounding boxes, track objects, count instances, or produce segmentation masks. A photograph containing several objects still receives one ranked set of image-level predictions. Use an object-detection model when object locations are required.

ImageNet uses a fixed category set, so specialized objects or images unlike the training data can be misclassified. Softmax values are model probabilities, not guaranteed-calibrated confidence estimates. Do not treat the output as a substitute for expert judgement in safety-critical decisions.

## Troubleshooting

**Image file does not exist** — Check the file name, current working directory, and path. Quote paths that contain spaces.

**The file is not a valid image** — Verify that the file is supported by Pillow and is not a renamed or truncated non-image file.

**The first run is slow or cannot load weights** — The first real run may need an internet connection to download the pretrained checkpoint. Retry with a working connection and check that the local Torch cache is writable. Unit tests do not require a download.

**CUDA is unavailable** — Run with `--device cpu` or `--device auto`, or install a PyTorch build compatible with your CUDA environment.

**The output directory cannot be written** — Select a writable destination and check that the target is not locked by another application.

**Model output and category counts do not match** — Confirm that the model head, selected weights, and corresponding category metadata are aligned. The classifier checks for this mismatch before returning predictions.

## Development and testing

- Run all unit tests from the project root with `python -m pytest -q`.
- Tests mock the model to avoid heavyweight checkpoint downloads.
- Model checkpoints and generated reports are excluded by `.gitignore`.

---

**Attribution:** VerityAgent Python AI Mini Projects.
