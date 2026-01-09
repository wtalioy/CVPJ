# ASRNet: Asymmetric Spectral Residual Network for Generalized AI-Generated Image Detection

This repository hosts the official PyTorch implementation of **ASRNet**, our method proposed in the paper [*Asymmetric Spectral Residual Network for Generalized AI-Generated Image Detection*](ASRNet.pdf). ASRNet is designed to differentiate AI-generated (“deepfake”) images from authentic ones, with a strong focus on generalizing to unseen generation methods.

## Key Features

ASRNet introduces a novel architecture that explicitly decouples semantic suppression from artifact mining. Its key contributions include:

-   **Anti-Semantic Constrained Convolution:** A learnable high-pass filter at the input stage that nullifies object-level semantics to prevent the model from overfitting to image content.
-   **Asymmetric Spectral Bottleneck:** A novel module that treats frequency components divergently, suppressing low-frequency structures that carry semantic information while preserving high-frequency generation fingerprints.
-   **Multi-Prototype Metric Learning Head:** Replaces the standard binary classifier to model the diversity of generative artifacts, improving cross-generator generalization.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/wtalioy/ASRNet.git
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset should be organized as follows:

-   `dataset/train/0_real`: Real images for training.
-   `dataset/train/1_fake`: Fake images for training.
-   `dataset/test`: Images for testing.
-   `dataset/label_test.csv`: Label file for the test set.

## Training

To train the ASRNet model, run the `train.py` script:

```bash
python train.py --model asrnet --epochs 50 --batch_size 32 --lr 2e-4
```

The training script will save TensorBoard logs and model checkpoints in the `runs` directory.

## Evaluation

To evaluate a trained ASRNet model, use the `evaluate.py` script, providing the path to the trained checkpoint:

```bash
python evaluate.py --model asrnet --checkpoint runs/<run-folder>/checkpoint_asrnet.pth
```
