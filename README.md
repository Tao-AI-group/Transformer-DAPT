# Transformer-DAPT: A Multi-head Attention-based Survival Analysis for Dynamic Ischemic and Bleeding Risk Assessment in Patients on DAPT Post-PCI

This repository contains the implementation of Transformer-DAPT, a novel transformer-based model developed for optimizing dual antiplatelet therapy (DAPT) management after percutaneous coronary intervention (PCI). The model analyzes multi-hot encoded clinical features to individually predict both ischemic events and bleeding complications within a 365-day window post-PCI. By providing separate predictions for each outcome type, Transformer-DAPT enables clinicians to assess a patient's specific risk profile for both complications independently. The model outperforms traditional survival models (DeepSurv and DeepHit) and existing clinical risk scores in time-specific concordance indices at clinically relevant intervals (30, 60, 90, 180, 270, and 365 days), offering accurate risk predictions to support personalized DAPT decision-making.

## Overview

Transformer-DAPT applies the transformer architecture to model survival data in the context of post-PCI management, with both clinical and technical innovations:

1. **Embeds clinical features into a learned representation space**, transforming multi-hot encoded patient data into dense vector representations
2. **Uses multi-head self-attention to model interactions between features**, capturing complex relationships between clinical variables that traditional statistical models miss
3. **Generates separate individual risk predictions for both ischemic events and bleeding complications** at multiple clinically relevant time points (30, 60, 90, 180, 270, and 365 days)
4. **Makes both survival probability predictions over time and binary classification predictions**, providing both detailed temporal risk trajectories and simplified risk scores
5. **Provides feature importance analysis through attention weights and integrated gradients**, identifying patient-specific clinical features that contribute to each type of risk
6. **Includes calibration capabilities through isotonic regression** to ensure well-calibrated probability estimates for reliable clinical decision-making
7. **Demonstrates strong performance in comparison with traditional survival analysis approaches** like DeepSurv and DeepHit

## Clinical Context and Applications

Coronary artery disease (CAD) is one of the leading causes of global death, accounting for approximately nine million deaths annually. The standard of care following percutaneous coronary intervention (PCI) with drug-eluting stent implantation includes dual antiplatelet therapy (DAPT), comprising aspirin and a P2Y12 receptor inhibitor. However, determining the optimal DAPT duration and regimen for individual patients remains challenging.

Existing risk stratification tools such as the DAPT score and PRECISE-DAPT score have modest discriminative ability (C-index ~0.70) and are limited to predefined temporal windows. Transformer-DAPT addresses these limitations by:

1. **Personalized Risk Assessment**: Generating separate risk predictions for ischemic events and bleeding complications at multiple time points post-PCI
2. **Dynamic Temporal Modeling**: Providing risk estimates across various clinically meaningful intervals throughout the first year after intervention
3. **Interpretable Predictions**: Identifying which clinical factors contribute to each individual patient's risk profile
4. **Superior Discrimination**: Achieving higher predictive accuracy than traditional models and clinical risk scores
5. **Clinical Decision Support**: Enabling evidence-based, individualized approaches to DAPT management

## Model Architecture

Transformer-DAPT consists of several specialized components designed for survival analysis:

- **FeatureEmbedding Layer**: Converts categorical indices into dense embeddings (shape: `[batch_size, num_features, embedding_size]`), with dropout regularization (rate: 0.2) and padding token handling.

- **Feature Encoder**: Processes individual feature embeddings through a linear layer, layer normalization, ReLU activation, and dropout, creating a representation of each feature in isolation.

- **Multi-Head Self-Attention**: The core component that dynamically models feature interactions, with configurable number of attention heads (2, 4, or 8 heads based on hyperparameter tuning). Importantly, it uses masked attention to handle variable-length feature sets, where padding tokens are ignored through a `key_padding_mask`.

- **Dual-Output Classification Heads**:
  - **PC-Hazard Head**: Outputs time-dependent hazard estimates for survival prediction using the PCHazard framework from PyCox library, with a configurable number of discrete time intervals (typically 18-30 intervals)
  - **Binary Classification Head**: Single sigmoid-activated neuron that provides an immediate event risk score for straightforward clinical interpretation

- **Calibration Layer**: Post-processing with isotonic regression to ensure reliable probability estimates, particularly important for clinical decision-making

## Repository Structure

```
Transformer-DAPT/
├── modules/                        # Core model implementation modules
│   ├── model.py                    # Main Transformer-DAPT model implementation
│   ├── modeling_bert.py            # Classification head components based on BERT architecture
│   ├── config.py                   # Model configuration settings
│   ├── train_utils.py              # Training utilities and Trainer class
│   ├── model_comparison.py         # Statistical comparison tools for model evaluation
│   ├── utils.py                    # General utility functions
│   ├── evaluate_utils.py           # Evaluation metrics and utility functions
│   └── calibration_advanced_V2.py  # Model calibration utilities
├── 01_Optuna_Hyperparameter_Tuning.py  # Hyperparameter optimization with Optuna
├── 02_Feature_Importance_IntegratedGradients.py  # Feature importance analysis
├── 03_Evaluation_Transformer_DAPT_Multiple_Intervals_Bootstrapping.py  # Model evaluation with bootstrapping
├── 04_Evaluation_Baseline_Deephit_Multiple_Intervals_Bootstrapping.py  # DeepHit baseline evaluation
├── 05_Evaluation_Baseline_Deepsurv_Multiple_Intervals_Bootstrapping.py  # DeepSurv baseline evaluation
├── 06_Compare_Models_Performance.py  # Model comparison framework
└── 07_Transformer_DAPT_Model_Calibration_V2.py  # Advanced calibration implementation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tao-AI-group/Transformar_DAPT.git
cd Transformar_DAPT
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Optuna
- PyCox
- EasyDict
- Captum (for integrated gradients)

## Usage

### Model Training with Hyperparameter Optimization

The model training pipeline includes hyperparameter optimization using Optuna:

```bash
python 01_Optuna_Hyperparameter_Tuning.py
```

This script:
- Loads and preprocesses the data
- Optimizes hyperparameters (learning rate, batch size, embedding dimensions, etc.)
- Trains the best model configuration
- Evaluates performance on validation and test sets

### Model Evaluation and Comparison

Evaluate the Transformer-DAPT model against baseline models (DeepSurv and DeepHit):

```bash
# Evaluate Transformer-DAPT
python 03_Evaluation_Transformer_DAPT_Multiple_Intervals_Bootstrapping.py

# Evaluate baseline models
python 04_Evaluation_Baseline_Deephit_Multiple_Intervals_Bootstrapping.py
python 05_Evaluation_Baseline_Deepsurv_Multiple_Intervals_Bootstrapping.py

# Compare model performance with statistical testing
python 06_Compare_Models_Performance.py
```

### Feature Importance Analysis

Analyze and visualize feature importance using Integrated Gradients:

```bash
python 02_Feature_Importance_IntegratedGradients.py --model_path [path_to_model] --saved_data_dir [path_to_data]
```

### Model Calibration

Apply and evaluate calibration techniques:

```bash
python 07_Transformer_DAPT_Model_Calibration_V2.py
```

## Key Features

1. **Separate Risk Prediction for Critical Outcomes**: Independently predicts both ischemic events and bleeding complications in patients after PCI, providing specific risk profiles for each outcome type without assuming a direct trade-off between them.

2. **Time-Specific Risk Assessment**: Provides risk predictions at multiple clinically relevant time points (30, 60, 90, 180, 270, and 365 days), allowing for dynamic risk assessment throughout the critical first year post-PCI.

3. **Feature Importance Analysis**: Identifies which clinical factors (diagnoses, medications, procedures) contribute most significantly to each individual risk through attention weights and integrated gradients, enhancing clinical interpretability.

4. **Well-Calibrated Probability Estimates**: Employs isotonic regression to ensure the predicted probabilities match observed event rates, crucial for reliable clinical risk stratification and shared decision-making.

5. **Handling Complex Clinical Data**: Effectively processes multi-hot encoded clinical features where each patient has a variable number of active conditions, medications, and procedures - a common challenge in electronic health record data.

6. **Statistical Validation**: Includes comprehensive bootstrap-based statistical evaluation with confidence intervals and rigorous comparison against both established baseline models (DeepSurv and DeepHit) and clinical risk scores.

## Citation

If you use Transformer-DAPT in your research, please cite our paper:

```bibtex
@article{transformer_dapt,
  title={Transformer-DAPT: A Deep Attention Model for Time-to-Event Prediction in Healthcare},
  author={[Author names]},
  journal={[Journal Name]},
  year={[Year]},
  volume={[Volume]},
  pages={[Pages]}
}
```

## License

[Specify the license, e.g., MIT, Apache 2.0, etc.]

## Contributors

[List of contributors]

## Acknowledgments

[Any acknowledgments, funding sources, etc.]
