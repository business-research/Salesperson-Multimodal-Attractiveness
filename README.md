
## Salesperson Attractiveness Beyond Looks: Mixed Method of Multimodal Machine Learning and Explainable AI


## Overview
This repository contains the code and data used in the research paper "Salesperson Attractiveness Beyond Looks: Mixed Method of Multimodal Machine Learning and Explainable AI". 

## Installation
Before running the scripts, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- Pandas
- Numpy
- Pickle
- Statsmodels
- Scikit-learn

You can install these packages using pip:
```bash
pip install torch pandas statsmodels numpy scikit-learn
```

## Dataset
- Due to GitHub's storage limitations, the complete dataset, including training, validation, and test data, has been uploaded to Google Drive.
- You can access the full dataset using the following link: https://drive.google.com/file/d/1QZ52F32h0jiPW31Z4fVyfHz-F6s5LMJJ/view?usp=drive_link.
- After download, save it in path: data


## Pretrained model
- Due to GitHub's storage limitations, the pretrained modela has been uploaded to Google Drive. 
- You can access the full dataset using the following link: https://drive.google.com/file/d/10EYH8j552ggWVG9oZ0cjd2JZO0UMSOq7/view?usp=drive_link.
- After download, save it in path: pretrained_model

### Multimodal Features
#### Verbal Features
- **Tool Used**: BERT.
- **Features**: 768 dimensions.

#### Vocal Features
- **Tool Used**: Librosa.
- **Features**: 24 dimensions.

#### Visual Features
- **Tool Used**: OpenFace 2.0.
- **Features**: 49 dimensions.


## Usage
To run the analysis, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/business-research/Salesperson-Multimodal-Attractiveness.
   ```
2. Navigate to the cloned directory.
3. Run the main script:
   ```bash
   python run_evaluation_pretrained_model.py
   ```

## Model Description
The codebase includes the Tensor Fusion Network (TFN) model, designed to handle verbal, vocal, and verbal  modalities and fusion techniques in the context of measuring salesperson attractiveness:

 **Full Fusion + Trimodal Data, TFN model:**
   - `TFN.py`: TFN model for verbal, vocal, and visual data.

The TFN model is based on pretrained models and demonstrates the prediction results. This model is specifically tailored for analyzing the salesperson attractiveness using multimodal data.



## Evaluation Metrics
The code includes functions for evaluating the model performance:

- Accuracy
- F1 Score
- Mean Absolute Error (MAE)
- Correlation Coefficient
- Loss

