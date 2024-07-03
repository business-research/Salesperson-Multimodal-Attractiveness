
## Do Looks Matter in Sales? Modelling Attractiveness Using MMML Based on Live Streaming Videos

## Overview
This repository contains the code and data used in the research paper "Do Looks Matter in Sales? Modelling Attractiveness Using MMML Based on Live Streaming Videos". 

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
--

## Dataset
Due to GitHub's storage limitations, the complete dataset, including training, validation, and test data, has been uploaded to Google Drive.
You can access the full dataset using the following link: https://drive.google.com/drive/folders/1DZ7ZsC79xEYmY98D_4c8ReETXHFlprm7?usp=drive_link.
After download, save it in path: data


## Pretrained model
Due to GitHub's storage limitations, the pretrained modela has been uploaded to Google Drive. 
You can access the full dataset using the following link: https://drive.google.com/drive/folders/1gwTUbA9rXLyN2_0xa65jtuJs4H8giWSA?usp=drive_link.
After download, save it in path: pretrained_model

### Multimodal Features 
#### Vocal Features
- **Tool Used**: Librosa.
- **Features**: 24 dimensions.


#### Visual Features
- **Tool Used**: OpenFace 2.0.
- **Features**: 465 dimensions.


#### Verbal Features
- **Tool Used**: BERT.
- **Features**: 768 dimensions.




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
The codebase includes the Tensor Fusion TFN model, designed to handle visual, voice, and verbal  modalities and fusion techniques in the context of measuring salesperson attractiveness:

 **Full Fusion + Trimodal Data, TFN model:**
   - `TFN.py`: TFN model for verbal, vocal, and visual data.

The TFN model is based on pretrained models and demonstrates the prediction results. This model is specifically tailored for analyzing the salesperson attractiveness using multimodal data.



## Evaluation Metrics
The code includes functions for evaluating the model performance:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Correlation Coefficient
- Accuracy
- F1 Score


