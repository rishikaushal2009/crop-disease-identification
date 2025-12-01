# Crop Disease Classification (Take-Home Task)

## Overview
This project builds a plant leaf disease classifier using a subset of the PlantVillage dataset (10–15 classes) with transfer learning (PyTorch). It includes exploratory data analysis, model training with augmentation, evaluation (confusion matrix & sample predictions), and an optional Gradio app for interactive inference.

## Folder Structure
```
.
├─ crop_disease_classification.ipynb  # Main notebook (EDA, training, evaluation)
├─ app.py                         # Optional Gradio app for inference
├─ models/                        # Saved model weights (.pth)
├─ requirements.txt               # Python dependencies
└─ README.md                      # Documentation & business recommendation
```

## Setup Instructions
1. Create and activate virtual environment (already present as `.venv` if you used VS Code):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
3. (Optional) Configure Kaggle API to auto-download dataset:

    - Obtain an API token from <https://www.kaggle.com/settings> (click "Create New Token"). This downloads `kaggle.json`.
    - Place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json` (create the folder if missing).
    - Restrict permissions (Windows PowerShell):

       ```powershell
       $kagPath = "$env:USERPROFILE\.kaggle"
       New-Item -ItemType Directory -Force -Path $kagPath
       Copy-Item .\kaggle.json $kagPath
       icacls "$kagPath\kaggle.json" /inheritance:r
       icacls "$kagPath\kaggle.json" /grant:r "$($env:USERNAME):(R)" "Administrators:(R)"
       ```

    - Test the CLI:

       ```powershell
       kaggle datasets list -s plant
       ```

      - Download one PlantVillage variant (choose ONE):
         - Full raw (example slug, adjusting if needed): `plantvillage/plantvillage-dataset`
         - Preprocessed subset: `emmarex/plantdisease`

      
         ```powershell
         python download_dataset.py --slug emmarex/plantdisease --output data/raw  --unzip
         # For raw slug requiring reorganization of loose files:
         python download_dataset.py --slug plantvillage/plantvillage-dataset --output data/raw --reorganize
         ```
         - Alternatively use helper script (handles download + optional reorg):
   
4. Run the notebook:

```powershell
jupyter notebook crop_disease_classification.ipynb
```
5. After training, launch Gradio app:

```powershell
python app.py
```



## Model

- Uses torchvision's `efficientnet_b0` or `mobilenet_v3_small` for speed & compact size.
- Fine-tunes final classification layer for selected disease classes.
- Data augmentations: RandomHorizontalFlip, RandomRotation, RandomResizedCrop, ColorJitter.

## Performance Summary

<!-- PERF_SUMMARY_START -->
| Model | Classes | Epochs | Best Epoch | Val Acc | Train Acc |
|-------|---------|--------|-----------|---------|-----------|
| MobileNetV3-S | 12 | 10 | 4 | 0.980 | 0.948 |

<!-- PERF_SUMMARY_END -->

## Evaluation Outputs

- Confusion matrix saved as `outputs/confusion_matrix.png` (create `outputs/` automatically).
- Sample correct & incorrect predictions displayed inline in notebook.

## Predictions Gallery

<!-- PRED_SAMPLES_START -->

**Correct Predictions (5)**
![](outputs/samples/correct_1.jpg)
![](outputs/samples/correct_2.jpg)
![](outputs/samples/correct_3.jpg)
![](outputs/samples/correct_4.jpg)
![](outputs/samples/correct_5.jpg)

**Incorrect Predictions (5)**
![](outputs/samples/incorrect_1.jpg)
![](outputs/samples/incorrect_2.jpg)
![](outputs/samples/incorrect_3.jpg)

<!-- PRED_SAMPLES_END -->

## Business Recommendation

Recommend deploying **MobileNetV3-Small** for Syngenta's farmer-facing mobile app.

- Accuracy: Achieves strong accuracy on PlantVillage subset with proper augmentation; sufficient for on-device triage.
- Speed: Optimized for edge inference; low latency on mid-range devices.
- Size: Compact (~2–3M params). Lower bandwidth and storage.

If final validation shows a consistent >2–3% accuracy gain for EfficientNet-B0 with acceptable latency on target devices, consider it; otherwise prioritize MobileNetV3-Small for better UX and device coverage.

## Reproducibility Notes

- Set random seeds in notebook for deterministic splits.
- Document selected classes in markdown cell.
- Save label mapping file `models/class_index.json` for app use.


