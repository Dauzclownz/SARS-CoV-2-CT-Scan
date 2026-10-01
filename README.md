# Beyond single-split accuracy: near-perfect scores on a COVID-19 CT benchmark do not transfer across cohorts

Code and figures for the manuscript of the same title.

Reported accuracies on the public SARS-CoV-2 CT-scan benchmark approach 99%, but almost all of them come
from a single slice-level split of one 120-patient cohort. This repository applies three inexpensive checks
to that benchmark:

1. pairwise McNemar testing repeated over five random seeds,
2. patient-disjoint (subject-disjoint) splitting, and
3. external validation on two independent cohorts without fine-tuning.

The checks are applied to a reproduced prototype-based explainable deep neural network (xDNN), ViT-Small,
Swin-Small, ConvNeXt-Small and their soft-voting ensemble.

## Repository contents

| File | Purpose |
| --- | --- |
| `Beyond_single_split_accuracy_SARS_CoV2_CT.ipynb` | Main notebook for Google Colab. It reproduces Tables 1-5 and 7 and Figures 1-6. It also runs the five-seed McNemar analysis behind Table 3 and the bootstrap confidence intervals reported in Section 4.6. |
| `Beyond_single_split_accuracy_SARS_CoV2_CT.py` | Plain-text export of the notebook for reading and diffing. It contains Colab shell commands (`!pip`, `files.upload()`), so run the notebook rather than this file. |
| `patient_level_evaluation.py` | Stand-alone script for Table 6 (Section 4.8). It compares slice-level and patient-level evaluation on the multiclass CT dataset. |
| `patient_level_evaluation_colab.ipynb` | Google Colab notebook that runs `patient_level_evaluation.py`. It keeps results and the checkpoint on Google Drive and compares the new numbers with Table 6 of the manuscript. |
| `img/` | Figure files produced by the notebook (`figure1_framework.png` to `figure6_gradcam.png`). In the manuscript, the titles drawn inside Figures 1, 3 and 5 are removed to follow the journal's artwork guidelines; the panels are otherwise identical. |

## Requirements

- All reported results were produced on Google Colab with an NVIDIA T4 GPU. Other CUDA GPUs work, but numbers can differ slightly (see [Reproducibility](#reproducibility)).
- Python 3.10 or later and the following packages:

  ```bash
  pip install torch torchvision "timm>=1.0.0" scikit-learn statsmodels scipy numpy pandas pillow \
              grad-cam py7zr nibabel cairosvg fvcore kaggle
  ```

  `patient_level_evaluation.py` needs only `torch`, `torchvision`, `timm`, `scikit-learn`, `statsmodels`, `scipy`,
  `numpy`, `pandas`, `pillow`, plus `kaggle` if it should download the data.
- A Kaggle API token (`kaggle.json`, created in your Kaggle account settings under API > Create New Token) to download the Kaggle-hosted datasets.

## Datasets

None of the datasets are redistributed here. They are downloaded from their original sources, and each remains
subject to the terms of that source. Please cite the original publications (see [Citation](#citation)).

| Dataset | Used for | Source | Downloaded by |
| --- | --- | --- | --- |
| SARS-CoV-2 CT-scan dataset: 2,482 slices from 120 patients [1] | Training and in-domain evaluation | Kaggle `plameneduardo/sarscov2-ctscan-dataset` | Notebook, Section 1 |
| Original xDNN features and code [2] | xDNN reproduction with the native VGG-16 features | GitHub `Plamen-Eduardo/xDNN-SARS-CoV-2-CT-Scan` | Notebook, Section 1 |
| UCSD COVID-CT: 349 COVID and 397 non-COVID images [4] | External validation (Table 4, Figures 4-6) | GitHub mirror `desaisrkr/https-github.com-UCSD-AI4H-COVID-CT` (the original UCSD-AI4H repository is offline) | Notebook, Section 1 |
| MosMedData [5] | External validation at the volume level (Table 5) | Kaggle `mathurinache/mosmeddata-chest-ct-scans-with-covid19` | Notebook, Section 1 |
| Multiclass CT dataset: 4,173 scans from 210 patients [3]; the Kaggle release holds 4,171 slices | Patient-level evaluation (Table 6) | Kaggle `plameneduardo/a-covid-multiclass-dataset-of-ct-scans`, or Synapse `syn22174850` | `patient_level_evaluation.py --download` |

## Reproducing Tables 1-5 and 7 and Figures 1-6 (notebook)

1. Open `Beyond_single_split_accuracy_SARS_CoV2_CT.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Select **Runtime > Run all**. When Section 1 asks for it, upload `kaggle.json`. All datasets for these results are then downloaded automatically.
4. Sections 1 to 5 (setup, data, utilities and external loaders) must run first. After that, each table or figure section is self-contained and reuses the cached models.

| Manuscript output | Notebook section |
| --- | --- |
| xDNN reproduction (published F1 97.31%) | 6 (Part A) |
| Table 1 | 7 |
| Table 2 | 8 (resumable) |
| Table 3 | 9b (five seeds; Section 9 runs a single seed) |
| Table 4 | 10 |
| Table 5 | 11, and 11b for the xDNN row |
| 95% bootstrap confidence intervals (Section 4.6) | 12b |
| Figure 1 | 12 |
| Figure 2 | 13 |
| Figure 3 | 14 |
| Figures 4-6 | 15 |
| Table 7 | 16 |
| Table 6 | `patient_level_evaluation_colab.ipynb` or `patient_level_evaluation.py`, described below (Section 17 of the notebook points to it) |

**Runtime on a T4.** The three deep models are trained once in Section 7 and reused by Tables 1, 3, 4 and 5 and by Figures 2 and 4-6.
- Table 2 (five-fold cross-validation) takes about 2-3 hours. Re-running the cell after a disconnect resumes from the last finished fold.
- Section 9b (four additional seeds) takes about 25 minutes.
- Every other section takes minutes.

## Reproducing Table 6 (`patient_level_evaluation.py`)

Table 6 measures slice-level leakage on the multiclass CT dataset of Soares et al. [3], collapsed to COVID versus
non-COVID, where non-COVID comprises healthy lungs and other pulmonary conditions. Dataset, task and training
pipeline are held fixed, and only the split changes:

- **Slice-level split.** One stratified 80/20 split of the slices, so slices of one patient can fall on both sides.
- **Patient-level evaluation.** Five-fold cross-validation with `StratifiedGroupKFold` on the patient identifiers, so no patient appears in both training and test data. Fold 1 also serves as the single patient-disjoint 80/20 split discussed in Section 4.8.

The script evaluates xDNN on re-extracted torchvision VGG-16 features, ViT-Small, Swin-Small, ConvNeXt-Small and
their soft-voting ensemble. Inside every fold it runs pairwise McNemar tests with a Bonferroni correction, as
for Table 3.

The preprocessing, training function, test-time augmentation, metrics, VGG-16 encoder and xDNN classifier are
copied from Sections 2-4 of the notebook with the same settings. The same number of epochs (default 8) is used
for every split.

**On Google Colab**, open `patient_level_evaluation_colab.ipynb`, select a T4 GPU and run all cells. The notebook:
- downloads the data and shows its folder layout;
- runs an optional 5-minute end-to-end check, then the full evaluation;
- writes results and the checkpoint to Google Drive, so a dropped session can be resumed by running all cells again;
- prints the new Table 6 next to the manuscript values.

**From the command line:**

```bash
# any machine with kaggle.json in ~/.kaggle: download the data, then run
python patient_level_evaluation.py --download --data_dir /content/multiclass

# with a copy that is already extracted (Kaggle or Synapse)
python patient_level_evaluation.py --data_dir /path/to/multiclass

# quick functional check on a few patients (minutes; not a result)
python patient_level_evaluation.py --data_dir /path/to/multiclass \
    --subsample_patients 6 --epochs 1 --out_dir results_smoke
```

**Patient identifiers.**
- **Default.** The script finds the folder that holds the class folders (COVID, healthy and other conditions; names are matched case-insensitively). It then treats the first sub-folder inside each class folder as the patient.
- **Images directly in the class folders.** The patient is read from the file name with `--patient_regex`. The default pattern matches names such as `Patient_12_slice_3.png`.
- **Explicit index.** Supply `--index_csv` with the columns `path,label,patient`.

The script stops with a message if any image lacks a patient identifier. It also prints the number of slices and
patients per class, so the expected 4,171 slices (Kaggle release) from 210 patients can be checked before training starts.

**Main options.**

| Option | Default | Meaning |
| --- | --- | --- |
| `--data_dir` | (required) | Root of the extracted dataset |
| `--download` | off | Download `plameneduardo/a-covid-multiclass-dataset-of-ct-scans` with the Kaggle API first |
| `--epochs` | 8 | Training epochs, identical for every split |
| `--folds` | 5 | Number of patient-level folds |
| `--seed` | 42 | Split seed. Model seeds follow the notebook: 42 for the single split and 42 + 10 x model + fold for cross-validation |
| `--models` | `xdnn,vit,swin,convnext` | Subset of methods; the ensemble is reported when all three deep models are run |
| `--model_size` | `small` | `small` as in the manuscript, or `base` |
| `--out_dir` | `results_patient_level` | Output folder |
| `--fresh` | off | Ignore an existing checkpoint |

**Outputs** (in `--out_dir`).

| File | Content |
| --- | --- |
| `table6.md`, `table6.csv` | Table 6: F1 and AUC on the slice-level split, mean ± SD over the patient-level folds, and their differences |
| `per_split_metrics.csv` | Accuracy, precision, recall, F1, AUC and kappa for every method on the slice split and on each fold |
| `mcnemar_per_fold.csv`, `mcnemar_summary.csv` | McNemar *p* values per fold, and the number of folds below the Bonferroni threshold |
| `pairwise_f1_differences.csv` | Mean fold-wise F1 difference between methods, and the folds in which each one is ahead |
| `dataset_index.csv` | Every slice with its label, patient, slice-split assignment and patient fold |
| `run_info.json` | Library versions, device, arguments, and how many test patients of the slice split also appear in training |
| `checkpoint.pkl`, `vgg16_features.npy` | Resume state and cached features |

**Runtime.** The full run consists of 18 trainings (three models on the slice split and five folds) and six xDNN fits.
- The run behind the manuscript took about 2.2 hours on a T4 (Google Colab).
- Every finished training is checkpointed, so re-running the same command resumes after a disconnect.
- A checkpoint made with a different dataset or configuration is refused rather than mixed in.

## Reproducibility

- Deterministic cuDNN and cuBLAS settings are enabled in both the notebook and the script, and all seeds are fixed. Even so, numbers can differ slightly between GPU types, driver and library versions, and repeated runs. The manuscript quantifies this run-to-run spread with five-fold cross-validation (Table 2) and five-seed McNemar testing (Table 3).
- Keep `MODEL_SIZE = 'small'` in the notebook and `--model_size small` in the script for the reported results.
- Tables 1 and 2 report xDNN with the native VGG-16 features released with the dataset; Table 1 also shows the re-extracted variant. Tables 3-6 use features re-extracted with torchvision VGG-16, because paired tests and external evaluation need predictions from one shared pipeline.
- Every figure is exported at 600 dpi for its printed width, with a vector PDF alongside each PNG.

## Citation

Citation details for the article will be added once it is published. Please also cite the original sources of the
data and of xDNN:

1. Soares E, Angelov P, Biaso S, Froes MH, Abe DK (2020) SARS-CoV-2 CT-scan dataset: a large dataset of real patients CT scans for SARS-CoV-2 identification. medRxiv. https://doi.org/10.1101/2020.04.24.20078584
2. Angelov P, Soares E (2020) Towards explainable deep neural networks (xDNN). Neural Netw 130:185-194. https://doi.org/10.1016/j.neunet.2020.07.010
3. Soares E, Angelov P, Biaso S, Cury M, Abe D (2024) A large multiclass dataset of CT scans for COVID-19 identification. Evol Syst 15:635-640. https://doi.org/10.1007/s12530-023-09511-2
4. Yang X, He X, Zhao J, Zhang Y, Zhang S, Xie P (2020) COVID-CT-dataset: a CT scan dataset about COVID-19. arXiv:2003.13865. https://doi.org/10.48550/arXiv.2003.13865
5. Morozov SP, Andreychenko AE, Pavlov NA, Vladzymyrskyy AV, Ledikhova NV, Gombolevskiy VA et al (2020) MosMedData: chest CT scans with COVID-19 related findings dataset. medRxiv. https://doi.org/10.1101/2020.05.20.20100362

## Contact

Muhammad Nur Firdaus, Faculty of Information Technology, Universitas Nusa Mandiri, Depok, Indonesia
(14250045@nusamandiri.ac.id).
