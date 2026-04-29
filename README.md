# Tool Wear Detection in CNC Milling Using Spindle Current Signals

A statistical and anomaly detection pipeline for identifying tool wear in CNC milling operations by analysing spindle motor current (S1_CurrentFeedback) signals.

---

## Overview

This project investigates whether spindle current signals can reliably indicate tool wear in CNC milling. It addresses two research questions:

- **RQ1** — Are there statistically significant differences between spindle current signals of worn and unworn tools?
- **RQ2** — Can unsupervised anomaly detection models learn normal spindle current behaviour and identify deviations associated with tool wear?

---

## Dataset

**Source:** [UC Berkeley SMART Lab — Tool Wear Detection in CNC Milling](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill)

The dataset contains 18 CNC milling experiments performed on wax workpieces under varying feed rates and clamp pressures. Experiments are split into 8 unworn and 10 worn tool runs, with time-series data recorded at 100 ms intervals.

### Experimental Groups

| Group | Feedrate | Clamp Pressure | Experiments (unworn / worn)            |
|-------|----------|----------------|----------------------------------------|
| 1     | 3        | 2.5            | experiment_17 / experiment_18          |
| 2     | 3        | 3.0            | experiment_12 / experiment_14          |
| 3     | 3        | 4.0            | experiment_11 / experiment_13          |
| 4     | 6        | 3.0            | experiment_03 / experiment_15          |
| 5     | 6        | 4.0            | experiment_01 / experiment_06          |
| 6     | 20       | 4.0            | experiment_02 / experiment_08          |

Only active cutting stages are retained: `Layer 1 Up/Down`, `Layer 2 Up/Down`, `Layer 3 Up/Down`.

---

## Project Structure

```
.
├── CNC_Milling_Analysis.ipynb   # Main analysis notebook
├── experiment_01.csv            # Raw experiment files (18 total)
├── ...
├── experiment_18.csv
└── cleaned_cnc_dataset.csv      # Generated during preprocessing
```

---

## Requirements

Install dependencies with:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

| Library      | Purpose                                      |
|--------------|----------------------------------------------|
| pandas       | Data loading, cleaning, and manipulation     |
| numpy        | Numerical computation and windowed features  |
| scipy        | Welch's t-test and Levene's test             |
| scikit-learn | Anomaly detection models and evaluation      |
| matplotlib   | Plotting                                     |
| seaborn      | Statistical visualisations and heatmaps      |

---

## Methodology

### 1. Data Preprocessing

- Raw CSV files are loaded and filtered to retain only the `S1_CurrentFeedback` and `Machining_Process` columns.
- Non-cutting stages (setup, repositioning) are removed.
- Metadata columns (`tool_condition`, `group`, `feedrate`, `clamp_pressure`) are added.
- All experiments are concatenated into a single `cleaned_cnc_dataset.csv`.

### 2. Feature Extraction

**Global features** are computed per experiment group and tool condition across the entire signal:

| Feature            | Description                      |
|--------------------|----------------------------------|
| `mean_current`     | Average cutting load             |
| `std_current`      | Signal variability               |
| `variance_current` | Dispersion of the signal         |
| `rms_current`      | Signal energy                    |
| `max_current`      | Peak load                        |
| `min_current`      | Minimum load                     |

**Window-based features** apply a sliding window over each group/process/tool combination:

| Parameter   | Value |
|-------------|-------|
| Window size | 50 samples |
| Step size   | 25 samples (50% overlap) |

The same six statistical features are computed per window, significantly increasing the number of observations for statistical testing.

### 3. Statistical Analysis (RQ1)

- **Welch's t-test** — tests for significant differences in mean spindle current between worn and unworn tools per group and process layer.
- **Levene's test** — tests for significant differences in signal variability between tool conditions.
- Significance threshold: **p < 0.05**
- Results are visualised as heatmaps across experimental groups and process layers.

### 4. Anomaly Detection (RQ2)

For each group, unworn tool data forms the normal class. The dataset is split as follows:

- **Training set:** 70% of unworn signal windows
- **Test set:** remaining 30% of unworn windows + all worn windows

Three unsupervised models are trained and evaluated:

| Model                | Detection Approach                             |
|----------------------|------------------------------------------------|
| Isolation Forest     | Isolates globally distinct observations        |
| One-Class SVM        | Learns a boundary around normal data           |
| Local Outlier Factor | Detects anomalies via local density difference |

**Evaluation metrics:** Accuracy, Precision, Recall, F1 Score (primary metric)

Feature separability (absolute difference in mean and std between worn/unworn distributions) is correlated with F1 score using Pearson correlation.

---

## Key Findings

- **Mean current** shows significant differences only under high-load conditions (Group 5+), and its direction is inconsistent across groups.
- **Signal variability (std)** is a more reliable and consistent indicator of tool wear, increasing progressively with machining load.
- **Anomaly detection performance** is strongly condition-dependent. All models perform poorly in Groups 1–3 (F1 < 0.3) but improve substantially in Groups 5–6 (OC-SVM achieves F1 = 1.00 in Group 6).
- Pearson correlation between feature separability and F1 score: **0.71 (mean)** and **0.93 (std)**, confirming that variability drives detection performance more than model choice.

---

## Usage

1. Place all raw experiment CSV files in the same directory as the notebook.
2. Run the notebook cells sequentially:
   - **Data Preprocessing** — generates `cleaned_cnc_dataset.csv`
   - **Data Cleaning** — quality checks and feature extraction
   - **Analysis: RQ1** — statistical tests and visualisations
   - **Analysis: RQ2** — anomaly detection modelling and evaluation

---

## Acknowledgements

This project was conducted in collaboration with **TrackMyMachine**, a Sheffield-based Industrial IoT provider, as part of a study on scalable spindle current monitoring for predictive maintenance.
