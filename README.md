# Tool Wear Detection in CNC Milling Using Spindle Current Signals

A statistical and anomaly detection pipeline for identifying tool wear in CNC milling operations by analysing spindle motor current (S1_CurrentFeedback) signals.

---

## Overview

This project investigates whether spindle current signals can reliably indicate tool wear in CNC milling. It addresses two research questions:

- **RQ1** — Are there statistically significant differences between spindle current signals of worn and unworn tools?
- **RQ2** — Can anomaly detection models learn normal spindle current behaviour and identify deviations associated with tool wear?

---

## Dataset

**Source:** [UC Berkeley SMART Lab — Tool Wear Detection in CNC Milling](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill) / provided as `CNC MILLING DATASET - UNIVERSITY OF MICHIGAN SMART LAB.zip`

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
├── CNC_Milling_Analysis.ipynb                              # Main analysis notebook
├── CNC MILLING DATASET - UNIVERSITY OF MICHIGAN SMART LAB.zip  # Raw dataset (extract before running)
│   ├── experiment_01.csv
│   ├── experiment_02.csv
│   └── ... (18 experiment files total)
└── cleaned_cnc_dataset.csv                                 # Generated during preprocessing
```

---

## Setup

1. Extract the ZIP file into the same directory as the notebook:
   ```bash
   unzip "CNC MILLING DATASET - UNIVERSITY OF MICHIGAN SMART LAB.zip"
   ```
   This will produce 18 individual experiment CSV files (`experiment_01.csv` through `experiment_18.csv`).

2. Install dependencies:
   ```bash
   pip install pandas numpy scipy scikit-learn matplotlib seaborn
   ```

### Dependencies

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

The 18 individual experiment CSV files extracted from the ZIP are loaded, cleaned, and merged into a single unified dataset (`cleaned_cnc_dataset.csv`). Each file is processed by the `clean_experiment()` function, which:

- Retains only the `S1_CurrentFeedback` (spindle current) and `Machining_Process` columns.
- Filters to active cutting stages only: `Layer 1 Up/Down`, `Layer 2 Up/Down`, `Layer 3 Up/Down`.
- Attaches metadata columns: `tool_condition` (worn/unworn), `group`, `feedrate`, and `clamp_pressure`.

All 12 experiments (2 per group × 6 groups) are concatenated into `cleaned_cnc_dataset.csv`, which serves as the input for all subsequent analysis.

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

Two complementary tests were applied to the window-based features to evaluate whether spindle current signals differ meaningfully between worn and unworn tools across all group and process layer combinations.

**Welch's t-test** was used to assess differences in mean spindle current between the two tool conditions. It was chosen over a standard t-test because it does not assume equal variances between groups, making it more appropriate here given that worn and unworn signals can differ in both their average level and their spread. A significant result indicates that tool wear shifts the average cutting load detectable in the signal.

**Levene's test** was applied to assess differences in signal variability (standard deviation) between tool conditions. Since milling signals are inherently non-stationary, variability is expected to be a more sensitive indicator of wear-induced instability than the mean alone. Levene's test was selected because it is robust to non-normal data distributions, which are common in real machining signals.

Both tests were run at a significance threshold of **p < 0.05**, and results were visualised as heatmaps across experimental groups and process layers to reveal whether significance is consistent or condition-dependent.

### 4. Anomaly Detection (RQ2)

Since labelled fault data is rarely available in industrial settings, an unsupervised anomaly detection approach was adopted. For each experimental group, unworn tool data was treated as representing normal machine behaviour, and worn tool data as anomalies. This reflects a realistic monitoring scenario where a model is trained on normal operation and used to flag deviations.

The unworn data was split 70/30, with 70% used for training and the remaining 30% combined with all worn tool windows to form a mixed test set. This ensures the model never sees worn data during training, and that the test set reflects real conditions where both normal and anomalous behaviour can occur.

Three unsupervised models were implemented to capture anomalies from different perspectives:

| Model                | Detection Approach                             | Why Selected                                                              |
|----------------------|------------------------------------------------|---------------------------------------------------------------------------|
| Isolation Forest     | Isolates globally distinct observations        | Effective for high-dimensional data; computationally efficient            |
| One-Class SVM        | Learns a boundary around normal data           | Well-suited for learning a tight decision boundary from normal samples    |
| Local Outlier Factor | Detects anomalies via local density difference | Captures localised deviations that global models may miss                 |

Using three models with different detection strategies allows for a more robust evaluation — if performance trends are consistent across all three, the findings are likely driven by the data rather than by any one model's assumptions.

Models were evaluated using Accuracy, Precision, Recall, and F1 Score, with **F1 Score as the primary metric** because it balances precision and recall, which is important given the class imbalance between normal and anomalous windows in the test set.

To understand what drives model performance, feature separability was quantified as the absolute difference in mean and standard deviation between worn and unworn window distributions per group. Pearson correlation was then computed between these separability values and the average F1 score per group, to determine whether groups with more distinct signal distributions are inherently easier for models to detect — regardless of which model is used.

---

## Results and Analysis

### RQ1 — Statistical Differences in Spindle Current Between Worn and Unworn Tools

#### Feature-Based Trend Analysis

Analysis of mean spindle current across groups shows that under low-load conditions (Groups 1–3), worn and unworn tools exhibit minimal differences, indicating that tool wear effects are not clearly reflected in average current. As machining load increases, differences become more noticeable, however, these changes are inconsistent in direction, with both increases and decreases observed. This highlights that mean current does not provide a stable relationship with tool wear because it is strongly influenced by machining parameters, meaning wear-related effects are only observable under sufficiently high-load conditions.

In contrast, signal variability demonstrates a clearer and more consistent pattern. While variability remains similar between tool conditions in Groups 1–3, it increases substantially for worn tools in higher-load conditions (Groups 5–6). This indicates that tool wear primarily manifests as increased instability in the cutting process, leading to greater signal fluctuations rather than consistent changes in average load.

#### Statistical Significance Testing

**Welch's t-test (Mean Differences):** For Groups 1–4, results are predominantly non-significant (p > 0.05), indicating that mean spindle current does not reliably distinguish between tool conditions under lower feedrate settings. Although some isolated significant results appear in Groups 2–3, these are inconsistent across process layers and do not form a stable pattern. In contrast, Group 5 exhibits consistent statistical significance (p < 0.05), indicating that differences in mean current become detectable only under higher machining loads.

**Levene's Test (Variance Differences):** A similar pattern is observed for Groups 1–4, where results are predominantly non-significant (p > 0.05), suggesting comparable signal stability between tool conditions. However, in Groups 5 and 6, significant differences (p < 0.05) are consistently observed across process layers, confirming that worn tools exhibit higher variability under more aggressive machining conditions.

#### Visual Validation

Raw spindle current signals were examined for Layer 2 Up, comparing Groups 1 and 5 as representative cases of low and high machining load. In Group 1, worn and unworn signals exhibit highly similar patterns with overlapping amplitudes. In contrast, Group 5 shows clear divergence — the worn tool signal displays increased fluctuations, irregular peaks, and greater instability, directly corresponding to the statistically significant differences observed in variability.

#### Answer to RQ1

Statistically significant differences do exist between worn and unworn spindle current signals, but their detectability depends on machining conditions and their reliability depends on the feature used. Signal variability is a more reliable and consistent indicator than mean current, particularly under high-load conditions. Spindle current can distinguish tool conditions only when appropriate features are applied under suitable operating conditions.

---

### RQ2 — Anomaly Detection of Tool Wear from Spindle Current Signals

#### Model Performance Across Experimental Groups

In Groups 1–3, all three models performed poorly, with F1 scores generally below 0.3. This was primarily driven by extremely low recall in Isolation Forest and LOF — while detected anomalies were reliable (high precision), most worn tool instances were not identified. OC-SVM showed relatively better and more balanced performance in these low-load groups.

From Group 4 onwards, performance improved consistently, with a substantial increase in Groups 5 and 6. OC-SVM achieved the highest performance, reaching F1 = 0.86 in Group 5 and a perfect F1 = 1.00 in Group 6. These results indicate that models become more effective at learning normal current behaviour and identifying wear-related deviations as machining load increases.

| Group | Isolation Forest (F1) | One-Class SVM (F1) | Local Outlier Factor (F1) |
|-------|-----------------------|--------------------|---------------------------|
| 1     | 0.13                  | 0.60               | 0.10                      |
| 2     | 0.12                  | 0.29               | 0.06                      |
| 3     | 0.07                  | 0.14               | 0.75                      |
| 4     | 0.25                  | 0.43               | 0.00                      |
| 5     | 0.71                  | 0.86               | 0.47                      |
| 6     | 0.44                  | **1.00**           | 0.25                      |

#### Anomaly Score Distribution Analysis

In Groups 1–2, significant overlap between worn and unworn anomaly score distributions across all models explains the low detection performance — deviations from normal behaviour are not clearly distinguishable. From Group 3 onwards, partial separation begins to emerge. In Groups 5–6, clear distributional shifts are observed, with worn samples consistently assigned lower anomaly scores, indicating they are effectively identified as anomalous. This confirms that while models can learn baseline behaviour, their ability to detect anomalies depends on the magnitude of deviation from that baseline.

#### Feature Separability as a Driver of Performance

Variability (std) exhibits a gradual and consistent increase in separability across all groups, with small shifts observable even in earlier groups — indicating sensitivity to early-stage wear. Mean current, by contrast, shows strong separation only in Groups 5–6, where its condition-dependent nature is amplified by higher machining loads.

Pearson correlation analysis confirms that standard deviation separation has a stronger relationship with model F1 score (r = **0.93**) than mean separation (r = **0.71**). This demonstrates that variability-based features are the primary driver of detection performance, while mean current provides complementary information only under more extreme machining conditions.

#### Answer to RQ2

Anomaly detection models can learn normal spindle current behaviour and identify tool wear, but their effectiveness depends more on feature separability and machining conditions than on model architecture. Under low-load conditions, signal overlap limits all models. As conditions become more aggressive, clearer feature separation leads to significantly improved detection. OC-SVM consistently outperformed the other models across groups.

---

## Usage

1. Extract the ZIP file and install dependencies as described in the [Setup](#setup) section above.
2. Run the notebook cells sequentially:
   - **Data Preprocessing** — loads and merges all experiment CSVs into `cleaned_cnc_dataset.csv`
   - **Data Cleaning** — quality checks and feature extraction
   - **Analysis: RQ1** — statistical tests and visualisations
   - **Analysis: RQ2** — anomaly detection modelling and evaluation

---

## Acknowledgements

This project was conducted in collaboration with **TrackMyMachine**, a Sheffield-based Industrial IoT provider, as part of a study on scalable spindle current monitoring for predictive maintenance.

Ethical approval for this project was granted by the **University of Sheffield**. The study was confirmed to involve only existing, robustly anonymised data, raising no concerns regarding privacy or data sensitivity.
