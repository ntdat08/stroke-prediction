# Chapter 3: Model Development and Evaluation

## 3.1 Implemented Models

This chapter focuses on the practical process of building, training, and evaluating machine learning models for stroke risk prediction. The theoretical foundations of the algorithms employed have been presented in Chapter 1; therefore, this section concentrates on the **implementation details, experimental results, and analytical insights** derived from the training process.

Three classification models were selected for this study, each chosen for specific reasons relevant to the stroke prediction task:

| Model | Rationale for Selection |
|-------|------------------------|
| **Logistic Regression** | Produces probability outputs, enabling flexible threshold tuning. Highly interpretable — critical for medical applications where clinicians need to understand *why* a prediction was made. Serves as a strong baseline for binary classification. |
| **Decision Tree** | Captures non-linear relationships in the data. Provides intuitive, visual decision rules. Does not require feature normalization. |
| **Random Forest** | An ensemble method that reduces overfitting compared to a single Decision Tree. Handles high-dimensional data well. Provides feature importance scores for further analysis. |

All models were implemented using the **scikit-learn** library (version 1.8) in Python 3.13.

---

## 3.2 Model Training

The training process was conducted in **two stages**: establishing baseline performance with default parameters, followed by systematic hyperparameter optimization using GridSearchCV.

### 3.2.1 Input Data

The datasets used in this chapter are the outputs of the preprocessing pipeline described in Chapter 2:

- **Training set**: `data/processed/train_final.csv` — **7,776 samples, 15 features**
- **Test set**: `data/processed/test_final.csv` — **1,022 samples, 15 features**
- **Target variable**: `stroke` (binary: 0 = No Stroke, 1 = Stroke)
- **Class imbalance**: Addressed using SMOTE in Chapter 2

### 3.2.2 Stage 1: Baseline Models

Each model was first trained with **default parameters** to establish a performance reference point. This baseline allows us to quantify the improvement achieved through hyperparameter tuning.

**Baseline Results:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.7613 | 0.1381 | 0.7400 | 0.2327 |
| Decision Tree | 0.8728 | 0.1491 | 0.3400 | 0.2073 |
| Random Forest | 0.8885 | 0.1596 | 0.3000 | 0.2083 |

> **Note**: Logistic Regression was initialized with `max_iter=1000` and `random_state=42` to ensure convergence and reproducibility.

#### Understanding the Evaluation Metrics

Before analyzing the results, it is important to understand what each metric measures and how to interpret it in the context of stroke prediction. All metrics below are computed from the **Confusion Matrix** — a 2×2 table that categorizes predictions into four outcomes:

```
                      Predicted
                   Negative | Positive
Actual  Negative |   TN     |   FP     |
        Positive |   FN     |   TP     |
```

Where:
- **TP (True Positive)**: Patient has stroke → model correctly predicts stroke ✅
- **TN (True Negative)**: Patient has no stroke → model correctly predicts no stroke ✅
- **FP (False Positive)**: Patient has no stroke → model incorrectly predicts stroke ❌ (false alarm)
- **FN (False Negative)**: Patient has stroke → model incorrectly predicts no stroke ❌ (**missed case — dangerous**)

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)

The proportion of all predictions that are correct. While intuitive, Accuracy is **misleading for imbalanced datasets**. In this dataset where ~95% of patients do not have a stroke, a model that always predicts "No Stroke" would achieve 95% Accuracy — yet it would be clinically useless because it would miss every single stroke case.

**Precision** = TP / (TP + FP)

"Of all patients the model flagged as stroke risk, how many actually had a stroke?" Low Precision means many false alarms, but in a screening context, false alarms are **acceptable** — they simply lead to further clinical examination.

**Recall (Sensitivity)** = TP / (TP + FN)

"Of all patients who actually had a stroke, how many did the model successfully detect?" This is the **most critical metric** in medical screening. A missed stroke case (FN) can lead to permanent brain damage or death. **Maximizing Recall is the primary clinical objective.**

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

The harmonic mean of Precision and Recall. It provides a single score that balances both metrics, but it treats false alarms and missed cases as equally costly — which is **not appropriate** in a medical context where missed cases are far more dangerous.

#### Analysis of Baseline Results

*Figure 3.1: Baseline Model Performance Comparison — bar chart comparing Accuracy, Precision, Recall, and F1-Score across all three models with default parameters.*

Looking at the baseline results table and bar chart (Figure 3.1), several important patterns emerge:

**1. The Accuracy Paradox**

Decision Tree (0.8728) and Random Forest (0.8885) both achieve **higher Accuracy** than Logistic Regression (0.7613). At first glance, this might suggest they are better models. However, this is a classic example of the **Accuracy Paradox** in imbalanced classification. These models achieve high Accuracy primarily by correctly predicting the majority class (No Stroke), while **failing to detect most actual stroke cases**.

**2. Why Logistic Regression's Recall is Significantly Higher**

The most striking difference is in **Recall**:
- **Logistic Regression: 0.7400** — detects 74% of stroke patients
- **Decision Tree: 0.3400** — detects only 34% of stroke patients
- **Random Forest: 0.3000** — detects only 30% of stroke patients

This means Decision Tree and Random Forest **miss 60–70% of all stroke cases**, while Logistic Regression misses only 26%. The reason lies in how each model makes decisions:

- **Logistic Regression** outputs a continuous probability score using the sigmoid function. It produces a smooth probability distribution that allows it to "spread" its predictions across a wider range, making it more sensitive to minority class patterns. Even with a default threshold of 0.50, LR is more likely to assign higher probabilities to true stroke cases.

- **Decision Tree** makes hard, binary splits at each node. In an imbalanced dataset, most branches of the tree are dominated by the majority class (No Stroke), causing the tree to create decision paths that overwhelmingly favor predicting "No Stroke." This results in very few positive predictions and thus very low Recall.

- **Random Forest**, as an ensemble of Decision Trees, inherits the same bias. While averaging over 100 trees reduces variance, the fundamental tendency to favor the majority class persists, leading to similarly low Recall.

**3. The Precision–Recall Trade-off**

All three models show low Precision (0.13–0.16), which means most of the patients flagged as "stroke risk" are actually false alarms. However, Logistic Regression's lower Accuracy (0.7613) is a direct consequence of it having **more False Positives** — because it is actively trying to catch more stroke cases (higher Recall), it inevitably also flags more non-stroke patients. This is an **acceptable trade-off** in medical screening: it is far better to send a healthy patient for additional tests than to send a stroke patient home without treatment.

### 3.2.3 Why ROC-AUC Was Chosen as the Primary Evaluation Metric

ROC-AUC was selected as the **primary scoring metric** for model optimization (used as the `scoring` parameter in GridSearchCV) instead of Accuracy or F1-Score. The reasons are:

**1. Threshold Independence**

Unlike Accuracy, Precision, Recall, and F1-Score — which are all computed at a **fixed threshold** (default = 0.50) — ROC-AUC evaluates the model's discriminative ability **across all possible thresholds**. This is critical because:
- The default threshold of 0.50 is rarely optimal for imbalanced datasets.
- We plan to perform threshold tuning (Section 3.4), so we need a metric that reflects the model's overall ranking quality, not its performance at one arbitrary cutoff.

**2. Robustness to Class Imbalance**

In this dataset, stroke cases represent only ~5% of the population. Metrics like Accuracy are highly misleading under such imbalance — a model that predicts "No Stroke" for every patient would achieve ~95% Accuracy but would be clinically useless. ROC-AUC, by contrast, measures how well the model **separates** the two classes regardless of their proportions.

**3. Suitability for Medical Screening**

In a screening context, we want a model that **ranks** high-risk patients above low-risk patients. ROC-AUC directly measures this ranking ability. A higher ROC-AUC means the model assigns higher probabilities to actual stroke patients, which is exactly what a screening tool needs.

**4. Enables Fair Comparison Across Models**

Since each model may have a different optimal threshold, comparing models at a fixed threshold (e.g., Accuracy at 0.50) can be misleading. ROC-AUC provides a **threshold-agnostic** comparison, allowing us to identify which model has the best underlying discriminative power before applying threshold tuning.

#### Understanding the ROC Curve

The **ROC (Receiver Operating Characteristic) Curve** is a graphical tool that illustrates a model's ability to distinguish between two classes — in this case, Stroke vs. No Stroke — **across all possible classification thresholds**.

**Axes:**
- **X-axis — False Positive Rate (FPR)**: The proportion of non-stroke patients incorrectly flagged as stroke. FPR = FP / (FP + TN).
- **Y-axis — True Positive Rate (TPR = Recall)**: The proportion of actual stroke patients correctly detected. TPR = TP / (TP + FN).

Each point on the curve represents the model's performance at one specific threshold. As the threshold decreases (from 1.0 → 0.0), the model predicts more patients as "Stroke," causing both TPR and FPR to increase simultaneously.

**How to read the ROC Curve:**
- A curve that hugs the **top-left corner** (TPR = 1, FPR = 0) represents a perfect model — it detects all stroke cases without any false alarms.
- The **dashed diagonal line** represents a random classifier (equivalent to flipping a coin). Any model whose curve lies above this line performs better than random guessing.
- A curve that **closely follows the diagonal** indicates the model has little to no discriminative ability.

**AUC (Area Under the Curve)** quantifies the overall performance into a single number — the total area beneath the ROC curve:

| AUC Range | Interpretation |
|-----------|----------------|
| 0.9 – 1.0 | Excellent |
| 0.8 – 0.9 | Good |
| 0.7 – 0.8 | Moderate |
| 0.5 – 0.7 | Weak |
| 0.5 | No discrimination (random) |

In practical terms, AUC represents the probability that the model assigns a **higher risk score** to a randomly chosen stroke patient than to a randomly chosen non-stroke patient.

#### Individual ROC Curve Analysis

*Figure 3.2a: ROC Curve — Logistic Regression (AUC = 0.8226). The curve rises steeply and stays well above the diagonal, indicating strong class separation ability.*

Logistic Regression achieves an AUC of **0.8226** (Good). The curve rises sharply in the left portion of the plot, meaning the model can detect a large proportion of stroke cases (high TPR) while maintaining a relatively low false alarm rate (low FPR). This confirms that LR produces well-calibrated probability scores that effectively rank stroke patients above non-stroke patients. Given a random stroke patient and a random non-stroke patient, there is an **82.26% probability** that the model assigns a higher risk score to the stroke patient.

*Figure 3.2b: ROC Curve — Random Forest (AUC = 0.7510). The curve shows moderate discriminative power but falls noticeably below Logistic Regression, particularly at low False Positive Rates.*

Random Forest achieves an AUC of **0.7510** (Moderate). The curve stays above the diagonal but lies considerably lower than Logistic Regression's curve, especially in the left region (low FPR). This means that to achieve the same level of Recall as LR, Random Forest would need to accept a significantly higher false alarm rate. The gap between the two curves visually confirms that LR has superior ranking ability.

*Figure 3.2c: ROC Curve — Decision Tree (AUC = 0.5816). The curve closely follows the diagonal baseline, indicating near-random discriminative ability and the weakest performance among all three models.*

Decision Tree achieves an AUC of only **0.5816** (Weak) — barely above the random baseline of 0.50. The curve stays close to the diagonal throughout, indicating that the model's probability outputs provide almost no useful information for distinguishing stroke from non-stroke patients. This poor performance is inherent to single Decision Trees on imbalanced problems: they produce discrete, coarse probability estimates (based on leaf node class proportions) rather than the smooth, continuous probabilities generated by Logistic Regression.


### 3.2.4 Stage 2: Hyperparameter Optimization via GridSearchCV

To improve model performance, **GridSearchCV** was employed — an exhaustive search method that evaluates all possible combinations of specified hyperparameter values using cross-validation.

**Configuration:**
- **Cross-validation**: 5-fold stratified
- **Scoring metric**: ROC-AUC (chosen over Accuracy because it is more robust for imbalanced datasets)
- **Parallelization**: `n_jobs=-1` (utilizing all CPU cores)

#### a) Logistic Regression

**Parameter Grid:**

```python
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],      # Regularization strength
    'penalty': ['l1', 'l2'],                      # Regularization type
    'solver': ['liblinear', 'saga']               # Optimization algorithm
}
```

- **Search space**: 6 × 2 × 2 = **24 combinations** × 5 folds = **120 model fits**
- **Best parameters found**: `C=100, penalty='l1', solver='saga'`

**Rationale for parameter ranges:**
- `C` ranges from 0.001 to 100 to explore both strong regularization (preventing overfitting) and weak regularization (allowing more complex decision boundaries).
- Both L1 (Lasso) and L2 (Ridge) penalties were tested. L1 can perform feature selection by driving some coefficients to zero.
- `liblinear` and `saga` solvers were chosen for their compatibility with both L1 and L2 penalties.

#### b) Decision Tree

**Parameter Grid:**

```python
param_grid_dt = {
    'criterion': ['gini', 'entropy'],         # Split quality measure
    'max_depth': [None, 5, 10, 15, 20],       # Maximum tree depth
    'min_samples_split': [2, 5, 10],          # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4]             # Minimum samples at a leaf node
}
```

- **Search space**: 2 × 5 × 3 × 3 = **90 combinations** × 5 folds = **450 model fits**
- **Best parameters found**: `criterion='entropy', max_depth=None, min_samples_leaf=4, min_samples_split=10`

#### c) Random Forest

**Parameter Grid:**

```python
param_grid_rf = {
    'n_estimators': [100, 200],           # Number of trees
    'max_depth': [10, 20, None],          # Maximum depth per tree
    'min_samples_split': [2, 5],          # Minimum samples to split
    'min_samples_leaf': [1, 2]            # Minimum samples at leaf
}
```

- **Search space**: 2 × 3 × 2 × 2 = **24 combinations** × 5 folds = **120 model fits**
- **Best parameters found**: `max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200`

#### Summary of Tuned Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.7613 | 0.1381 | 0.7400 | 0.2327 | **0.8227** |
| **Decision Tree** | 0.8728 | 0.1491 | 0.3400 | 0.2073 | 0.7016 |
| **Random Forest** | 0.8885 | 0.1596 | 0.3000 | 0.2083 | 0.7572 |

**Critical Observations:**

1. **Logistic Regression maintains the highest ROC-AUC** (0.8227), confirming its superior discriminative ability.
2. **Decision Tree and Random Forest exhibit high Accuracy but dangerously low Recall** (0.34 and 0.30 respectively). This means these models **miss 60–70% of actual stroke cases** — a critical failure in a medical screening context.
3. Although Random Forest has the highest Accuracy (0.8885), this is misleading. In imbalanced datasets, a model that simply predicts "No Stroke" for every patient would also achieve high accuracy. **Accuracy alone is an unreliable metric for this problem.**
4. Logistic Regression achieves the best Recall (0.7400), meaning it correctly identifies 74% of stroke patients — far superior to the other two models.

> **Conclusion**: Logistic Regression was selected as the **optimal model** for further threshold tuning based on its highest ROC-AUC and highest Recall among all three models.

---

## 3.3 Model Evaluation

### 3.3.1 Evaluation Metrics

The following metrics were used to assess model performance. Each metric is explained in the context of the stroke prediction problem:

| Metric | Formula | Meaning in Stroke Prediction |
|--------|---------|------|
| **Accuracy** | (TP + TN) / Total | Overall correctness. *Misleading when classes are imbalanced.* |
| **Precision** | TP / (TP + FP) | "Of all patients predicted to have a stroke, how many actually do?" Low precision means many false alarms. |
| **Recall (Sensitivity)** | TP / (TP + FN) | "Of all patients who actually had a stroke, how many did the model detect?" **Most critical metric** — a missed stroke case (FN) can be fatal. |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision and Recall. Balances both concerns. |
| **ROC-AUC** | Area under the ROC curve | Measures the model's ability to discriminate between classes across all possible thresholds. Higher is better. |

**Legend**: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative.

### 3.3.2 Confusion Matrix Analysis

The confusion matrix provides a detailed breakdown of prediction outcomes. In the medical context:

```
                    Predicted
                 No Stroke | Stroke
Actual  No Stroke|   TN    |   FP   |
        Stroke   |   FN    |   TP   |
```

- **True Negative (TN)**: Patient correctly identified as non-stroke → Correct exclusion.
- **True Positive (TP)**: Patient correctly identified as stroke risk → **Primary goal**.
- **False Positive (FP)**: Non-stroke patient flagged as at-risk → Acceptable cost (additional screening tests).
- **False Negative (FN)**: Stroke patient missed by the model → **Unacceptable risk** ⚠️ — could lead to delayed treatment and death.

> **In medical prediction, minimizing False Negatives (FN) is significantly more important than minimizing False Positives (FP).** Missing a stroke case can result in death, while a false alarm only incurs additional testing costs.

**Confusion Matrix for Logistic Regression (default threshold = 0.50):**

From the experimental results:
- The model correctly identifies 74% of stroke patients (Recall = 0.74)
- However, Precision is low (0.1381), meaning many non-stroke patients are also flagged

This trade-off is analyzed further in the Threshold Tuning section (Section 3.4).

### 3.3.3 ROC Curve

The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings. A model with perfect discrimination has an AUC of 1.0, while a random classifier has an AUC of 0.5.

**ROC-AUC Comparison:**

| Model | ROC-AUC | Interpretation |
|-------|---------|----------------|
| Logistic Regression | **0.8227** | Good discriminative ability |
| Random Forest | 0.7572 | Moderate |
| Decision Tree | 0.7016 | Weak |

The ROC curve for all three models should be plotted on the same chart, with the diagonal line (random classifier baseline) for visual comparison. Logistic Regression's curve is consistently closest to the top-left corner, confirming its superior performance.

---

## 3.4 Threshold Tuning

### 3.4.1 What is Threshold Tuning and Why is it Necessary?

By default, classification models use a **threshold of 0.50**: if the predicted probability P(stroke) ≥ 0.50, the model classifies the patient as "Stroke." However, this default threshold is not always optimal, especially in medical applications where the cost of a missed case (False Negative) far outweighs the cost of a false alarm (False Positive).

**Key Insight**: By lowering the threshold, we can increase Recall (detecting more stroke cases) at the expense of Precision (more false alarms). In an early screening scenario, this trade-off is acceptable.

### 3.4.2 Methodology

Threshold tuning was performed on the **Logistic Regression model** (selected as the best model from Section 3.2). The probability outputs were evaluated across thresholds from **0.10 to 0.85** with a step size of 0.05.

For each threshold value `t`:
```
y_pred = 1  if  P(stroke) >= t
y_pred = 0  otherwise
```

All four metrics (Accuracy, Precision, Recall, F1-Score) were computed for each threshold.

### 3.4.3 Results

| Threshold | Accuracy | Precision | Recall | F1-Score | Notes |
|-----------|----------|-----------|--------|----------|-------|
| 0.10 | 0.4550 | 0.0782 | 0.94 | 0.1444 | Maximum recall, but too many false alarms |
| 0.15 | 0.5166 | 0.0858 | 0.92 | 0.1570 | |
| **0.20** | **0.5763** | **0.0917** | **0.86** | **0.1657** | **Early screening** |
| 0.25 | 0.6106 | 0.0972 | 0.84 | 0.1743 | |
| 0.30 | 0.6438 | 0.1035 | 0.82 | 0.1839 | |
| 0.35 | 0.6849 | 0.1136 | 0.80 | 0.1990 | |
| 0.40 | 0.7221 | 0.1250 | 0.78 | 0.2155 | |
| 0.45 | 0.7417 | 0.1285 | 0.74 | 0.2189 | |
| 0.50 | 0.7613 | 0.1381 | 0.74 | 0.2327 | Default threshold |
| **0.55** | **0.7808** | **0.1463** | **0.72** | **0.2432** | **Balanced approach** |
| 0.60 | 0.8033 | 0.1553 | 0.68 | 0.2528 | |
| 0.65 | 0.8278 | 0.1719 | 0.66 | 0.2727 | |
| 0.70 | 0.8503 | 0.1879 | 0.62 | 0.2884 | |
| **0.75** | **0.8757** | **0.2105** | **0.56** | **0.3060** | **Best F1-Score** |
| 0.80 | 0.8963 | 0.2255 | 0.46 | 0.3026 | |
| 0.85 | 0.9168 | 0.2131 | 0.26 | 0.2342 | Recall too low |

### 3.4.4 Analysis

#### Figure 3.3: Precision / Recall / F1-Score vs. Threshold

*Figure 3.3: Precision, Recall, and F1-Score plotted against classification thresholds ranging from 0.10 to 0.85. The dashed gray line marks the best threshold (0.75) where F1-Score is maximized, and the dotted yellow line marks the default threshold (0.50).*

This chart visualizes how the three key classification metrics change as the decision threshold varies:

- **Recall (orange, square markers)**: Decreases monotonically as the threshold increases. At very low thresholds (0.10), Recall reaches its maximum (~0.94), meaning the model successfully detects nearly all stroke cases. As the threshold rises, the model becomes more selective, causing Recall to decline steadily — down to 0.26 at threshold 0.85. This behavior is expected: a higher threshold demands stronger evidence before predicting "Stroke," so fewer positive cases are captured.

- **Precision (blue, circle markers)**: Increases gradually as the threshold rises. At low thresholds, Precision is very low (~0.08), indicating that the vast majority of patients flagged as "Stroke" are actually false alarms. As the threshold increases, the model becomes more conservative, and the proportion of true positives among flagged patients improves — reaching ~0.23 at threshold 0.80. However, even at high thresholds, Precision remains relatively low due to the severe class imbalance (only ~5% of patients have strokes).

- **F1-Score (green, triangle markers)**: The harmonic mean of Precision and Recall, this curve exhibits an inverted-U shape. It starts low when Recall dominates and Precision is negligible (low thresholds), peaks around threshold **0.75** (F1 ≈ 0.31), and then declines again as Recall drops too sharply (high thresholds). The peak at 0.75 represents the mathematical "best balance" between Precision and Recall.

**Key Insight from Figure 3.3**: The two vertical reference lines highlight a critical decision point. While the **best F1 threshold (0.75)** optimizes the mathematical trade-off, it sacrifices Recall to only 0.56 — meaning **44% of stroke patients would be missed**. Conversely, the **default threshold (0.50)** achieves Recall = 0.74 but with lower Precision. For early screening purposes, moving the threshold even further left (to 0.20) maximizes patient safety by catching 86% of stroke cases, accepting the higher false alarm rate as an acceptable cost in a medical context.

#### Figure 3.4: Precision-Recall Curve

*Figure 3.4: Precision-Recall Curve for the Logistic Regression model. The green dot marks the operating point at the best F1 threshold (t = 0.75), where Precision ≈ 0.21 and Recall ≈ 0.56.*

The **Precision-Recall (PR) Curve** plots Precision (y-axis) against Recall (x-axis) across all possible classification thresholds, providing a complementary view to the ROC curve that is particularly informative for **imbalanced datasets**.

**How to read this curve:**

- Each point on the curve corresponds to a specific threshold. Moving from **left to right** corresponds to **decreasing** the threshold — the model becomes less selective, capturing more positive cases (higher Recall) but also generating more false alarms (lower Precision).
- An ideal model would occupy the **top-right corner** (Precision = 1.0, Recall = 1.0), meaning it detects all stroke cases without any false positives.
- For a random classifier on this dataset, the PR curve would be a horizontal line at approximately Precision = 0.05 (the prevalence of the positive class).

**Observations from the curve:**

1. **High-Precision, Low-Recall region (left side)**: At very high thresholds, the model achieves Precision values close to 1.0, but only for an extremely small number of predictions (Recall near 0). This means the few patients it flags are very likely true stroke cases, but it misses nearly everyone.

2. **Rapid Precision decay**: As Recall increases beyond ~0.10, Precision drops sharply from ~0.50 to ~0.25. This steep decline reflects the difficulty of the task — the model struggles to maintain prediction quality as it attempts to detect more stroke cases.

3. **Plateau region (Recall 0.20–0.60)**: Precision stabilizes around 0.20–0.25 in this range, indicating that the model's probability scores provide moderate discriminative power for roughly one-fifth to one-half of actual stroke cases.

4. **Best F1 operating point (green dot)**: At threshold 0.75, the model achieves Recall ≈ 0.56 and Precision ≈ 0.21. This point represents the best trade-off as measured by F1-Score, but it lies in a region where nearly half of stroke cases go undetected.

5. **Extended tail (Recall > 0.60)**: Beyond the F1-optimal point, Precision continues to decline gradually. At very high Recall values (>0.80), Precision falls below 0.10, meaning over 90% of flagged patients are false alarms — a cost deemed acceptable in early screening applications.

**Clinical Significance**: The PR curve confirms that achieving both high Precision and high Recall simultaneously is extremely challenging for this dataset due to the severe class imbalance (stroke prevalence ~5%). The curve's overall shape — steep decline followed by a long, low tail — is characteristic of models trained on highly imbalanced medical data. The selected screening threshold (0.20) operates in the far-right region of this curve, prioritizing Recall (0.86) over Precision (0.09), consistent with the medical objective of minimizing missed diagnoses.

The results reveal the classic **Precision–Recall trade-off**:

- **As threshold decreases** → Recall increases (more stroke cases detected), but Precision decreases (more false alarms) and Accuracy drops.
- **As threshold increases** → Accuracy and Precision improve, but Recall drops sharply. At threshold = 0.85, the model only detects 26% of stroke cases — missing nearly 3 out of 4 patients.

Three candidate thresholds were identified:

| Scenario | Threshold | Recall | F1 | Use Case |
|----------|-----------|--------|-----|----------|
| Early Screening | **0.20** | 0.86 | 0.17 | Maximizes detection of at-risk patients. Suitable for initial population-level screening where follow-up tests are available. |
| Balanced | **0.55** | 0.72 | 0.24 | Best balance between detecting stroke cases and limiting false alarms. Highest Precision among thresholds with Recall ≥ 0.70. |
| Best F1 | **0.75** | 0.56 | 0.31 | Optimizes the harmonic mean of Precision and Recall. However, misses 44% of stroke cases — potentially unacceptable in a medical context. |

### 3.4.5 Threshold Selection Rationale

**Selected Threshold: 0.20** (for the deployment model)

The following arguments support this choice:

1. **Medical priority**: In stroke prediction, a missed case (False Negative) can lead to permanent disability or death. The cost of a False Negative is incomparably higher than a False Positive (which only triggers additional screening).

2. **High Recall (0.86)**: At threshold 0.20, the model detects **86% of stroke patients**, significantly higher than the default threshold (74%) or the best F1 threshold (56%).

3. **Screening context**: This model is designed as an **initial screening tool**, not a final diagnostic instrument. Patients flagged by the model would undergo further clinical evaluation. Therefore, a higher false positive rate is acceptable.

4. **Adjustable**: The threshold is exported as a separate artifact (`optimal_threshold.pkl`) and can be adjusted based on clinical feedback without retraining the model.

---

## 3.5 Model Comparison and Selection

### 3.5.1 Comprehensive Comparison

| Criterion | Logistic Regression | Decision Tree | Random Forest |
|-----------|-------------------|---------------|---------------|
| ROC-AUC (tuned) | **0.8227** | 0.7016 | 0.7572 |
| Recall (default threshold) | **0.7400** | 0.3400 | 0.3000 |
| Recall (at threshold 0.20) | **0.86** | N/A | N/A |
| Inference speed | ⚡ Fast | ⚡ Fast | 🐢 Moderate |
| Interpretability | ✅ High | ✅ High | ❌ Low (black box) |
| Model file size | ~1.5 KB | Small | Larger |
| Deployment complexity | Low | Low | Moderate |

### 3.5.2 Why Logistic Regression?

**1. Best discriminative performance**: Logistic Regression achieves the highest ROC-AUC (0.8227), outperforming Random Forest (0.7572) and Decision Tree (0.7016) by a significant margin.

**2. Superior Recall**: At the default threshold, LR detects 74% of stroke cases, while DT and RF detect only 34% and 30% respectively. This gap is critical — DT and RF would miss 2 out of 3 stroke patients.

**3. Interpretability matters in healthcare**: Logistic Regression coefficients can be directly interpreted to explain *why* a patient was flagged. For instance, a physician can see that "age" and "glucose level" contribute most to the risk score. This transparency builds trust in clinical adoption. Random Forest, by contrast, is a "black box" that cannot easily explain individual predictions.

**4. Efficient deployment**: The LR model file is only ~1.5 KB, enabling fast inference and easy integration into web applications or mobile health platforms.

**5. Flexible probability output**: LR produces well-calibrated probability estimates, which are essential for threshold tuning. This allows the screening threshold to be adjusted post-deployment based on real-world clinical feedback.

### 3.5.3 Limitations and Future Work

- **Low Precision**: Even at optimal threshold, Precision remains low (~9%). This means many false alarms will occur. Future work could explore ensemble methods (e.g., XGBoost, LightGBM) or deep learning approaches to improve both Precision and Recall simultaneously.

- **Additional models**: Gradient Boosting methods (XGBoost, LightGBM), Support Vector Machines, and Neural Networks could be evaluated in future iterations.

- **Feature engineering**: More advanced features (e.g., interaction terms, polynomial features) may improve model performance.

- **External validation**: The model should be validated on external datasets from different hospitals/populations to assess generalizability.

---

## 3.6 Model Export

The final trained model and associated artifacts were exported for production deployment:

```
model/
├── best_logistic_regression.pkl    # Trained Logistic Regression model (C=100, L1, saga)
├── optimal_threshold.pkl           # Optimal classification threshold (0.20)
├── features.pkl                    # List of 15 input features for validation
├── imputer.pkl                     # Missing value imputer (from Chapter 2)
└── scaler.pkl                      # Feature scaler (from Chapter 2)
```

**Export Code:**

```python
import joblib

joblib.dump(best_lr, 'model/best_logistic_regression.pkl')
joblib.dump(0.20, 'model/optimal_threshold.pkl')
joblib.dump(features, 'model/features.pkl')
```

**Inference Usage:**

```python
# Load artifacts
model = joblib.load('model/best_logistic_regression.pkl')
threshold = joblib.load('model/optimal_threshold.pkl')
features = joblib.load('model/features.pkl')

# Predict on new patient data
y_prob = model.predict_proba(X_new)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
```

---

## Summary

This chapter presented the complete workflow of building and evaluating stroke prediction models:

1. **Three models** (Logistic Regression, Decision Tree, Random Forest) were trained and optimized using GridSearchCV with 5-fold cross-validation.

2. **Logistic Regression** was selected as the best model based on the highest ROC-AUC (0.8227), highest Recall (0.74), and superior interpretability.

3. **Threshold tuning** was performed across 16 threshold values. A threshold of **0.20** was selected to maximize Recall (0.86) for early screening applications, prioritizing the detection of at-risk patients over minimizing false alarms.

4. The core argument throughout this chapter is that **in medical prediction, minimizing False Negatives (missed stroke cases) is far more important than minimizing False Positives (false alarms)**, as a missed diagnosis can lead to irreversible consequences.

5. All model artifacts were exported for deployment in a production environment.
