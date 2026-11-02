# Symptom-Based Disease Diagnosis with Dynamic Questioning (ML)

An interactive, multi-class disease prediction system. The user enters initial symptoms, and the model asks targeted follow-up questions, updating its disease probabilities after every answer until it reaches a confident diagnosis. Built with a Random Forest classifier, Bayesian-smoothed symptom statistics, and probability calibration, and evaluated on noise-corrupted inputs to test robustness.

> **Disclaimer:** This is an educational machine learning project. It is not a medical device and must not be used for real medical diagnosis or treatment decisions.

---

## Highlights

- **Multi-class classification** across **41 diseases** and **131 unique symptoms** (4,920 records, 120 per disease, perfectly class-balanced).
- **Dynamic questioning engine** that picks the most discriminative next symptom to ask, instead of requiring all symptoms up front.
- **Robustness testing:** the held-out test set is corrupted by randomly flipping 10.5% of symptom values to simulate wrong or missing patient input.
- **Probability calibration** (`CalibratedClassifierCV`) so confidence scores are usable for stopping decisions.
- **Exploratory analysis and feature engineering:** missing-value analysis, binary symptom encoding, Bayesian smoothing, TF-IDF weighting, and a symptom co-occurrence graph with community detection.

---

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| ML | scikit-learn (RandomForestClassifier, RandomizedSearchCV, CalibratedClassifierCV, TfidfTransformer) |
| Data | pandas, NumPy |
| Graph analysis | NetworkX (co-occurrence network, greedy modularity communities) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook / Google Colab |

---

## Dataset

`dataset.csv` has one row per patient record: a `Disease` label and up to 17 symptom columns (`Symptom_1` to `Symptom_17`).

| Property | Value |
| --- | --- |
| Records | 4,920 |
| Diseases (classes) | 41 |
| Unique symptoms | 131 |
| Records per disease | 120 (balanced) |

---

## Methodology

### 1. Data preprocessing
- Visualized missing values (heatmap and per-column bar chart); the symptom columns are sparse, with later columns mostly empty.
- Replaced missing entries with a placeholder and extracted the list of unique symptoms.
- Converted each record into a **binary symptom matrix** (1 = symptom present, 0 = absent), producing a 131-feature model input.

### 2. Feature analysis
- **Bayesian smoothing** of disease-wise symptom probabilities:
  `P(symptom | disease) = (count + alpha) / (n_disease + alpha + beta)` with `alpha = 1`, `beta = 5`. This avoids zero-probability estimates for rare symptoms and powers the question-selection logic.
- **TF-IDF weighting** of symptoms to examine how informative each symptom is across diseases.
- **Symptom co-occurrence network** with greedy modularity community detection to find clusters of symptoms that appear together.
- Top-20 symptom frequency and Random Forest feature importance plots.

### 3. Modeling
- 80/20 **stratified** train/test split (`random_state=42`).
- **Baseline:** `RandomForestClassifier(class_weight="balanced")`.
- **Tuning:** `RandomizedSearchCV` (20 candidates, 3-fold CV) over `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`.
- **Calibration:** `CalibratedClassifierCV(cv=3)` on the tuned model.
- **Noise injection:** 10.5% of symptom values in the test set are randomly flipped (`add_noise_to_symptoms`) to measure real-world robustness.

### 4. Dynamic diagnosis engine
1. The user enters initial symptoms (comma-separated, with tolerant matching on spaces, underscores, and partial names).
2. The model outputs the **top-3 disease probabilities**.
3. For the leading candidates, the system scores every unasked symptom by
   `importance = P(disease) x discriminative_power x P(symptom | disease)`,
   where discriminative power is the symptom's probability in the candidate disease minus its highest probability in the other top candidates.
4. It asks the user a yes/no question about the highest-scoring symptom and re-predicts.
5. It stops when confidence exceeds 95%, or when confidence exceeds 80% with a lead of more than 30 points over the runner-up, or after a maximum of 10 questions, or when no informative questions remain.

Example session (input: `itching`):

```
Initial symptoms: itching
- Drug Reaction: 9.3%  - Fungal infection: 9.2%  - Jaundice: 5.2%
Do you have 'skin rash'?              yes  -> Fungal infection 13.0%
Do you have 'nodal skin eruptions'?   yes  -> Fungal infection 34.8%
Do you have 'dischromic patches'?     yes  -> Fungal infection 57.5%
Best estimate: Fungal infection
```

---

## Results

Evaluated on the held-out test set **with 10.5% symptom noise** (984 samples, 24 per class):

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Baseline Random Forest | 70.9% | 0.690 |
| Tuned Random Forest | 77.2% | 0.759 |
| **Tuned + Calibrated Random Forest** | **88.3%** | n/a |

Tuning plus calibration improved noisy-input accuracy by about **17 percentage points** over the baseline.

---

## Repository Structure

```
Symptom-Disease-Diagnosis-ML/
├── dataset.csv                          # Raw symptom-disease records
├── symptom_disease_diagnosis.ipynb      # Preprocessing, EDA, modeling, calibration
├── symptom_disease_diagnosis_d.ipynb    # Full pipeline + dynamic questioning system
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/raman7976/Symptom-Disease-Diagnosis-ML.git
cd Symptom-Disease-Diagnosis-ML
pip install numpy pandas scikit-learn seaborn matplotlib networkx jupyter
jupyter notebook symptom_disease_diagnosis_d.ipynb
```

Update the `pd.read_csv(...)` path in the first cells to point to your local `dataset.csv` (the notebooks use Colab paths by default). Run all cells; the final cell launches the interactive diagnosis session.

---

## Limitations and Future Work

- The source dataset contains many repeated records (about 300 distinct symptom patterns expanded to 4,920 rows), so a random split can place near-identical rows in both train and test sets. Reported accuracy is likely optimistic; a group-aware split or a larger real-world dataset would give a more honest estimate.
- Symptoms are binary; severity, duration, age, and medical history are not modeled.
- Possible extensions: a Streamlit or Flask web interface, gradient boosting and neural baselines, SHAP-based explanations, and information-gain-based question selection.

---

## Author

[Pulkit Pareek], The LNM Institute of Information Technology, Jaipur
