# Day 10 — Multi-label AMR Resistance Classifier
### 🧬 30 Days of Bioinformatics | Subhadip Jana

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Multi-output Random Forest classifier predicting resistance to 12 antibiotics simultaneously from bacterial species, ward, age, gender and year. All 12 labels achieve AUC > 0.81.

---

## 📊 Dashboard
![Dashboard](outputs/multilabel_rf_dashboard.png)

---

## 🔬 Problem Formulation

| Item | Detail |
|------|--------|
| **Task** | Multi-label classification (12 simultaneous labels) |
| **Labels** | R=1 / S+I=0 for 12 antibiotics |
| **Features** | Species (one-hot) + Ward + Age + Gender + Year (22 features) |
| **Model** | MultiOutputClassifier(RandomForestClassifier) |
| **Evaluation** | 5-fold stratified CV per label |

---

## 📈 Per-label CV Performance

| Antibiotic | Class | AUC | F1 | %R |
|------------|-------|-----|----|----|
| **VAN** | Glycopeptide | **0.993** | **0.946** | 35.6% |
| **CAZ** | Cephalosporin | **0.984** | **0.928** | 60.2% |
| **PEN** | Penicillin | **0.959** | **0.910** | 60.0% |
| **CLI** | Macrolide | **0.950** | **0.870** | 46.5% |
| GEN | Aminoglycoside | 0.930 | 0.751 | 22.8% |
| ERY | Macrolide | 0.916 | 0.805 | 54.2% |
| CIP | Fluoroquinolone | 0.824 | 0.445 | 11.4% |
| TMP | Sulfonamide | 0.812 | 0.606 | 28.5% |

> ✅ All 12 antibiotics achieve AUC > 0.81 — strong predictive signal from species + clinical metadata alone!

---

## 📊 Multi-label Global Metrics

| Metric | Value |
|--------|-------|
| Hamming Loss | 0.1345 |
| Subset Accuracy | 0.416 |
| Macro F1 | 0.784 |
| Weighted F1 | 0.831 |

---

## 🚀 How to Run
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
python multilabel_rf.py
```

---

## 📁 Structure
```
day10-multilabel-rf/
├── multilabel_rf.py
├── data/
│   └── isolates.csv
├── outputs/
│   ├── cv_metrics.csv
│   └── multilabel_rf_dashboard.png
└── README.md
```

---

## 🔗 Part of #30DaysOfBioinformatics
**Author:** Subhadip Jana | [GitHub](https://github.com/SubhadipJana1409) | [LinkedIn](https://linkedin.com/in/subhadip-jana1409)
