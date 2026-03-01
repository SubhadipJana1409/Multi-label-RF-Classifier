"""
================================================================
Day 10 — Multi-label AMR Resistance Classifier (REAL DATA)
Author  : Subhadip Jana
Dataset : example_isolates — AMR R package
          2,000 clinical isolates × 40 antibiotics (R/S/I)

Problem:
  Given a bacterial isolate's species, ward, age, gender —
  predict resistance to MULTIPLE antibiotics simultaneously.
  This is a MULTI-LABEL classification problem.

Approach:
  • Label: Binary resistance (R=1, S/I=0) for top 12 antibiotics
  • Features: Species (one-hot) + Ward (one-hot) + Age + Gender
  • Model: Random Forest Classifier (per-label + multi-output)
  • Evaluation: Per-label ROC-AUC, F1, Precision, Recall
  • Extra: Hamming loss, subset accuracy, label correlation
  • Feature importance per antibiotic label
  • Confusion matrices for top antibiotics
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, hamming_loss, confusion_matrix,
                              classification_report, roc_curve)
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────

print("🔬 Loading example_isolates dataset...")
df = pd.read_csv("data/isolates.csv")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

META = ["date","patient","age","gender","ward","mo","year"]
ALL_AB = [c for c in df.columns if c not in META]

# ── Select top 12 antibiotics (best coverage + diverse classes) ──
TARGET_ABS = ["ERY","AZM","AMC","VAN","GEN","CAZ",
              "CXM","SXT","PEN","CIP","CLI","TMP"]

AB_FULLNAMES = {
    "ERY":"Erythromycin","AZM":"Azithromycin","AMC":"Amoxicillin-clav",
    "VAN":"Vancomycin","GEN":"Gentamicin","CAZ":"Ceftazidime",
    "CXM":"Cefuroxime","SXT":"Trimethoprim-sulfa","PEN":"Penicillin",
    "CIP":"Ciprofloxacin","CLI":"Clindamycin","TMP":"Trimethoprim",
}

AB_CLASS = {
    "ERY":"Macrolide","AZM":"Macrolide","AMC":"Penicillin",
    "VAN":"Glycopeptide","GEN":"Aminoglycoside","CAZ":"Cephalosporin",
    "CXM":"Cephalosporin","SXT":"Sulfonamide","PEN":"Penicillin",
    "CIP":"Fluoroquinolone","CLI":"Macrolide","TMP":"Sulfonamide",
}

CLASS_COLORS = {
    "Macrolide":"#9B59B6","Penicillin":"#E74C3C","Glycopeptide":"#3498DB",
    "Aminoglycoside":"#F1C40F","Cephalosporin":"#E67E22",
    "Sulfonamide":"#2ECC71","Fluoroquinolone":"#1ABC9C",
}

print(f"✅ {len(df)} isolates | Target antibiotics: {len(TARGET_ABS)}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

print("\n⚙️  Engineering features...")

# Keep top 15 species, group rest as "Other"
top_species = df["mo"].value_counts().head(15).index.tolist()
df["species_grp"] = df["mo"].apply(lambda x: x if x in top_species else "Other")

# One-hot encode species + ward
species_dummies = pd.get_dummies(df["species_grp"], prefix="sp")
ward_dummies    = pd.get_dummies(df["ward"],         prefix="ward")
gender_bin      = (df["gender"] == "M").astype(int)
age_norm        = (df["age"] - df["age"].mean()) / df["age"].std()
year_norm       = (df["year"] - df["year"].mean()) / df["year"].std()

X_full = pd.concat([species_dummies, ward_dummies,
                    gender_bin.rename("gender_M"),
                    age_norm.rename("age"),
                    year_norm.rename("year")], axis=1).astype(float)

print(f"   Feature matrix: {X_full.shape}")

# ─────────────────────────────────────────────────────────────
# SECTION 3: PREPARE MULTI-LABEL TARGET MATRIX
# ─────────────────────────────────────────────────────────────

print("\n🎯 Preparing multi-label targets...")

# Build Y matrix: 1=Resistant, 0=Susceptible/Intermediate
# Drop rows where ALL selected antibiotics are NaN
Y_raw = df[TARGET_ABS].copy()
Y_bin = (Y_raw == "R").astype(float)   # NaN stays NaN

# For each antibiotic, use only rows with non-null label
print("   Label statistics:")
for ab in TARGET_ABS:
    n_total = Y_bin[ab].notna().sum()
    n_R     = (Y_bin[ab] == 1).sum()
    pct_R   = n_R / n_total * 100 if n_total > 0 else 0
    print(f"   {ab:5s}: {n_total:4d} labeled | {pct_R:.1f}% R")

# For multi-output model: rows where ALL 12 labels are present
complete_mask = Y_bin[TARGET_ABS].notna().all(axis=1)
X_complete    = X_full[complete_mask].reset_index(drop=True)
Y_complete    = Y_bin[TARGET_ABS][complete_mask].reset_index(drop=True).astype(int)

print(f"\n   Complete rows (all 12 labels): {len(X_complete)}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: MULTI-OUTPUT RANDOM FOREST
# ─────────────────────────────────────────────────────────────

print("\n🌲 Training Multi-Output Random Forest...")

rf_base = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
multi_rf = MultiOutputClassifier(rf_base)
multi_rf.fit(X_complete, Y_complete)

Y_pred      = multi_rf.predict(X_complete)
Y_pred_prob = np.array([est.predict_proba(X_complete)[:,1]
                         for est in multi_rf.estimators_]).T

print("✅ Multi-output RF trained")

# ─────────────────────────────────────────────────────────────
# SECTION 5: PER-LABEL CROSS-VALIDATED METRICS
# ─────────────────────────────────────────────────────────────

print("\n📊 Computing per-label CV metrics...")

cv_results = {}
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for ab in TARGET_ABS:
    # Use all rows with valid label for this antibiotic
    mask    = Y_bin[ab].notna()
    X_ab    = X_full[mask].values
    y_ab    = Y_bin[ab][mask].astype(int).values

    if y_ab.sum() < 10 or (len(y_ab) - y_ab.sum()) < 10:
        continue

    rf_single = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )

    auc_scores = []
    f1_scores  = []
    for train_idx, test_idx in kf.split(X_ab, y_ab):
        rf_single.fit(X_ab[train_idx], y_ab[train_idx])
        y_prob = rf_single.predict_proba(X_ab[test_idx])[:,1]
        y_pred = rf_single.predict(X_ab[test_idx])
        try:
            auc_scores.append(roc_auc_score(y_ab[test_idx], y_prob))
        except:
            pass
        f1_scores.append(f1_score(y_ab[test_idx], y_pred, zero_division=0))

    cv_results[ab] = {
        "AUC_mean"  : np.mean(auc_scores) if auc_scores else 0,
        "AUC_std"   : np.std(auc_scores)  if auc_scores else 0,
        "F1_mean"   : np.mean(f1_scores),
        "F1_std"    : np.std(f1_scores),
        "N"         : int(mask.sum()),
        "Pct_R"     : round(y_ab.mean()*100, 1),
        "Class"     : AB_CLASS.get(ab, "Other"),
        "Fullname"  : AB_FULLNAMES.get(ab, ab),
    }
    print(f"   {ab:5s}: AUC={cv_results[ab]['AUC_mean']:.3f}±"
          f"{cv_results[ab]['AUC_std']:.3f}  "
          f"F1={cv_results[ab]['F1_mean']:.3f}")

cv_df = pd.DataFrame(cv_results).T.sort_values("AUC_mean", ascending=False)
cv_df.to_csv("outputs/cv_metrics.csv")

# ─────────────────────────────────────────────────────────────
# SECTION 6: MULTI-LABEL METRICS (complete rows)
# ─────────────────────────────────────────────────────────────

print("\n📏 Multi-label global metrics...")

hl   = hamming_loss(Y_complete, Y_pred)
sa   = (Y_complete.values == Y_pred).all(axis=1).mean()
f1_m = f1_score(Y_complete, Y_pred, average="macro",  zero_division=0)
f1_w = f1_score(Y_complete, Y_pred, average="weighted",zero_division=0)

print(f"   Hamming Loss     : {hl:.4f}")
print(f"   Subset Accuracy  : {sa:.4f}")
print(f"   Macro F1         : {f1_m:.4f}")
print(f"   Weighted F1      : {f1_w:.4f}")

# ─────────────────────────────────────────────────────────────
# SECTION 7: FEATURE IMPORTANCE (top antibiotic)
# ─────────────────────────────────────────────────────────────

top_ab = cv_df.index[0]
idx    = TARGET_ABS.index(top_ab)
fi     = pd.Series(multi_rf.estimators_[idx].feature_importances_,
                   index=X_complete.columns)
top_fi = fi.nlargest(20)

# ─────────────────────────────────────────────────────────────
# SECTION 8: ROC CURVES (top 6 antibiotics)
# ─────────────────────────────────────────────────────────────

roc_data = {}
for ab in TARGET_ABS[:6]:
    mask  = Y_bin[ab].notna()
    X_ab  = X_full[mask].values
    y_ab  = Y_bin[ab][mask].astype(int).values
    rf_s  = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=42)
    rf_s.fit(X_ab, y_ab)
    y_prob = rf_s.predict_proba(X_ab)[:,1]
    fpr, tpr, _ = roc_curve(y_ab, y_prob)
    auc = roc_auc_score(y_ab, y_prob)
    roc_data[ab] = (fpr, tpr, auc)

# ─────────────────────────────────────────────────────────────
# SECTION 9: DASHBOARD
# ─────────────────────────────────────────────────────────────

print("\n🎨 Generating dashboard...")

fig = plt.figure(figsize=(24, 20))
fig.suptitle(
    "Multi-label AMR Resistance Classifier — REAL CLINICAL DATA\n"
    "Random Forest | 12 antibiotics | example_isolates (AMR R package)\n"
    "Features: Species + Ward + Age + Gender + Year",
    fontsize=15, fontweight="bold", y=0.99
)

COLORS_AB = [CLASS_COLORS.get(AB_CLASS.get(ab,"Other"),"#BDC3C7")
             for ab in cv_df.index]

# ── Plot 1: Per-label AUC bar chart ──
ax1 = fig.add_subplot(3, 3, 1)
aucs = cv_df["AUC_mean"].astype(float)
errs = cv_df["AUC_std"].astype(float)
bars = ax1.bar(range(len(aucs)), aucs.values, yerr=errs.values,
               color=COLORS_AB, edgecolor="black", linewidth=0.5,
               alpha=0.87, capsize=4)
ax1.axhline(0.5, color="gray", lw=1.5, linestyle="--", label="Random (0.5)")
ax1.axhline(0.7, color="green",lw=1,   linestyle=":",  label="Good (0.7)")
ax1.axhline(0.9, color="blue", lw=1,   linestyle=":",  label="Excellent (0.9)")
for bar, val in zip(bars, aucs.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f"{val:.2f}", ha="center", fontsize=7, fontweight="bold")
ax1.set_xticks(range(len(cv_df)))
ax1.set_xticklabels(cv_df.index, fontsize=9)
ax1.set_ylabel("ROC-AUC (5-fold CV)")
ax1.set_title("Per-label ROC-AUC\n(5-fold Cross-Validation)",
              fontweight="bold", fontsize=10)
ax1.set_ylim(0.3, 1.05)
ax1.legend(fontsize=7)
patches = [mpatches.Patch(color=c, label=k)
           for k, c in CLASS_COLORS.items()]
ax1.legend(handles=patches, fontsize=6, loc="lower right", ncol=2)

# ── Plot 2: Per-label F1 score ──
ax2 = fig.add_subplot(3, 3, 2)
f1s  = cv_df["F1_mean"].astype(float)
f1e  = cv_df["F1_std"].astype(float)
ax2.bar(range(len(f1s)), f1s.values, yerr=f1e.values,
        color=COLORS_AB, edgecolor="black", linewidth=0.5,
        alpha=0.87, capsize=4)
ax2.axhline(0.7, color="green", lw=1, linestyle=":", label="Good (0.7)")
for i, val in enumerate(f1s.values):
    ax2.text(i, val+0.01, f"{val:.2f}", ha="center", fontsize=7, fontweight="bold")
ax2.set_xticks(range(len(cv_df)))
ax2.set_xticklabels(cv_df.index, fontsize=9)
ax2.set_ylabel("F1 Score (5-fold CV)")
ax2.set_title("Per-label F1 Score\n(5-fold Cross-Validation)",
              fontweight="bold", fontsize=10)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=8)

# ── Plot 3: ROC curves (top 6) ──
ax3 = fig.add_subplot(3, 3, 3)
roc_colors = plt.cm.Set1(np.linspace(0,0.8,6))
for (ab, (fpr, tpr, auc)), color in zip(roc_data.items(), roc_colors):
    ax3.plot(fpr, tpr, lw=2, color=color,
             label=f"{ab} (AUC={auc:.3f})")
ax3.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
ax3.fill_between([0,1],[0,1], alpha=0.05, color="gray")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curves\n(Top 6 antibiotics — train set)",
              fontweight="bold", fontsize=10)
ax3.legend(fontsize=7, loc="lower right")

# ── Plot 4: Feature importance (top antibiotic) ──
ax4 = fig.add_subplot(3, 3, 4)
fi_colors = ["#E74C3C" if "sp_" in f
             else "#3498DB" if "ward_" in f
             else "#2ECC71" if f=="age"
             else "#F39C12" if f=="gender_M"
             else "#9B59B6"
             for f in top_fi.index]
ax4.barh(range(len(top_fi)), top_fi.values[::-1],
         color=fi_colors[::-1], edgecolor="black", linewidth=0.4)
ax4.set_yticks(range(len(top_fi)))
ax4.set_yticklabels([f.replace("sp_","").replace("ward_","") 
                     for f in top_fi.index[::-1]], fontsize=8)
ax4.set_xlabel("Feature Importance")
ax4.set_title(f"Feature Importance\n({top_ab} — {AB_FULLNAMES[top_ab]})",
              fontweight="bold", fontsize=10)
legend_fi = [
    mpatches.Patch(color="#E74C3C", label="Species"),
    mpatches.Patch(color="#3498DB", label="Ward"),
    mpatches.Patch(color="#2ECC71", label="Age"),
    mpatches.Patch(color="#F39C12", label="Gender"),
    mpatches.Patch(color="#9B59B6", label="Year"),
]
ax4.legend(handles=legend_fi, fontsize=7)

# ── Plot 5: Label co-occurrence (correlation) heatmap ──
ax5 = fig.add_subplot(3, 3, 5)
label_corr = Y_complete.corr()
mask_upper = np.triu(np.ones_like(label_corr, dtype=bool), k=1)
sns.heatmap(label_corr, ax=ax5, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, annot=True, fmt=".2f",
            linewidths=0.4, cbar_kws={"label":"Correlation","shrink":0.8},
            annot_kws={"size":6})
ax5.tick_params(axis="both", labelsize=8)
ax5.set_title("Label Co-occurrence Correlation\n(Antibiotic Resistance Pairs)",
              fontweight="bold", fontsize=10)

# ── Plot 6: AUC vs Resistance prevalence scatter ──
ax6 = fig.add_subplot(3, 3, 6)
pct_r_vals = cv_df["Pct_R"].astype(float)
auc_vals   = cv_df["AUC_mean"].astype(float)
ax6.scatter(pct_r_vals, auc_vals,
            c=[CLASS_COLORS.get(AB_CLASS.get(ab,"Other"),"gray")
               for ab in cv_df.index],
            s=120, alpha=0.85, edgecolors="black", linewidth=0.5)
for ab, (pct, auc) in zip(cv_df.index, zip(pct_r_vals, auc_vals)):
    ax6.annotate(ab, (pct, auc), fontsize=8,
                 xytext=(3, 4), textcoords="offset points")
ax6.set_xlabel("Resistance Prevalence (%R)")
ax6.set_ylabel("CV ROC-AUC")
ax6.set_title("AUC vs Resistance Prevalence\n(class imbalance effect)",
              fontweight="bold", fontsize=10)
ax6.axhline(0.7, color="green", lw=1, linestyle="--", alpha=0.5)

# ── Plot 7: Confusion matrix for best antibiotic ──
ax7 = fig.add_subplot(3, 3, 7)
mask_ab  = Y_bin[top_ab].notna()
X_top    = X_full[mask_ab].values
y_top    = Y_bin[top_ab][mask_ab].astype(int).values
rf_top   = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   random_state=42)
rf_top.fit(X_top, y_top)
y_pred_top = rf_top.predict(X_top)
cm = confusion_matrix(y_top, y_pred_top)
sns.heatmap(cm, ax=ax7, annot=True, fmt="d", cmap="Blues",
            xticklabels=["S/I","R"], yticklabels=["S/I","R"],
            cbar_kws={"shrink":0.7})
ax7.set_xlabel("Predicted"); ax7.set_ylabel("Actual")
ax7.set_title(f"Confusion Matrix\n({top_ab} — {AB_FULLNAMES[top_ab]})",
              fontweight="bold", fontsize=10)

# ── Plot 8: Multi-label metrics summary ──
ax8 = fig.add_subplot(3, 3, 8)
metrics_bar = {
    "Hamming\nLoss"   : hl,
    "Subset\nAccuracy": sa,
    "Macro\nF1"       : f1_m,
    "Weighted\nF1"    : f1_w,
}
colors_m = ["#E74C3C","#2ECC71","#3498DB","#9B59B6"]
bars8 = ax8.bar(range(len(metrics_bar)),
                list(metrics_bar.values()),
                color=colors_m, edgecolor="black",
                linewidth=0.6, alpha=0.87)
for bar, val in zip(bars8, metrics_bar.values()):
    ax8.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.01,
             f"{val:.3f}", ha="center",
             fontsize=10, fontweight="bold")
ax8.set_xticks(range(len(metrics_bar)))
ax8.set_xticklabels(list(metrics_bar.keys()), fontsize=9)
ax8.set_ylim(0, 1.05)
ax8.set_title("Multi-label Global Metrics\n(train set)",
              fontweight="bold", fontsize=10)
ax8.axhline(0.7, color="gray", lw=1, linestyle="--", alpha=0.5)

# ── Plot 9: Summary table ──
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis("off")
rows = []
for ab in cv_df.index:
    row = cv_df.loc[ab]
    rows.append([
        ab,
        row["Fullname"][:18],
        row["Class"],
        f"{float(row['AUC_mean']):.3f}",
        f"{float(row['F1_mean']):.3f}",
        f"{float(row['Pct_R']):.1f}%",
    ])
tbl = ax9.table(
    cellText=rows,
    colLabels=["AB","Full Name","Class","AUC","F1","%R"],
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1.4, 1.7)
for j in range(6): tbl[(0,j)].set_facecolor("#BDC3C7")
for i, ab in enumerate(cv_df.index, 1):
    c = CLASS_COLORS.get(AB_CLASS.get(ab,"Other"),"#BDC3C7")
    tbl[(i,2)].set_facecolor(c)
    tbl[(i,2)].set_text_props(color="white", fontweight="bold")
ax9.set_title("Per-label Classification Summary",
              fontweight="bold", fontsize=11, pad=20)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("outputs/multilabel_rf_dashboard.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dashboard saved → outputs/multilabel_rf_dashboard.png")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\nMulti-label metrics (complete rows = {len(X_complete)}):")
print(f"  Hamming Loss    : {hl:.4f}")
print(f"  Subset Accuracy : {sa:.4f}")
print(f"  Macro F1        : {f1_m:.4f}")
print(f"  Weighted F1     : {f1_w:.4f}")
print(f"\nPer-label CV performance:")
print(cv_df[["Fullname","Class","AUC_mean","F1_mean","Pct_R"]].to_string())
print(f"\nBest performing: {cv_df.index[0]} "
      f"(AUC={float(cv_df['AUC_mean'].iloc[0]):.3f})")
print(f"Hardest label  : {cv_df.index[-1]} "
      f"(AUC={float(cv_df['AUC_mean'].iloc[-1]):.3f})")
print("\n✅ All outputs saved!")
