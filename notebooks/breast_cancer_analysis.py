from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'monospace',
})

BRCA_COLOR = '#f778ba'   # pink — breast cancer
DEAD_COLOR = '#f85149'   # red — mortality
ALIVE_COLOR = '#56d364'   # green — survival
NEUTRAL = '#388bfd'   # blue

print("=" * 60)
print("  BREAST CANCER — TCGA Clinical ML Analysis")
print("  Author: Rosemary Kamau | ORCiD: 0009-0001-3303-2573")
print("=" * 60)

print("\n📊 Loading TCGA-BRCA data...")
df = pd.read_csv('data/tcga_brca.csv')
print(f"Raw shape: {df.shape}")
print(f"Vital status:\n{df['vital_status'].value_counts()}")

df = df[df['vital_status'].isin(['Alive', 'Dead'])]

# Convert age from days to years
df['age_years'] = (df['age_at_diagnosis'] / 365.25).round(1)

# Construct survival time
df['survival_days'] = np.where(
    df['vital_status'] == 'Dead',
    df['days_to_death'],
    df['days_to_last_follow_up']
)

# Binary outcome
df['outcome'] = (df['vital_status'] == 'Dead').astype(int)

# Survival in years
df['survival_years'] = df['survival_days'] / 365.25

print("\n✅ Cleaning done")
print(f"Usable cases: {len(df)}")
print(f"Mortality rate: {df['outcome'].mean()*100:.1f}%")
print(f"Median age: {df['age_years'].median():.1f} years")
print(f"Median survival: {df['survival_years'].median():.1f} years")


def simplify_diagnosis(d):
    if 'duct' in str(d).lower():
        return 'Ductal'
    elif 'lobular' in str(d).lower():
        return 'Lobular'
    elif 'metaplastic' in str(d).lower():
        return 'Metaplastic'
    elif 'mucinous' in str(d).lower():
        return 'Mucinous'
    else:
        return 'Other'


df['diagnosis_group'] = df['primary_diagnosis'].apply(simplify_diagnosis)

print("\nDiagnosis groups:")
print(df['diagnosis_group'].value_counts())

df['age_group'] = pd.cut(
    df['age_years'],
    bins=[0, 40, 60, 100],
    labels=['Young (<40)', 'Middle (40-60)', 'Older (60+)']
)

print("\nAge group distribution:")
print(df['age_group'].value_counts())
print(f"\nYoung onset (<40): {(df['age_years'] < 40).sum()} patients")

df['metastasis'] = df['metastasis_at_diagnosis'].map({
    'yes': 1,
    'no': 0,
    'Yes': 1,
    'No': 0
})

print("\nMetastasis at diagnosis:")
print(df['metastasis'].value_counts(dropna=False))
print(
    f"\nMetastatic mortality rate: {df[df['metastasis']==1]['outcome'].mean()*100:.1f}%")
print(
    f"Non-metastatic mortality rate: {df[df['metastasis']==0]['outcome'].mean()*100:.1f}%")

df = df.drop(columns=['metastasis_at_diagnosis', 'metastasis'],
             errors='ignore')

print("\n✅ Metastasis dropped — documented as limitation")
print(f"Final clean dataset: {len(df)} cases")
print(f"Columns available: {list(df.columns)}")

le = LabelEncoder()

df['diagnosis_enc'] = le.fit_transform(
    df['diagnosis_group'].fillna('Other'))

df['gender_enc'] = le.fit_transform(
    df['gender'].fillna('female'))

df['race_enc'] = le.fit_transform(
    df['race'].fillna('not reported'))

df['age_group_enc'] = le.fit_transform(
    df['age_group'].astype(str))

print("\n✅ Encoding done")
print(df[['diagnosis_group', 'diagnosis_enc']].drop_duplicates())

print("\n" + "="*60)
print("  SECTION 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    'TCGA Breast Cancer — Clinical Overview (1097 Real Cases)',
    fontsize=15, fontweight='bold', color='#e6edf3', y=1.02
)
ax = axes[0, 0]
counts = df['vital_status'].value_counts()
ax.pie(
    counts.values,
    labels=counts.index,
    colors=[ALIVE_COLOR, DEAD_COLOR],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'color': '#e6edf3'}
)
ax.set_title('Vital Status Distribution', fontweight='bold')

ax = axes[0, 1]
alive = df[df['vital_status'] == 'Alive']['age_years'].dropna()
dead = df[df['vital_status'] == 'Dead']['age_years'].dropna()

ax.hist(alive, bins=25, alpha=0.7, color=ALIVE_COLOR,
        label=f'Alive (n={len(alive)})', density=True)
ax.hist(dead, bins=25, alpha=0.7, color=DEAD_COLOR,
        label=f'Dead (n={len(dead)})', density=True)
ax.axvline(alive.median(), color=ALIVE_COLOR,
           linestyle='--', linewidth=2,
           label=f'Alive median: {alive.median():.0f}yrs')
ax.axvline(dead.median(), color=DEAD_COLOR,
           linestyle='--', linewidth=2,
           label=f'Dead median: {dead.median():.0f}yrs')
ax.set_title('Age at Diagnosis by Outcome')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 2]
diag_mort = df.groupby('diagnosis_group')['outcome'].agg(
    ['mean', 'count']).reset_index()
diag_mort['mean'] *= 100
diag_mort = diag_mort.sort_values('mean', ascending=True)

bars = ax.barh(
    diag_mort['diagnosis_group'],
    diag_mort['mean'],
    color=BRCA_COLOR, alpha=0.85,
    edgecolor='#30363d'
)
for bar, (_, row) in zip(bars, diag_mort.iterrows()):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height()/2,
        f"{row['mean']:.1f}% (n={int(row['count'])})",
        va='center', fontsize=8, color='#e6edf3'
    )
ax.set_title('Mortality Rate by Diagnosis Group')
ax.set_xlabel('Mortality Rate (%)')
ax.set_xlim(0, 40)
ax.grid(axis='x', alpha=0.3)


ax = axes[1, 0]
alive_surv = df[df['vital_status'] == 'Alive']['survival_years'].dropna()
dead_surv = df[df['vital_status'] == 'Dead']['survival_years'].dropna()

ax.hist(alive_surv, bins=25, alpha=0.7,
        color=ALIVE_COLOR, label='Alive (follow-up)',
        density=True)
ax.hist(dead_surv, bins=25, alpha=0.7,
        color=DEAD_COLOR, label='Dead (survival)',
        density=True)
ax.set_title('Survival/Follow-up Time (Years)')
ax.set_xlabel('Years')
ax.set_ylabel('Density')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1, 1]
age_mort = df.groupby('age_group', observed=True)['outcome'].agg(
    ['mean', 'count']).reset_index()
age_mort['mean'] *= 100

bars = ax.bar(
    age_mort['age_group'].astype(str),
    age_mort['mean'],
    color=BRCA_COLOR, alpha=0.85,
    edgecolor='#30363d'
)
for bar, (_, row) in zip(bars, age_mort.iterrows()):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.3,
        f"{row['mean']:.1f}%\n(n={int(row['count'])})",
        ha='center', va='bottom',
        fontsize=9, color='#e6edf3'
    )
ax.set_title('Mortality Rate by Age Group')
ax.set_ylabel('Mortality Rate (%)')
ax.set_ylim(0, 30)
ax.tick_params(axis='x', rotation=10)
ax.grid(axis='y', alpha=0.3)


ax = axes[1, 2]
alive_plot = df[df['vital_status'] == 'Alive']
dead_plot = df[df['vital_status'] == 'Dead']

ax.scatter(
    alive_plot['age_years'],
    alive_plot['survival_years'],
    alpha=0.4, color=ALIVE_COLOR,
    s=20, label='Alive'
)
ax.scatter(
    dead_plot['age_years'],
    dead_plot['survival_years'],
    alpha=0.6, color=DEAD_COLOR,
    s=20, label='Dead'
)
ax.set_title('Age vs Survival Time')
ax.set_xlabel('Age at Diagnosis (years)')
ax.set_ylabel('Survival/Follow-up (years)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    'outputs/01_eda_overview.png',
    dpi=150, bbox_inches='tight',
    facecolor='#0d1117'
)
plt.close()
print("\n✅ EDA plot saved → outputs/01_eda_overview.png")

# ── SECTION 3: FEATURE ENGINEERING ──
print("\n" + "="*60)
print("  SECTION 3: FEATURE ENGINEERING")
print("="*60)


df['young_onset'] = (df['age_years'] < 40).astype(int)

df['long_survival'] = (df['survival_years'] > 5).astype(int)

df['diagnosis_ductal'] = (
    df['diagnosis_group'] == 'Ductal').astype(int)

print(f"Young onset (<40):     {df['young_onset'].sum()} patients")
print(f"Long survivors (>5yr): {df['long_survival'].sum()} patients")
print(f"Ductal carcinoma:      {df['diagnosis_ductal'].sum()} patients")

feature_cols = [
    'age_years',
    'gender_enc',
    'race_enc',
    'age_group_enc',
    'diagnosis_enc',
    'young_onset',
    'diagnosis_ductal',
]

df_ml = df[feature_cols + ['outcome']].dropna()
print(f"\n✅ ML dataset: {len(df_ml)} cases")
print(f"Features: {feature_cols}")
print(
    f"Class balance — Dead: {df_ml['outcome'].sum()}, Alive: {(df_ml['outcome']==0).sum()}")

# ── SECTION 4: ML MODELS ──
print("\n" + "="*60)
print("  SECTION 4: ML MODELS")
print("="*60)


X = df_ml[feature_cols]
y = df_ml['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Training set: {len(X_train)} cases")
print(f"Test set:     {len(X_test)} cases")


models = {
    'Logistic Regression': LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}


models = {
    'Logistic Regression': LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print(f"\n  {'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'CV Score':>12}")
print(f"  {'-'*60}")

for name, model in models.items():
    X_tr = X_train_s if name == 'Logistic Regression' else X_train
    X_te = X_test_s if name == 'Logistic Regression' else X_test
    X_cv = X_train_s if name == 'Logistic Regression' else X_train

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(
        model, X_cv, y_train,
        cv=cv, scoring='roc_auc'
    )

    results[name] = {
        'model': model,
        'accuracy': acc,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    print(f"  {name:<25} {acc*100:>9.1f}% {auc:>10.3f}"
          f"  {cv_scores.mean():.3f}±{cv_scores.std():.3f}")

# ── ROC Curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Breast Cancer — Model Performance',
    fontsize=14, fontweight='bold', color='#e6edf3'
)

colors_roc = ['#f0883e', '#56d364', '#79c0ff']
ax = axes[0]
for (name, res), col in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
    ax.plot(fpr, tpr, color=col, linewidth=2,
            label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'white', linestyle='--', alpha=0.4,
        label='Random (AUC=0.5)')
ax.set_title('ROC Curves — Mortality Prediction')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ── Feature Importance ──
ax = axes[1]
rf = results['Random Forest']['model']
importances = pd.Series(
    rf.feature_importances_,
    index=feature_cols
).sort_values(ascending=True)

ax.barh(
    importances.index,
    importances.values,
    color=BRCA_COLOR, alpha=0.85,
    edgecolor='#30363d'
)
ax.set_title('Feature Importance — Random Forest')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(
    'outputs/02_model_performance.png',
    dpi=150, bbox_inches='tight',
    facecolor='#0d1117'
)
plt.close()
print("\n✅ Model performance saved → outputs/02_model_performance.png")

# ── SECTION 5: SURVIVAL ANALYSIS ──
print("\n" + "="*60)
print("  SECTION 5: SURVIVAL ANALYSIS")
print("="*60)


def kaplan_meier(times, events):
    times = np.array(times)
    events = np.array(events)
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    unique_times = np.unique(times[events == 1])
    n = len(times)
    surv_prob = 1.0
    surv_times = [0]
    surv_probs = [1.0]
    for t in unique_times:
        at_risk = np.sum(times >= t)
        deaths = np.sum((times == t) & (events == 1))
        surv_prob *= (1 - deaths / at_risk)
        surv_times.append(t)
        surv_probs.append(surv_prob)
    return np.array(surv_times) / 365.25, np.array(surv_probs)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'Breast Cancer — Kaplan-Meier Survival Analysis\n(TCGA-BRCA Real Patient Data)',
    fontsize=14, fontweight='bold', color='#e6edf3'
)

# ── By Diagnosis Group ──
ax = axes[0]
colors_km = [BRCA_COLOR, ALIVE_COLOR, DEAD_COLOR, NEUTRAL, '#79c0ff']
for group, color in zip(df['diagnosis_group'].unique(), colors_km):
    subset = df[df['diagnosis_group'] == group].dropna(
        subset=['survival_days'])
    if len(subset) < 5:
        continue
    t, s = kaplan_meier(subset['survival_days'], subset['outcome'])
    ax.step(t, s*100, color=color, linewidth=2,
            label=f"{group} (n={len(subset)})", where='post')
    ax.fill_between(t, s*100, alpha=0.1, color=color, step='post')

ax.set_title('Survival by Diagnosis Subtype')
ax.set_xlabel('Years from Diagnosis')
ax.set_ylabel('Survival Probability (%)')
ax.set_ylim(0, 105)
ax.legend(fontsize=8, loc='lower left')
ax.axhline(50, color='white', linestyle=':', alpha=0.4)
ax.grid(alpha=0.3)

# ── By Age Group ──
ax = axes[1]
age_colors = ['#79c0ff', '#56d364', '#f85149']
for age_grp, color in zip(
        ['Young (<40)', 'Middle (40-60)', 'Older (60+)'],
        age_colors):
    subset = df[df['age_group'] == age_grp].dropna(
        subset=['survival_days'])
    if len(subset) < 5:
        continue
    t, s = kaplan_meier(subset['survival_days'], subset['outcome'])
    ax.step(t, s*100, color=color, linewidth=2,
            label=f"{age_grp} (n={len(subset)})",
            where='post')
    ax.fill_between(
        t, s*100, alpha=0.1,
        color=color, step='post')

ax.set_title('Survival by Age Group')
ax.set_xlabel('Years from Diagnosis')
ax.set_ylabel('Survival Probability (%)')
ax.set_ylim(0, 105)
ax.legend(fontsize=8, loc='lower left')
ax.axhline(50, color='white', linestyle=':', alpha=0.4)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    'outputs/03_survival_analysis.png',
    dpi=150, bbox_inches='tight',
    facecolor='#0d1117'
)
plt.close()
print("✅ Survival analysis saved → outputs/03_survival_analysis.png")

# ── SECTION 6: FINAL SUMMARY ──
print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)

best = max(results.items(), key=lambda x: x[1]['auc'])
print(f"""
  DATASET
    Source:          TCGA-BRCA (real patient data)
    Total cases:     1098
    Usable cases:    1097
    Alive:           945
    Dead:            152
    Mortality rate:  13.9%

  CLINICAL INSIGHTS
    Median age:      59.0 years
    Young onset:     69 patients (6.3%)
    Long survivors:  233 patients (21.2%)
    Dominant type:   Ductal (811 cases, 73.9%)

  KEY FINDINGS
    1. Age is strongest predictor of mortality
    2. Metaplastic carcinoma has worst survival
    3. Mucinous carcinoma has best survival
    4. Young onset (<40) shows aggressive early decline
    5. Clinical features alone insufficient for
       strong mortality prediction — molecular
       features (ER/PR/HER2, BRCA status) needed

  BEST ML MODEL
    Model:    {best[0]}
    ROC-AUC:  {best[1]['auc']:.3f}

  LIMITATIONS
    - tumor_stage 100% missing
    - No molecular subtype data
    - No treatment information
    - Class imbalance (86% alive)
""")

df.to_csv('data/tcga_brca_cleaned.csv', index=False)
print("✅ Analysis complete — ready for GitHub + Zenodo + ORCiD")
