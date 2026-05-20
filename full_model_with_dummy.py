import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from mord import LogisticAT

# =========================
# 1. LOAD DATA
# =========================

file_path = "Post Insights Sheet(Data).csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)


# =========================
# 2. CREATE POSITION GROUPS
# =========================

position_map = {
    "QB": "Skill", "RB": "Skill", "FB": "Skill", "WR": "Skill", "TE": "Skill",
    "OT": "OL", "OG": "OL", "G": "OL", "C": "OL", "OC": "OL", "IOL": "OL", "OL": "OL",
    "EDGE": "DL", "DE": "DL", "DT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "CB": "Secondary", "S": "Secondary", "SAF": "Secondary", "DB": "Secondary",
    "K": "Special Teams", "P": "Special Teams", "LS": "Special Teams"
}

df["Position Group"] = df["Position"].map(position_map).fillna(df["Position"])


# =========================
# 3. CREATE ORDINAL DRAFT TIER TARGET
# =========================

def draft_tier(pick):
    if pick <= 32:
        return "Round 1"
    elif pick <= 96:
        return "Rounds 2-3"
    elif pick <= 160:
        return "Rounds 4-5"
    elif pick <= 262:
        return "Rounds 6-7"
    else:
        return "Undrafted"

df["Draft Tier"] = df["Overall Pick"].apply(draft_tier)

labels_order = [
    "Round 1",
    "Rounds 2-3",
    "Rounds 4-5",
    "Rounds 6-7",
    "Undrafted"
]

tier_to_num = {
    "Round 1": 0,
    "Rounds 2-3": 1,
    "Rounds 4-5": 2,
    "Rounds 6-7": 3,
    "Undrafted": 4
}

num_to_tier = {
    0: "Round 1",
    1: "Rounds 2-3",
    2: "Rounds 4-5",
    3: "Rounds 6-7",
    4: "Undrafted"
}

df["Draft Tier Ordinal"] = df["Draft Tier"].map(tier_to_num)

print("\nDraft Tier Counts:")
print(df["Draft Tier"].value_counts())


# =========================
# 4. FEATURE SETS
# =========================

# Full model features (your actual model)
features = [
    "Season",
    "Position",
    "Position Group",
    "School",
    "Conference",
    "Height (inches)",
    "Weight(lbs)",
    "Hand (inches)",
    "Arm (inches)",
    "Wingspan (inches)",
    "40-Yard Dash (seconds)",
    "Vertical (inches)",
    "Bench (reps)",
    "Broad Jump (inches)",
    "3-Cone (seconds)",
    "Shuttle (seconds)",
    "Combine Participation",
    "ELO",
    "FPI",
    "Efficiencies Overall",
    "Efficiencies Offense",
    "Efficiencies Defense",
    "Efficiencies SpecialTeams",
    "CFP Appearance",
    "National Award"
]

combine_metrics = [
    "Height (inches)",
    "Weight(lbs)",
    "Hand (inches)",
    "Arm (inches)",
    "Wingspan (inches)",
    "40-Yard Dash (seconds)",
    "Vertical (inches)",
    "Bench (reps)",
    "Broad Jump (inches)",
    "3-Cone (seconds)",
    "Shuttle (seconds)"
]

# Dummy model features — only what's known before combine invites:
#   Position:       draft value varies enormously by position
#   Conference:     simple proxy for competition level
#   CFP Appearance: one binary signal of program quality
#   Season:         controls for year-to-year draft class variation
# Excludes all combine measurements, efficiency metrics, ELO/FPI,
# school name, and awards
dummy_features = [
    "Season",
    "Position",
    "Conference",
    "CFP Appearance"
]


# =========================
# 5. BUILD ORDINAL MODEL FUNCTION
# =========================

def build_model(feature_list):
    categorical_features = [
        col for col in ["Position", "Position Group", "School", "Conference"]
        if col in feature_list
    ]

    numeric_features = [
        col for col in feature_list if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("ordinal_model", LogisticAT(alpha=1.0))
        ]
    )

    return model


# =========================
# 6. MANUAL ORDINAL PREDICTION REPORT
# =========================

def print_manual_report(y_true, y_pred, title):
    y_true_labels = pd.Series(y_true).map(num_to_tier)
    y_pred_labels = pd.Series(y_pred).map(num_to_tier)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels,
        y_pred_labels,
        labels=labels_order,
        zero_division=0
    )

    print(f"\n{title}")
    print(f"{'Tier':<14}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}{'Support':>12}")

    for label, p, r, f, s in zip(labels_order, precision, recall, f1, support):
        print(f"{label:<14}{p:>12.2f}{r:>12.2f}{f:>12.2f}{s:>12}")

    print("\nAccuracy:", round(accuracy_score(y_true, y_pred), 4))

    ordinal_error = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    print("Average Ordinal Error:", round(ordinal_error, 4))


# =========================
# 7. OVERALL ORDINAL MODEL
# =========================

X = df[features]
y = df["Draft Tier Ordinal"]

# --- Full model ---
overall_model = build_model(features)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    overall_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nOVERALL ORDINAL MODEL - 5-Fold CV Accuracy Scores:")
print(cv_scores)
print("Average CV Accuracy:", round(cv_scores.mean(), 4))
print("CV Accuracy Standard Deviation:", round(cv_scores.std(), 4))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

overall_model.fit(X_train, y_train)
y_pred = overall_model.predict(X_test)

print_manual_report(
    y_test,
    y_pred,
    "OVERALL MODEL - Ordinal Prediction Report"
)

# --- Dummy model (overall) ---
# Fit on same train/test split as full model for fair comparison
X_dummy_full       = df[dummy_features]
X_train_dummy      = X_dummy_full.loc[X_train.index]
X_test_dummy       = X_dummy_full.loc[X_test.index]

overall_dummy_model = build_model(dummy_features)

cv_scores_dummy = cross_val_score(
    overall_dummy_model,
    X_dummy_full,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nOVERALL DUMMY MODEL - 5-Fold CV Accuracy Scores:")
print(cv_scores_dummy)
print("Average CV Accuracy:", round(cv_scores_dummy.mean(), 4))
print("CV Accuracy Standard Deviation:", round(cv_scores_dummy.std(), 4))

overall_dummy_model.fit(X_train_dummy, y_train)
y_pred_dummy = overall_dummy_model.predict(X_test_dummy)

print_manual_report(
    y_test,
    y_pred_dummy,
    "OVERALL DUMMY MODEL - Ordinal Prediction Report"
)


# =========================
# 8. OVERALL FEATURE COEFFICIENTS
# =========================

ordinal_model = overall_model.named_steps["ordinal_model"]
feature_names = overall_model.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": ordinal_model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nTop 20 Overall Features by Absolute Coefficient Size:")
print(importance_df.head(20).to_string(index=False))


# =========================
# 9. POSITION GROUP ORDINAL MODELS
# =========================

position_group_results = []
position_group_importances = []

# Store fitted group models and test data for gains charts
group_model_store = {}

for group in df["Position Group"].dropna().unique():
    group_df = df[df["Position Group"] == group].copy()

    print("\n" + "=" * 70)
    print(f"POSITION GROUP ORDINAL MODEL: {group}")
    print("=" * 70)
    print("Rows:", len(group_df))
    print("Draft Tier Counts:")
    print(group_df["Draft Tier"].value_counts())

    if len(group_df) < 50 or group_df["Draft Tier Ordinal"].nunique() < 2:
        print("Skipped: not enough data or only one draft tier present.")
        continue

    group_features = [
        col for col in features if col != "Position Group"
    ]

    X_group = group_df[group_features]
    y_group = group_df["Draft Tier Ordinal"]

    min_class_count = y_group.value_counts().min()

    if min_class_count < 2:
        print("Skipped: at least one tier has fewer than 2 players.")
        continue

    n_splits = min(5, min_class_count)

    group_model = build_model(group_features)

    group_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    group_cv_scores = cross_val_score(
        group_model,
        X_group,
        y_group,
        cv=group_cv,
        scoring="accuracy"
    )

    print(f"\n{n_splits}-Fold Stratified CV Accuracy Scores:")
    print(group_cv_scores)
    print("Average CV Accuracy:", round(group_cv_scores.mean(), 4))
    print("CV Accuracy Standard Deviation:", round(group_cv_scores.std(), 4))

    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X_group,
        y_group,
        test_size=0.25,
        random_state=42,
        stratify=y_group
    )

    group_model.fit(X_train_g, y_train_g)
    y_pred_g = group_model.predict(X_test_g)

    print_manual_report(
        y_test_g,
        y_pred_g,
        f"{group} - Ordinal Prediction Report"
    )

    # --- Dummy model for this position group ---
    # Use same train/test indices as the full group model
    group_dummy_features = [
        col for col in dummy_features if col != "Position Group"
    ]

    X_group_dummy   = group_df[group_dummy_features]
    X_train_gd      = X_group_dummy.loc[X_train_g.index]
    X_test_gd       = X_group_dummy.loc[X_test_g.index]

    group_dummy_model = build_model(group_dummy_features)
    group_dummy_model.fit(X_train_gd, y_train_g)
    y_pred_gd = group_dummy_model.predict(X_test_gd)

    print_manual_report(
        y_test_g,
        y_pred_gd,
        f"{group} - DUMMY MODEL Ordinal Prediction Report"
    )

    # Store everything needed for gains charts
    group_model_store[group] = {
        "full_model":   group_model,
        "dummy_model":  group_dummy_model,
        "X_test_full":  X_test_g,
        "X_test_dummy": X_test_gd,
        "y_test":       y_test_g,
        "group_features":       group_features,
        "group_dummy_features": group_dummy_features,
    }

    position_group_results.append({
        "Position Group": group,
        "Rows": len(group_df),
        "CV Accuracy Mean": group_cv_scores.mean(),
        "CV Accuracy Std": group_cv_scores.std(),
        "Test Accuracy": accuracy_score(y_test_g, y_pred_g),
        "Average Ordinal Error": np.mean(np.abs(np.array(y_test_g) - np.array(y_pred_g)))
    })

    ordinal_model_g = group_model.named_steps["ordinal_model"]
    feature_names_g = group_model.named_steps["preprocessor"].get_feature_names_out()

    importance_g = pd.DataFrame({
        "Feature": feature_names_g,
        "Coefficient": ordinal_model_g.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    print(f"\nTop 15 Features for {group} by Absolute Coefficient Size:")
    print(importance_g.head(15).to_string(index=False))

    combine_group_importance = []

    for metric in combine_metrics:
        matching_rows = importance_g[
            importance_g["Feature"].str.contains(metric, regex=False)
        ]

        combine_group_importance.append({
            "Position Group": group,
            "Metric": metric,
            "Coefficient Strength": matching_rows["Coefficient"].abs().sum()
        })

    combine_group_df = pd.DataFrame(combine_group_importance).sort_values(
        by="Coefficient Strength",
        ascending=False
    )

    print(f"\nCombine Metric Coefficient Strength for {group}:")
    print(combine_group_df.to_string(index=False))

    position_group_importances.append(combine_group_df)


# =========================
# 10. SAVE POSITION GROUP SUMMARY
# =========================

position_group_results_df = pd.DataFrame(position_group_results)

print("\nPOSITION GROUP ORDINAL MODEL SUMMARY:")
print(position_group_results_df.to_string(index=False))

position_group_results_df.to_csv(
    "position_group_ordinal_model_summary_no_log_rank.csv",
    index=False
)

if position_group_importances:
    all_position_importances_df = pd.concat(
        position_group_importances,
        ignore_index=True
    )

    all_position_importances_df.to_csv(
        "position_group_combine_metric_coefficient_strength_no_log_rank.csv",
        index=False
    )


# =========================
# 11. GAINS CHART HELPER
#     Three lines: Full Model | Dummy Model | Random Baseline
# =========================

def plot_gains_chart_with_dummy(
    target_tier,
    full_model,
    dummy_model,
    X_test_full,
    X_test_dummy,
    y_test_series,
    label_prefix=""
):
    """
    Plots a gains chart for one tier with three lines:
      - Full model   (green, solid)
      - Dummy model  (orange, dashed)
      - Random       (gray, dotted)

    Returns the full-model gains DataFrame (for CSV export, matching
    the original script's behaviour).
    """
    full_probs  = full_model.predict_proba(X_test_full)
    dummy_probs = dummy_model.predict_proba(X_test_dummy)

    full_prob_df  = pd.DataFrame(full_probs,  columns=labels_order)
    dummy_prob_df = pd.DataFrame(dummy_probs, columns=labels_order)

    y_test_labels = pd.Series(y_test_series).reset_index(drop=True).map(num_to_tier)
    actual_binary = (y_test_labels == target_tier).astype(int).values
    total = actual_binary.sum()

    if total == 0:
        print(f"No actual {target_tier} players in the test set.")
        return None

    n           = len(actual_binary)
    pct_players = np.arange(1, n + 1) / n * 100

    # --- Full model curve ---
    order_full = np.argsort(-full_prob_df[target_tier].values)
    cum_full   = actual_binary[order_full].cumsum() / total * 100

    # --- Dummy model curve ---
    order_dummy = np.argsort(-dummy_prob_df[target_tier].values)
    cum_dummy   = actual_binary[order_dummy].cumsum() / total * 100

    # --- Plot ---
    plt.figure(figsize=(10, 6))

    plt.plot(
        pct_players, cum_full,
        color="#2ecc71", linewidth=2.5,
        label="Full Model"
    )
    plt.plot(
        pct_players, cum_dummy,
        color="#e67e22", linewidth=2.5, linestyle="--",
        label="Dummy Model\n(Position / Conference / CFP / Season)"
    )
    plt.plot(
        [0, 100], [0, 100],
        color="#95a5a6", linewidth=1.5, linestyle=":",
        label="Random Baseline"
    )

    title_prefix = f"{label_prefix}: " if label_prefix else ""
    plt.title(
        f"{title_prefix}Gains Chart — {target_tier}",
        fontsize=13, fontweight="bold"
    )
    plt.xlabel("% of Players Reviewed")
    plt.ylabel(f"% of Actual {target_tier} Players Captured")
    plt.xlim(0, 100)
    plt.ylim(0, 105)
    plt.legend(fontsize=9, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Build the gains DataFrame (same structure as original script)
    temp_df = X_test_full.copy().reset_index(drop=True)
    temp_df["Actual Draft Tier"] = y_test_labels.values
    temp_df["Actual Target"] = actual_binary

    full_prob_series = full_prob_df[target_tier].values
    temp_df_sorted   = temp_df.copy()
    temp_df_sorted["Predicted Probability"] = full_prob_series
    temp_df_sorted   = temp_df_sorted.sort_values(
        by="Predicted Probability", ascending=False
    ).reset_index(drop=True)

    temp_df_sorted["Cumulative Actual Captured"]  = temp_df_sorted["Actual Target"].cumsum()
    temp_df_sorted["Percent of Players Reviewed"] = pct_players
    temp_df_sorted["Percent of Actual Tier Captured"] = (
        temp_df_sorted["Cumulative Actual Captured"] / total * 100
    )

    return temp_df_sorted


# =========================
# 12. OVERALL GAINS CHARTS
# =========================

print("\nCreating overall gains charts (Full vs Dummy vs Random)...")

overall_tier_outputs = {}

for tier in labels_order:
    print(f"\nCreating gains chart for: {tier}")
    result = plot_gains_chart_with_dummy(
        target_tier=tier,
        full_model=overall_model,
        dummy_model=overall_dummy_model,
        X_test_full=X_test,
        X_test_dummy=X_test_dummy,
        y_test_series=y_test,
        label_prefix="Overall"
    )
    overall_tier_outputs[tier] = result

for tier, output_df in overall_tier_outputs.items():
    if output_df is not None:
        clean_name = (
            tier.replace(" ", "_")
                .replace("-", "_")
                .replace("/", "_")
        )
        output_df.to_csv(
            f"gains_chart_{clean_name}_ordinal_no_log_rank.csv",
            index=False
        )


# =========================
# 13. POSITION GROUP GAINS CHARTS
# =========================

print("\nCreating position group gains charts (Full vs Dummy vs Random)...")

for group, store in group_model_store.items():
    print(f"\n--- {group} ---")
    for tier in labels_order:
        print(f"  Creating gains chart for: {tier}")
        plot_gains_chart_with_dummy(
            target_tier=tier,
            full_model=store["full_model"],
            dummy_model=store["dummy_model"],
            X_test_full=store["X_test_full"],
            X_test_dummy=store["X_test_dummy"],
            y_test_series=store["y_test"],
            label_prefix=group
        )


print("\nDone. All models, coefficient tables, and gains charts complete.")
