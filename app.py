

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(page_title="EEG Emotion Recognition Dashboard", layout="wide")

st.title("EEG Emotion Recognition Dashboard")
st.info("Cross-Subject EEG Emotion Recognition using LOSO Evaluation")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Controls")

model = st.sidebar.selectbox(
    "Select Model",
    ["SVM", "Random Forest", "KNN"]
)

model_map = {
    "SVM": "svm",
    "Random Forest": "rf",
    "KNN": "knn"
}

folder = model_map[model]

# --------------------------------------------------
# Helper
# --------------------------------------------------

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# --------------------------------------------------
# Load Data
# --------------------------------------------------

# BASELINE (RAW)
predictions_baseline = load_csv(f"{folder}/predictions_baseline_{folder}.csv")
results_baseline = load_csv(f"{folder}/results_baseline_{folder}.csv")
summary_baseline = load_csv(f"{folder}/summary_baseline_{folder}.csv")

# Z-SCORE
predictions_zscore = load_csv(f"{folder}/predictions_zscore_{folder}.csv")
results_zscore = load_csv(f"{folder}/results_zscore_{folder}.csv")
summary_zscore = load_csv(f"{folder}/summary_zscore_{folder}.csv")

# SRAN
predictions_sran = load_csv(f"{folder}/predictions_sran_{folder}.csv")
results_sran = load_csv(f"{folder}/results_sran_{folder}.csv")
summary_sran = load_csv(f"{folder}/summary_sran_{folder}.csv")

# SRAN FULL (optional)
has_full = os.path.exists(f"{folder}/summary_sran_full_{folder}.csv")

if has_full:
    predictions_full = load_csv(f"{folder}/predictions_sran_full_{folder}.csv")
    results_full = load_csv(f"{folder}/results_sran_full_{folder}.csv")
    summary_full = load_csv(f"{folder}/summary_sran_full_{folder}.csv")

# --------------------------------------------------
# Method Lists
# --------------------------------------------------

methods = ["BASELINE (RAW)", "Z-SCORE", "SRAN"]
if has_full:
    methods.append("SRAN with Transfer Learning")

# --------------------------------------------------
# Sidebar Selection
# --------------------------------------------------

method_choice = st.sidebar.selectbox("Select Method", methods)
comparison_method = st.sidebar.selectbox("Compare Against", methods)

# --------------------------------------------------
# Data Getter
# --------------------------------------------------

def get_data(method):
    if method == "BASELINE (RAW)":
        return predictions_baseline, results_baseline, summary_baseline
    elif method == "Z-SCORE":
        return predictions_zscore, results_zscore, summary_zscore
    elif method == "SRAN":
        return predictions_sran, results_sran, summary_sran
    else:
        return predictions_full, results_full, summary_full

predictions, results, summary = get_data(method_choice)
ref_predictions, ref_results, ref_summary = get_data(comparison_method)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

current_acc = summary["mean_accuracy"].iloc[0] * 100
current_f1 = summary["mean_f1"].iloc[0] * 100
current_std = summary["std_accuracy"].iloc[0] * 100

ref_acc = ref_summary["mean_accuracy"].iloc[0] * 100
ref_f1 = ref_summary["mean_f1"].iloc[0] * 100
ref_std = ref_summary["std_accuracy"].iloc[0] * 100

# --------------------------------------------------
# Performance Overview
# --------------------------------------------------

st.header("Performance Overview")

col1, col2 = st.columns(2)

col1.metric(f"{comparison_method} Accuracy", f"{ref_acc:.2f}%")

col2.metric(
    f"{method_choice} Accuracy",
    f"{current_acc:.2f}%",
    f"{current_acc - ref_acc:.2f}%"
)

col3, col4 = st.columns(2)

col3.metric(f"{comparison_method} F1", f"{ref_f1:.2f}%")

col4.metric(
    f"{method_choice} F1",
    f"{current_f1:.2f}%",
    f"{current_f1 - ref_f1:.2f}%"
)

st.write(f"{comparison_method} Std Dev: {ref_std:.2f}%")
st.write(f"{method_choice} Std Dev: {current_std:.2f}%")

st.divider()

# --------------------------------------------------
# Analysis
# --------------------------------------------------

st.subheader("What Do These Results Mean?")

delta_acc = current_acc - ref_acc
delta_f1 = current_f1 - ref_f1

st.write(f"This compares {method_choice} against {comparison_method}.")

if method_choice == "Z-SCORE":
    st.write("Z-score standardises EEG features by removing mean and scaling variance.")
elif method_choice == "SRAN":
    st.write("SRAN removes deeper subject-specific patterns beyond basic scaling.")
elif method_choice == "SRAN_FULL":
    st.write("SRAN_FULL further improves alignment and transfer across subjects.")

if delta_acc > 0:
    st.write(f"Accuracy improved by {delta_acc:.2f}%.")
else:
    st.write(f"Accuracy decreased by {delta_acc:.2f}%.")

if delta_f1 > 0:
    st.write("F1-score improvement indicates better class balance.")
else:
    st.write("F1-score did not improve significantly.")

if current_std < ref_std:
    st.write("Performance is more consistent across subjects.")
else:
    st.write("Variability across subjects remains due to individual differences.")

st.write(
    "The goal is to reduce subject-specific variability and improve generalisation to unseen individuals."
)

# --------------------------------------------------
# Method Comparison (All)
# --------------------------------------------------

st.subheader("Method Comparison")

accs = []
f1s = []

for m in methods:
    _, _, s = get_data(m)
    accs.append(s["mean_accuracy"].iloc[0] * 100)
    f1s.append(s["mean_f1"].iloc[0] * 100)

df_comp = pd.DataFrame({
    "Method": methods,
    "Accuracy": accs,
    "F1": f1s
})

fig_bar = px.bar(df_comp, x="Method", y=["Accuracy", "F1"], barmode="group")
st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------
# Per Subject Improvement
# --------------------------------------------------

st.subheader("Per-Subject Improvement")

ref_plot = ref_results.rename(columns={"accuracy": "Accuracy", "subject": "Subject"})
current_plot = results.rename(columns={"accuracy": "Accuracy", "subject": "Subject"})

improvement = (current_plot["Accuracy"] - ref_plot["Accuracy"]) * 100

fig_imp = go.Figure()
fig_imp.add_trace(go.Bar(
    x=ref_plot["Subject"],
    y=improvement,
    marker_color=["green" if v > 0 else "red" for v in improvement]
))

st.plotly_chart(fig_imp, use_container_width=True)

positive = (improvement > 0).sum()
st.write(f"The model improved for {positive} out of 20 subjects.")

# --------------------------------------------------
# Variability
# --------------------------------------------------

st.subheader("Inter-Subject Variability")
st.write(
    "This chart visualises inter-subject variability in model performance. "
    "Each point represents a subject, while the box shows the overall distribution. "
    "A tighter box (lower spread) indicates more consistent performance across individuals, "
    "which suggests better generalisation."
)
df_box = pd.DataFrame({
    "Accuracy": list(ref_plot["Accuracy"] * 100) + list(current_plot["Accuracy"] * 100),
    "Method": [comparison_method] * 20 + [method_choice] * 20
})

fig_box = px.box(df_box, x="Method", y="Accuracy", points="all")
st.plotly_chart(fig_box, use_container_width=True)

st.caption("Lower variability is preferred, as it indicates the model performs more consistently across different individuals.")

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

st.subheader("Confusion Matrix")

st.write(
    "This confusion matrix shows how well the model classifies each emotion. "
    "Each row represents the true emotion, and each column represents the predicted emotion. "
    "Values closer to 1 (darker cells) indicate correct predictions, especially along the diagonal. "
    "Off-diagonal values represent misclassifications."
)

labels = [0, 1, 2]  # numeric labels in your dataset
label_names = ["Positive", "Neutral", "Negative"]

cm = pd.crosstab(predictions["true"], predictions["pred"])

# Force all classes to appear (VERY IMPORTANT)
cm = cm.reindex(index=labels, columns=labels, fill_value=0)

cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)

fig_cm = px.imshow(
    cm_norm,
    text_auto=".2f",
    x=label_names,
    y=label_names,
    color_continuous_scale="Blues"
)

st.plotly_chart(fig_cm, use_container_width=True)

st.caption(
    "Ideal case: most values should be concentrated along the diagonal (top-left to bottom-right)."
)
# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

st.header("Model Comparison")

models = ["svm", "rf", "knn"]
names = ["SVM", "RF", "KNN"]

accs = []
f1s = []

for m in models:
    path_full = f"{m}/summary_sran_full_{m}.csv"

    if os.path.exists(path_full):
        t = pd.read_csv(path_full)
    else:
        t = pd.read_csv(f"{m}/summary_sran_{m}.csv")

    accs.append(t["mean_accuracy"].iloc[0] * 100)
    f1s.append(t["mean_f1"].iloc[0] * 100)

df_models = pd.DataFrame({
    "Model": names,
    "Accuracy": accs,
    "F1": f1s
})

fig_models = px.bar(df_models, x="Model", y=["Accuracy", "F1"], barmode="group")
st.plotly_chart(fig_models, use_container_width=True)

best = df_models.loc[df_models["Accuracy"].idxmax()]
# st.write(f"Best Model: {best['Model']} ({best['Accuracy']:.2f}%)")

# --------------------------------------------------
# Tables
# --------------------------------------------------

st.subheader("Results Tables")

tab1, tab2 = st.tabs([comparison_method, method_choice])

with tab1:
    st.dataframe(ref_results)

with tab2:
    st.dataframe(results)

st.download_button(
    "Download Current Results",
    results.to_csv(index=False),
    "results.csv"
)