# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 08:20:23 2025

@author: Flavio Simoes da Silva

Purpose: Master's Thesis 
    Portfolio Optimization and NLP:
    Algorithmic trading with Black-Litterman and views from stock market sentiment coming from StockTwits.
    
    Supervisor: M.-A Divernois, Ph.D
"""
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report



#================= Load Sentiment Data from data_tokenizer.py =================
df_SCSSR = pd.concat([pd.read_pickle('df_SCSSR_1_with_sentiment.pkl'), 
                      pd.read_pickle('df_SCSSR_2_with_sentiment.pkl')])

# test = df_SCSSR.tail(50000)
# test.to_csv('test.csv')


#================= Evaluate Sentiment Model Performance =================
# Mapping predicted sentiment to numeric values
predicted_mapping = {"bearish": -1, "neutral": 0, "bullish": 1}
df_SCSSR["predicted_sentiment_num"] = df_SCSSR["predicted_sentiment"].map(predicted_mapping)

# Filtering out rows where 'sent' is 0
df_filtered = df_SCSSR[df_SCSSR["sent"] != 0]

# Get true and predicted labels
y_true = df_filtered["sent"]
y_pred = df_filtered["predicted_sentiment_num"]

# Compute classification metrics
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
print(report_df)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[-1, 1, 0])

# Convert confusion matrix to DataFrame
cm = pd.DataFrame(
    cm[[1, 0], :],
    index=["Bullish", "Bearish"],
    columns=["Bullish", "Neutral", "Bearish"]
)

# Display the corrected confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm.T, annot=True, fmt="d", cmap="Reds", cbar=False)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.show()






from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

# 1. Keep only tweets with an explicit user label (±1)
mask_lbl = df_SCSSR['sent'].isin([-1, 1])
df_eval   = df_SCSSR.loc[mask_lbl].copy()

# 2. Ground truth string labels
mapping   = {1: 'bullish', -1: 'bearish'}
y_true    = df_eval['sent'].map(mapping)
y_pred    = df_eval['predicted_sentiment']  # model output

# 3. Classification report + confusion matrix
print(classification_report(y_true, y_pred, digits=3))

cm = confusion_matrix(y_true, y_pred, labels=['bullish', 'bearish'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['bull', 'bear'],
            yticklabels=['bull', 'bear'])
plt.title("Confusion matrix: user-labelled vs FinTwitBERT (sent ∈ {±1})")
plt.xlabel("Model prediction"); plt.ylabel("User label")
plt.tight_layout(); plt.savefig("cm_user_vs_model.png", dpi=300)
plt.show()

# 4. What about the 0-label tweets?
pct_zeros = 100 * (~mask_lbl).mean()
print(f"{pct_zeros:.1f}% of messages have no user sentiment tag (sent == 0).")




# ------------------------------------------------------------------
# 0.  Keep only tweets with an explicit user label  (sent ∈ {+1,-1})
# ------------------------------------------------------------------
mask_lbl = df_SCSSR["sent"].isin([-1, 1])
df_eval   = df_SCSSR.loc[mask_lbl].copy()

# ------------------------------------------------------------------
# 1.  Ground-truth and model labels
# ------------------------------------------------------------------
mapping = {1: "bullish", -1: "bearish"}      # True classes (no neutral!)
y_true  = df_eval["sent"].map(mapping)
y_pred  = df_eval["predicted_sentiment"]

# ------------------------------------------------------------------
# 2.  Standard metrics on bull/bear only
# ------------------------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(
        y_true, y_pred,
        digits=3, labels=["bullish", "bearish"]))

cm_2x2 = confusion_matrix(
        y_true, y_pred, labels=["bullish", "bearish"])

sns.heatmap(cm_2x2, annot=True, fmt='d', cmap='Blues',
            xticklabels=['bull','bear'],
            yticklabels=['bull','bear'])
plt.title("2×2 confusion matrix (human-labelled subset)")
plt.xlabel("Model prediction"); plt.ylabel("User label")
plt.tight_layout(); plt.savefig("cm_2x2.png", dpi=300)
plt.show()

# ------------------------------------------------------------------
# 3.  2×3 matrix – how often the model says “neutral”
# ------------------------------------------------------------------
import numpy as np, pandas as pd

labels_true = ["bullish", "bearish"]
labels_pred = ["bullish", "bearish", "neutral"]

cm_2x3 = (pd.crosstab(y_true, y_pred, dropna=False)
            .reindex(index=labels_true, columns=labels_pred)
            .fillna(0).astype(int))

sns.heatmap(cm_2x3, annot=True, fmt='d', cmap='Purples',
            xticklabels=['bull','bear','neut'],
            yticklabels=['bull','bear'])
plt.title("2×3 confusion matrix (model outputs incl. “neutral”)")
plt.xlabel("Model prediction"); plt.ylabel("User label")
plt.tight_layout(); plt.savefig("cm_2x3.png", dpi=300)
plt.show()

print("\n2×3 confusion matrix counts:\n", cm_2x3)

# ------------------------------------------------------------------
# 4.  Share of tweets with no user sentiment tag
# ------------------------------------------------------------------
pct_zeros = (~mask_lbl).mean() * 100
print(f"{pct_zeros:.1f}% of messages have no user tag (sent == 0).")










# #================= Evaluate Sentiment Model Performance =================
# # Mapping predicted sentiment to numeric values
# predicted_mapping = {"bearish": -1, "neutral": 0, "bullish": 1}
# df_SCSSR["predicted_sentiment_num"] = df_SCSSR["predicted_sentiment"].map(predicted_mapping)

# # Filtering out rows where 'sent' is 0, as they don't express sentiment
# df_filtered = df_SCSSR[df_SCSSR["sent"] != 0]

# # Extracting true and predicted labels
# y_true = df_filtered["sent"]
# y_pred = df_filtered["predicted_sentiment_num"]

# # Compute classification metrics
# report = classification_report(y_true, y_pred, output_dict=True)
# report_df = pd.DataFrame(report).transpose()
# print("Classification Report:")
# print(report_df)

# # Compute confusion matrix with all three predicted categories
# cm_final_fixed = confusion_matrix(y_true, y_pred, labels=[-1, 1, 0])

# # Convert confusion matrix to DataFrame, keeping only true bearish and bullish while retaining neutral predictions
# cm_df_final_fixed = pd.DataFrame(
#     cm_final_fixed[:2, :],  # Select only the first two rows (true bearish, true bullish)
#     index=["True Bearish", "True Bullish"], 
#     columns=["Pred Bearish", "Pred Neutral", "Pred Bullish"]
# ).T  # Transpose the matrix

# # Display the corrected confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm_df_final_fixed, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("True Labels")
# plt.ylabel("Predicted Labels")
# plt.title("Confusion Matrix (Transposed)")
# plt.show()












































# # Mapping predictions to numerical values
# sentiment_mapping = {"bullish": 1, "bearish": -1, "neutral": 0}

# # Filtering out rows where `sent` is 0 (no stated sentiment)
# df_filtered = df_SCSSR[df_SCSSR["sent"] != 0].copy()

# # Convert true and predicted sentiments to numerical values
# df_filtered["predicted_sentiment"] = df_filtered["predicted_sentiment"].map(sentiment_mapping)

# # Extracting true and predicted labels
# y_true = df_filtered["sent"]
# y_pred = df_filtered["predicted_sentiment"]

# # Compute accuracy
# accuracy = accuracy_score(y_true, y_pred)

# # Compute precision, recall, F1-score
# precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

# # Compute confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, -1])  # 1 (Bullish), -1 (Bearish)

# # Compile results
# metrics = {
#     "Accuracy": accuracy,
#     "Precision": precision,
#     "Recall": recall,
#     "F1-score": f1
# }


# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Define class labels
# class_labels = ["Bullish (1)", "Bearish (-1)"]

# # Plot confusion matrix
# plt.figure(figsize=(6, 5))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()
