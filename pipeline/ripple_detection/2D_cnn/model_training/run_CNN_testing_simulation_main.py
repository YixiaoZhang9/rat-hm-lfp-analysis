from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from demo_training_functions_m import *
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torchvision import models

# %%
# set the path to images, and architecture.
"""
1. image_path is a directory of spectrogram images (jpg) containing subfolders `train`, `test`, and `valid`.
2. train and valid have subfolders `Yes` and `No` of positive and negative image cases, respectively.
3. test has uncategorized test images.
Check to confirm more than 1 image in each subfolder.
"""
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd()
sim_root = project_root / "data" / "simulation"
image_path = sim_root / "spectral_image" / "main"
check_image_count(image_path)
data = get_the_data(image_path, models.resnet34)

# Apply the CNN to `test` data
# the results is a dictionary, keys : image_number; prediction; probability; predicted_label
code_dir = Path.cwd()
filename = code_dir / "retraining_saved_model.pth"
test_results = test_the_model(data, models.resnet34, filename)
label_predicted = test_results["prediction"].to_numpy()
# load the label of testing dataset
label_dir = image_path / "test"
npz_files = sorted(label_dir.glob("*.npz"))
label_test = []
for file in npz_files:
    d = np.load(file, allow_pickle=True)
    label_test.append(d["label"])
label_test = np.array(label_test)

# %%
# compute the precision; recall; F1 ad plot the results
accuracy = accuracy_score(label_test, label_predicted)
precision = precision_score(label_test, label_predicted, zero_division=0)
recall = recall_score(label_test, label_predicted, zero_division=0)
f1 = f1_score(label_test, label_predicted, zero_division=0)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 score : {f1:.4f}")
# Print a full classification report
print(classification_report(label_test, label_predicted, digits=4, zero_division=0))

# Plot confusion matrix
cm = confusion_matrix(label_test, label_predicted)
plt.figure(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot ROC curve
y_score = test_results["probability"].to_numpy()
fpr, tpr, _ = roc_curve(label_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# plot Precision-Recall curve
precision_curve, recall_curve, _ = precision_recall_curve(label_test, y_score)
ap = average_precision_score(label_test, y_score)

plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve, label=f"AP = {ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()


# plot Predicted probability
y_score = test_results["probability"].to_numpy()
# Separate probabilities according to the true label
pos_prob = y_score[label_test == 1]
neg_prob = y_score[label_test == 0]
plt.figure(figsize=(6, 5))

plt.hist(neg_prob, bins=40, alpha=0.6, label="Negative (GT=0)", density=True)

plt.hist(pos_prob, bins=40, alpha=0.6, label="Positive (GT=1)", density=True)

plt.xlabel("Predicted Probability (Positive class)")
plt.ylabel("Density")
plt.title("Prediction Probability Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# plot calibration curve
prob_true, prob_pred = calibration_curve(
    label_test, y_score, n_bins=10, strategy="uniform"
)
# Plot reliability curve
plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Reliability Curve (Calibration Plot)")
plt.legend()
plt.tight_layout()
plt.show()
