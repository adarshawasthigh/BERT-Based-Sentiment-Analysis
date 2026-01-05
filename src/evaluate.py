import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from transformers import Trainer

# Assumes trainer is already created during training
predictions = trainer.predict(test_dataset)

preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids
probs = predictions.predictions[:, 1]

cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

precision, recall, _ = precision_recall_curve(labels, probs)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f"AUC={pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
