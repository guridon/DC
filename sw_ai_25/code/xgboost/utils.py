from dataset import Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def metrics(y_val, val_probs, dataset : Dataset):
    print(f"[Metrics] Calculating evaluation metrics...")
    auc = roc_auc_score(y_val, val_probs)
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    val_preds = (val_probs >= 0.5).astype(int)
    f1 = f1_score(y_val, val_preds)   
    metric_scores={
        "⚠️ Validation AUC": auc,
        "⚠️ Validation Precision": precision.mean(), 
        "⚠️ Validation F1-score": f1,
        "⚠️ Validation Recall":recall.mean()
    }
    for key, value in metric_scores.items():
        print(f"{key}: {value:.5f}")

    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(f"precision_recall_curve_{dataset.now}.png")
    plt.show()
    dataset.log(metric_scores)
    return 
