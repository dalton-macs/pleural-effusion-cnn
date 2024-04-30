import os
import pandas as pd
from sklearn.metrics import matthews_corrcoef

def gen_metrics(file_path: str, save_path: str, true_col: str = 'labels', 
                pred_col: str = 'predictions') -> pd.DataFrame:
    df = pd.read_csv(file_path)

    tp = ((df[true_col] == 1) & (df[pred_col] == 1)).sum()
    fp = ((df[true_col] == 0) & (df[pred_col] == 1)).sum()
    fn = ((df[true_col] == 1) & (df[pred_col] == 0)).sum()
    tn = ((df[true_col] == 0) & (df[pred_col] == 0)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    mcc = matthews_corrcoef(df[true_col], df[pred_col])

    metrics_df = pd.DataFrame({'TP': [tp], 'TN': [tn], 'FP': [fp], 'FN': [fn],
                               'Accuracy': [accuracy], 'Precision': [precision],
                               'Recall': [recall], 'F1': [f1_score],
                               'Mathews Correlation Cefficient': [mcc]})
    
    if not os.path.isdir(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    
    metrics_df.to_csv(save_path)

    return metrics_df