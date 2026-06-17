import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    log_loss, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split


def evaluate_model(model, X_val, y_val) -> dict:
    """Células [31] — métricas padrão de crédito."""
    prob_val = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, prob_val)
    ap  = average_precision_score(y_val, prob_val)
    ks, _ = ks_2samp(prob_val[y_val == 1], prob_val[y_val == 0])

    print(f"AUC-ROC : {auc:.4f}")
    print(f"KS      : {ks:.4f}")
    print(f"Avg Prec: {ap:.4f}")

    return {'auc': auc, 'ks': ks, 'ap': ap, 'prob_val': prob_val}


def calibrate_model(model, X_train, y_train, X_val, y_val):
    """Células [45-47] — calibração com hold-out interno."""
    _, X_hold, _, y_hold = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    calibrador = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
    calibrador.fit(X_hold, y_hold)

    ll_before = log_loss(y_val, model.predict_proba(X_val)[:, 1])
    ll_after  = log_loss(y_val, calibrador.predict_proba(X_val)[:, 1])

    print(f"Log Loss antes : {ll_before:.4f}")
    print(f"Log Loss depois: {ll_after:.4f}")

    return calibrador if ll_after < ll_before else model


def plot_gains_chart(prob_val, y_val):
    """Células [40] — Gains Chart com reset_index correto."""
    df_gains = (
        pd.DataFrame({'prob': prob_val, 'target': y_val.values})
        .sort_values('prob', ascending=False)
        .reset_index(drop=True)
    )
    df_gains['gains'] = df_gains['target'].cumsum() / df_gains['target'].sum()
    df_gains['pop']   = (df_gains.index + 1) / len(df_gains)

    plt.figure(figsize=(7, 4))
    plt.plot(df_gains['pop'] * 100, df_gains['gains'] * 100)
    plt.plot([0, 100], [0, 100], '--')
    plt.title('Gains Chart')
    plt.xlabel('% população')
    plt.ylabel('% inadimplentes capturados')
    plt.tight_layout()
    plt.show()