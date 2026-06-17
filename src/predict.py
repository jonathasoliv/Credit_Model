import pandas as pd
import numpy as np
from src.features import build_behavioral_features, build_context_features, impute_missing, COLS_LAG


FEATURES_HIST = [
    'target_lag_1', 'target_lag_2', 'target_lag_3',
    'inad_ult_3', 'atraso_ult_3', 'atraso_ult_6',
    'taxa_hist_inad', 'atraso_hist_medio',
    'n_cobrancas_hist', 'valor_medio_cliente',
    'flag_cliente_novo',
]


def classificar_risco(prob: float) -> str:
    if prob >= 0.60:
        return 'ALTO_RISCO'
    elif prob >= 0.30:
        return 'MEDIO_RISCO'
    return 'BAIXO_RISCO'


def predict_inadimplencia(
    teste_raw: pd.DataFrame,
    df_historico: pd.DataFrame,
    model,
    train_medians: pd.Series,
    FEATURES: list,
    cat_cols: list,
    CUTOFF: str,
) -> pd.DataFrame:
    """
    Pipeline completo de inferência — replica seções 11.1 a 11.4.
    Recebe dados brutos, retorna ranking de risco pronto para uso.
    """
    # 1. Histórico por cliente (seção 11.1)
    hist_cliente = (
        df_historico[df_historico['DATA_VENCIMENTO'] < CUTOFF]
        .sort_values(['ID_CLIENTE', 'DATA_VENCIMENTO'])
        .groupby('ID_CLIENTE')[FEATURES_HIST]
        .last()
        .reset_index()
    )

    # 2. Merge e features derivadas (seção 11.2)
    feat = teste_raw.merge(hist_cliente, on='ID_CLIENTE', how='left')
    feat['flag_cliente_novo'] = feat['taxa_hist_inad'].isna().astype(int)
    feat = build_context_features(feat)

    # 3. Imputação com medianas do treino (seção 11.3)
    feat, _ = impute_missing(feat, train_medians=train_medians)

    # 4. Encoding e predição (seção 11.4)
    X = feat[FEATURES].copy()
    for col in cat_cols:
        X[col] = X[col].astype('category')

    probs = model.predict_proba(X)[:, 1]

    return (
        feat[['ID_CLIENTE', 'SAFRA_REF']]
        .assign(PROBABILIDADE_INADIMPLENCIA=probs)
        .assign(FAIXA_RISCO=lambda d: d['PROBABILIDADE_INADIMPLENCIA'].map(classificar_risco))
        .sort_values('PROBABILIDADE_INADIMPLENCIA', ascending=False)
        .reset_index(drop=True)
    )