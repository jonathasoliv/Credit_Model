import pandas as pd
import numpy as np

COLS_LAG = [
    'target_lag_1', 'target_lag_2', 'target_lag_3',
    'inad_ult_3', 'atraso_ult_3', 'atraso_ult_6',
    'taxa_hist_inad', 'atraso_hist_medio',
    'n_cobrancas_hist', 'valor_medio_cliente',
]

def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Células [13] — lags, rolling, expanding com shift(1) dentro do groupby."""
    df = df.sort_values(['ID_CLIENTE', 'SAFRA_REF', 'DATA_VENCIMENTO']).reset_index(drop=True)

    # Lags
    for lag in [1, 2, 3]:
        df[f'target_lag_{lag}'] = df.groupby('ID_CLIENTE')['target'].shift(lag)

    # Rolling
    df['inad_ult_3'] = (
        df.groupby('ID_CLIENTE')['target']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['atraso_ult_3'] = (
        df.groupby('ID_CLIENTE')['dias_atraso']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['atraso_ult_6'] = (
        df.groupby('ID_CLIENTE')['dias_atraso']
        .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    )

    # Expanding
    df['taxa_hist_inad'] = (
        df.groupby('ID_CLIENTE')['target']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['atraso_hist_medio'] = (
        df.groupby('ID_CLIENTE')['dias_atraso']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['n_cobrancas_hist'] = (
        df.groupby('ID_CLIENTE')['target']
        .transform(lambda x: x.shift(1).expanding().count())
    )
    df['valor_medio_cliente'] = (
        df.groupby('ID_CLIENTE')['VALOR_A_PAGAR']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Flag cliente novo (antes do fillna)
    df['flag_cliente_novo'] = df['target_lag_1'].isna().astype(int)

    return df


def build_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Células [15] — features de contexto temporal e da cobrança."""
    df['DATA_CADASTRO'] = pd.to_datetime(df['DATA_CADASTRO'], errors='coerce')
    df['DATA_EMISSAO_DOCUMENTO'] = pd.to_datetime(df['DATA_EMISSAO_DOCUMENTO'], errors='coerce')

    df['tempo_cliente'] = (df['DATA_VENCIMENTO'] - df['DATA_CADASTRO']).dt.days
    df['dias_prazo_cobranca'] = (df['DATA_VENCIMENTO'] - df['DATA_EMISSAO_DOCUMENTO']).dt.days
    df['mes_safra'] = pd.to_datetime(df['SAFRA_REF']).dt.month
    df['trimestre_safra'] = pd.to_datetime(df['SAFRA_REF']).dt.quarter

    return df


def impute_missing(df: pd.DataFrame, train_medians: pd.Series = None) -> tuple:
    """
    Células [17] — imputação consistente.
    Se train_medians=None, calcula e retorna as medianas (uso no treino).
    Se train_medians fornecida, aplica sem recalcular (uso no teste).
    """
    df[COLS_LAG] = df[COLS_LAG].fillna(0)

    num_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in COLS_LAG and df[c].isna().any()
    ]

    if train_medians is None:
        train_medians = df[num_cols].median()

    for col in num_cols:
        if col in train_medians.index:
            df[col] = df[col].fillna(train_medians[col])

    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].fillna('DESCONHECIDO')

    return df, train_medians