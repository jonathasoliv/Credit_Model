import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
#from scipy.stats import ks_2samp
#from sklearn.metrics import roc_auc_score
import os
import sys

# ── page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk · Inadimplência",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #c8cdd8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e8eaf0 !important; }

/* main bg */
.main { background: #0d1117; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

/* metric cards */
.metric-card {
    background: #161b27;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a6478;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #e8eaf0;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
}
.metric-sub {
    font-size: 11px;
    color: #3d8bcd;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

/* risk badges */
.badge-alto   { background:#3d1a1a; color:#f87171; border:1px solid #7f1d1d; border-radius:4px; padding:2px 10px; font-size:11px; font-weight:500; font-family:'IBM Plex Mono',monospace; }
.badge-medio  { background:#2d2412; color:#fbbf24; border:1px solid #78350f; border-radius:4px; padding:2px 10px; font-size:11px; font-weight:500; font-family:'IBM Plex Mono',monospace; }
.badge-baixo  { background:#0d2618; color:#34d399; border:1px solid #065f46; border-radius:4px; padding:2px 10px; font-size:11px; font-weight:500; font-family:'IBM Plex Mono',monospace; }

/* section header */
.section-header {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d8bcd;
    font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid #1e2a3a;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* dataframe tweaks */
[data-testid="stDataFrame"] { border: 1px solid #1e2a3a !important; border-radius: 6px; }

/* hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── constants ──────────────────────────────────────────────────
#CUTOFF      = '2018-10-01'
ARTEFATOS   = 'artefatos'
COLS_LAG    = [
    'target_lag_1','target_lag_2','target_lag_3',
    'inad_ult_3','atraso_ult_3','atraso_ult_6',
    'taxa_hist_inad','atraso_hist_medio',
    'n_cobrancas_hist','valor_medio_cliente',
]
FEATURES_HIST = COLS_LAG + ['flag_cliente_novo']
RISCO_COLOR = {'ALTO_RISCO':'#f87171','MEDIO_RISCO':'#fbbf24','BAIXO_RISCO':'#34d399'}

# ── helpers ────────────────────────────────────────────────────
def classificar_risco_dinamico(probs: pd.Series, pct_alto: int, pct_medio: int):
    """
    Classifica risco por percentil da distribuição real de scores.
    pct_alto  = % da carteira marcada como ALTO  (ex: 10 -> top 10%)
    pct_medio = % adicional marcada como MEDIO   (ex: 20 -> top 10-30%)
    Thresholds calculados sobre os scores reais — sem valores fixos.
    """
    t_alto  = float(probs.quantile(1 - pct_alto / 100))
    t_medio = float(probs.quantile(1 - (pct_alto + pct_medio) / 100))

    def _clf(p):
        if p >= t_alto:  return 'ALTO_RISCO'
        if p >= t_medio: return 'MEDIO_RISCO'
        return 'BAIXO_RISCO'

    return probs.map(_clf), t_alto, t_medio

@st.cache_resource(show_spinner=False)
def load_artefatos():

    try:
        model         = joblib.load(f'{ARTEFATOS}/model_final.pkl')
        train_medians = joblib.load(f'{ARTEFATOS}/train_medians.pkl')
        features      = joblib.load(f'{ARTEFATOS}/features.pkl')
        cat_cols      = joblib.load(f'{ARTEFATOS}/cat_cols.pkl')
        cutoff        = joblib.load(f'{ARTEFATOS}/cutoff.pkl')

        return (
            model,
            train_medians,
            features,
            cat_cols,
            cutoff,
            None
        )

    except Exception as e:

        return (
            None,
            None,
            None,
            None,
            None,
            str(e)
        )
   
# ==========================================================
# Carregar artefatos
# ==========================================================

model, train_medians, features, cat_cols, CUTOFF, err = load_artefatos()

if err:
    st.error(f"Erro ao carregar artefatos: {err}")
    st.stop()

st.sidebar.success(f"Cutoff carregado: {CUTOFF}")

def build_context_features(df):
    df['DATA_CADASTRO']          = pd.to_datetime(df['DATA_CADASTRO'], errors='coerce')
    df['DATA_EMISSAO_DOCUMENTO'] = pd.to_datetime(df['DATA_EMISSAO_DOCUMENTO'], errors='coerce')
    df['DATA_VENCIMENTO']        = pd.to_datetime(df['DATA_VENCIMENTO'], errors='coerce')
    df['tempo_cliente']          = (df['DATA_VENCIMENTO'] - df['DATA_CADASTRO']).dt.days
    df['dias_prazo_cobranca']    = (df['DATA_VENCIMENTO'] - df['DATA_EMISSAO_DOCUMENTO']).dt.days
    df['mes_safra']              = pd.to_datetime(df['SAFRA_REF']).dt.month
    df['trimestre_safra']        = pd.to_datetime(df['SAFRA_REF']).dt.quarter
    return df

def predict_pipeline(teste_raw, df_historico, model, train_medians, features, cat_cols, cutoff):
    # 1. histórico por cliente
    hist_base = df_historico.copy()
    hist_base['DATA_VENCIMENTO'] = pd.to_datetime(hist_base['DATA_VENCIMENTO'], errors='coerce')
    hist_cliente = (
        hist_base[hist_base['DATA_VENCIMENTO'] < cutoff]
        .sort_values(['ID_CLIENTE','DATA_VENCIMENTO'])
        .groupby('ID_CLIENTE')[FEATURES_HIST]
        .last()
        .reset_index()
    )

    # 2. merge + features
    feat = teste_raw.copy()
    feat = feat.merge(hist_cliente, on='ID_CLIENTE', how='left')
    feat['flag_cliente_novo'] = feat['taxa_hist_inad'].isna().astype(int)
    feat = build_context_features(feat)

    # 3. imputação
    feat[COLS_LAG] = feat[COLS_LAG].fillna(0)
    for col in train_medians.index:
        if col in feat.columns:
            feat[col] = feat[col].fillna(train_medians[col])
    for col in feat.select_dtypes(include=['object','string']).columns:
        feat[col] = feat[col].fillna('DESCONHECIDO')

    # 4. encoding + predição
    X = feat[features].copy()
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')

    probs = model.predict_proba(X)[:,1]

    resultado = feat[['ID_CLIENTE','SAFRA_REF']].copy()
    resultado['PROBABILIDADE'] = probs
    resultado['PRIORIDADE']    = resultado['PROBABILIDADE'].rank(ascending=False).astype(int)
    return resultado.sort_values('PROBABILIDADE', ascending=False).reset_index(drop=True)

def metric_card(label, value, sub=''):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>"""

def badge_html(faixa):
    cls = {'ALTO_RISCO':'badge-alto','MEDIO_RISCO':'badge-medio','BAIXO_RISCO':'badge-baixo'}
    label = faixa.replace('_RISCO','').replace('_',' ')
    return f'<span class="{cls.get(faixa,"")}">  {label}  </span>'

# ── sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Credit Risk")
    st.markdown("<div style='font-size:12px;color:#5a6478;margin-bottom:24px'>Previsão de Inadimplência</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Bases carregadas automaticamente**")

    st.success("✓ base_pagamentos_teste.csv")
    st.success("✓ base_pagamentos_desenvolvimento.csv")
    st.success("✓ base_info.csv")
    st.success("✓ base_cadastral.csv")


    st.markdown("---")
    st.markdown("**Segmentação de Risco**")
    st.markdown("<div style='font-size:11px;color:#5a6478;margin-bottom:8px'>Thresholds por percentil da carteira</div>", unsafe_allow_html=True)

    pct_alto = st.slider(
        "🔴 Alto risco — top X% da carteira",
        min_value=1, max_value=30, value=10, step=1,
        help="Ex: 10 = os 10% de clientes com maior score são ALTO RISCO"
    )
    pct_medio = st.slider(
        "🟡 Médio risco — próximos X%",
        min_value=5, max_value=40, value=20, step=5,
        help="Ex: 20 = os 20% seguintes são MÉDIO RISCO"
    )

    # validação: alto + medio não pode passar de 95%
    if pct_alto + pct_medio > 95:
        st.warning("Alto + Médio ultrapassa 95% da carteira. Reduza um dos valores.")
        pct_medio = max(5, 95 - pct_alto)

    st.markdown("---")
    st.markdown("**Filtros de visualização**")
    faixas_selecionadas = st.multiselect(
        "Faixa de risco",
        ['ALTO_RISCO','MEDIO_RISCO','BAIXO_RISCO'],
        default=['ALTO_RISCO','MEDIO_RISCO','BAIXO_RISCO'],
    )
    top_n = st.slider("Top N clientes", min_value=10, max_value=500, value=100, step=10)

    st.markdown("---")
    st.markdown(f"<div style='font-size:11px;color:#3a4255'>modelo: LightGBM<br>cutoff: {CUTOFF}</div>", unsafe_allow_html=True)

# ── main ───────────────────────────────────────────────────────
st.markdown("<h1 style='color:#e8eaf0;font-size:24px;font-weight:600;margin-bottom:4px'>Modelo de Risco de Inadimplência</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#5a6478;font-size:13px;margin-bottom:32px'>Ranking de clientes por probabilidade de não pagamento</div>", unsafe_allow_html=True)

# ── estado da sessão ───────────────────────────────────────────
if 'resultado' not in st.session_state:
    st.session_state.resultado = None

# ── processar quando arquivos carregados ───────────────────────
if st.session_state.resultado is None:
    with st.spinner("Processando pipeline de inferência..."):
        try:
            
            teste_raw = pd.read_csv('data/base_pagamentos_teste.csv',sep=';')

            df_historico = pd.read_csv('data/base_pagamentos_desenvolvimento.csv', sep=';')
                
            info = pd.read_csv('data/base_info.csv', sep=';')
                
            cadastral = pd.read_csv('data/base_cadastral.csv', sep=';')

            info = pd.read_csv('data/base_info.csv', sep=';')
            cadastral = pd.read_csv('data/base_cadastral.csv', sep=';')


            for base in [teste_raw, df_historico, info, cadastral]:
                base.columns = base.columns.str.strip()

            teste_raw = teste_raw.merge(
                info,
                on=['ID_CLIENTE', 'SAFRA_REF'],
                how='left'
            )

            teste_raw = teste_raw.merge(
                cadastral,
                on='ID_CLIENTE',
                how='left'
            )

            df_historico = df_historico.merge(
                info,
                on=['ID_CLIENTE', 'SAFRA_REF'],
                how='left'
            )

            df_historico = df_historico.merge(
                cadastral,
                on='ID_CLIENTE',
                how='left'
            )

            # ── calcular dias_atraso e target SEMPRE, logo após merge ──
            for col in ['DATA_VENCIMENTO', 'DATA_PAGAMENTO']:
                if col in df_historico.columns:
                    df_historico[col] = pd.to_datetime(df_historico[col], errors='coerce')

            # dias_atraso: pagamentos em aberto (sem DATA_PAGAMENTO) recebem 999
            df_historico['dias_atraso'] = (
                df_historico['DATA_PAGAMENTO'] - df_historico['DATA_VENCIMENTO']
            ).dt.days.fillna(999)

            df_historico['target'] = (df_historico['dias_atraso'] >= 5).astype(int)

            # features comportamentais no histórico (se não existirem)
            if 'taxa_hist_inad' not in df_historico.columns:
                df_historico = df_historico.sort_values(['ID_CLIENTE','DATA_VENCIMENTO'])
                df_historico['taxa_hist_inad'] = (
                    df_historico.groupby('ID_CLIENTE')['target']
                    .transform(lambda x: x.shift(1).expanding().mean())
                )
                for lag in [1,2,3]:
                    df_historico[f'target_lag_{lag}'] = (
                        df_historico.groupby('ID_CLIENTE')['target'].shift(lag)
                    )
                for col, src in [('inad_ult_3','target'),('atraso_ult_3','dias_atraso'),('atraso_ult_6','dias_atraso')]:
                    w = 3 if '3' in col else 6
                    df_historico[col] = (
                        df_historico.groupby('ID_CLIENTE')[src]
                        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                    )
                df_historico['atraso_hist_medio']   = df_historico.groupby('ID_CLIENTE')['dias_atraso'].transform(lambda x: x.shift(1).expanding().mean())
                df_historico['n_cobrancas_hist']    = df_historico.groupby('ID_CLIENTE')['target'].transform(lambda x: x.shift(1).expanding().count())
                df_historico['valor_medio_cliente'] = df_historico.groupby('ID_CLIENTE')['VALOR_A_PAGAR'].transform(lambda x: x.shift(1).expanding().mean())
                df_historico['flag_cliente_novo']   = df_historico['target_lag_1'].isna().astype(int)
                df_historico[FEATURES_HIST]         = df_historico[FEATURES_HIST].fillna(0)

            st.session_state.resultado = predict_pipeline(
                teste_raw, df_historico, model, train_medians, features, cat_cols, CUTOFF
            )
            st.success(f"Pipeline concluído — {len(st.session_state.resultado):,} clientes processados")
        except Exception as e:
            st.error(f"Erro no pipeline: {e}")
            st.stop()

# ── dashboard ──────────────────────────────────────────────────
if st.session_state.resultado is not None:
    res = st.session_state.resultado.copy()

    # ── classificação dinâmica por percentil ──
    res['FAIXA_RISCO'], t_alto, t_medio = classificar_risco_dinamico(
        res['PROBABILIDADE'], pct_alto, pct_medio
    )

    # ── info dos thresholds no sidebar ──
    st.sidebar.markdown(
        f"<div style='font-size:11px;color:#5a6478;margin-top:8px'>"
        f"Threshold ALTO &nbsp;≥ <b style='color:#f87171'>{t_alto:.3f}</b><br>"
        f"Threshold MÉDIO ≥ <b style='color:#fbbf24'>{t_medio:.3f}</b>"
        f"</div>",
        unsafe_allow_html=True
    )

    # aplicar filtros de visualização
    res_filtrado = res[res['FAIXA_RISCO'].isin(faixas_selecionadas)].head(top_n)

    # ── métricas ──
    total      = len(res)
    alto       = (res['FAIXA_RISCO'] == 'ALTO_RISCO').sum()
    medio      = (res['FAIXA_RISCO'] == 'MEDIO_RISCO').sum()
    baixo      = (res['FAIXA_RISCO'] == 'BAIXO_RISCO').sum()
    media_prob = res['PROBABILIDADE'].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(metric_card("Total clientes", f"{total:,}", "base de teste"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Alto risco", f"{alto:,}", f"{alto/total*100:.1f}% da base"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Médio risco", f"{medio:,}", f"{medio/total*100:.1f}% da base"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Baixo risco", f"{baixo:,}", f"{baixo/total*100:.1f}% da base"), unsafe_allow_html=True)
    with c5: st.markdown(metric_card("Prob. média", f"{media_prob:.1%}", "score médio"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── gráficos ──
    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.markdown('<div class="section-header">Distribuição do Score</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3))
        fig.patch.set_facecolor('#161b27')
        ax.set_facecolor('#161b27')
        ax.hist(res['PROBABILIDADE'], bins=40, color='#3d8bcd', alpha=0.85, edgecolor='none')
        ax.axvline(t_medio, color='#fbbf24', lw=1.2, ls='--', label=f'Médio ({t_medio:.3f})')
        ax.axvline(t_alto,  color='#f87171', lw=1.2, ls='--', label=f'Alto ({t_alto:.3f})')
        ax.set_xlabel('Probabilidade de inadimplência', color='#5a6478', fontsize=9)
        ax.set_ylabel('Frequência', color='#5a6478', fontsize=9)
        ax.tick_params(colors='#5a6478', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#1e2a3a')
        ax.legend(fontsize=8, facecolor='#0f1117', labelcolor='#c8cdd8', edgecolor='#1e2a3a')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_graf2:
        st.markdown('<div class="section-header">Segmentação de Risco</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3))
        fig.patch.set_facecolor('#161b27')
        ax.set_facecolor('#161b27')
        labels = ['Baixo', 'Médio', 'Alto']
        vals   = [baixo, medio, alto]
        colors = ['#34d399','#fbbf24','#f87171']
        bars = ax.barh(labels, vals, color=colors, height=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + total*0.005, bar.get_y()+bar.get_height()/2,
                    f'{v:,}', va='center', color='#c8cdd8', fontsize=9,
                    fontfamily='monospace')
        ax.set_xlabel('Número de clientes', color='#5a6478', fontsize=9)
        ax.tick_params(colors='#5a6478', labelsize=9)
        for spine in ax.spines.values(): spine.set_color('#1e2a3a')
        ax.set_xlim(0, max(vals)*1.15)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── ranking ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Ranking de Clientes por Risco</div>', unsafe_allow_html=True)

    df_show = res_filtrado[['PRIORIDADE','ID_CLIENTE','SAFRA_REF','PROBABILIDADE','FAIXA_RISCO']].copy()
    df_show['PROBABILIDADE'] = df_show['PROBABILIDADE'].map(lambda x: f"{x:.1%}")

    st.dataframe(
        df_show.rename(columns={
            'PRIORIDADE':'#','ID_CLIENTE':'Cliente',
            'SAFRA_REF':'Safra','PROBABILIDADE':'Probabilidade','FAIXA_RISCO':'Faixa'
        }),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    # ── download ──
    st.markdown("<br>", unsafe_allow_html=True)
    col_dl1, col_dl2, _ = st.columns([1,1,3])

    with col_dl1:
        csv_completo = res.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇ Ranking completo (.csv)",
            data=csv_completo,
            file_name="ranking_risco_inadimplencia.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        csv_alto = res[res['FAIXA_RISCO']=='ALTO_RISCO'].to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇ Somente alto risco (.csv)",
            data=csv_alto,
            file_name="clientes_alto_risco.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    # estado vazio
    st.markdown("""
    <div style='text-align:center;padding:80px 40px;color:#3a4255'>
        <div style='font-size:40px;margin-bottom:16px'>📂</div>
        <div style='font-size:15px;font-weight:500;color:#5a6478;margin-bottom:8px'>Nenhum dado carregado</div>
        <div style='font-size:13px;color:#3a4255'>Faça upload da base de teste e da base de desenvolvimento<br>no painel lateral para gerar o ranking de risco.</div>
    </div>
    """, unsafe_allow_html=True)