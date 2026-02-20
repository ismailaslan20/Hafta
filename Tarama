"""
BIST Mum Formasyonu & EMA Tarayıcı
====================================
Kurulum:
    pip install streamlit yfinance pandas numpy plotly

Çalıştırma:
    streamlit run bist_tarayici.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ──────────────────────────────────────────────────────────────────────────────
# SAYFA AYARLARI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIST Formasyon Tarayıcı",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Başlık */
.main-header {
    background: linear-gradient(135deg, #0f1629 0%, #1a2444 50%, #0f1629 100%);
    border: 1px solid #2a3a6e;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 70% 50%, rgba(64,120,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #7a8ab0;
    margin: 0;
    font-size: 0.95rem;
}

/* Metrik kartlar */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.metric-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 18px 22px;
    flex: 1;
    text-align: center;
}
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #4a8cff;
}
.metric-card .lbl {
    font-size: 0.78rem;
    color: #5a6a90;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Sonuç satırı */
.result-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-left: 4px solid #4a8cff;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.result-card.hammer   { border-left-color: #f59e0b; }
.result-card.engulf   { border-left-color: #10b981; }
.result-card.morning  { border-left-color: #8b5cf6; }
.result-card.ema      { border-left-color: #06b6d4; }

.ticker-badge {
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.05rem;
    color: #fff;
    background: #1e2d4a;
    padding: 4px 12px;
    border-radius: 6px;
    min-width: 80px;
    text-align: center;
}

.tag {
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.tag-hammer  { background: rgba(245,158,11,0.15);  color: #f59e0b; }
.tag-engulf  { background: rgba(16,185,129,0.15);  color: #10b981; }
.tag-morning { background: rgba(139,92,246,0.15);  color: #8b5cf6; }
.tag-ema     { background: rgba(6,182,212,0.15);   color: #06b6d4; }

.price-info {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #9ab;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #9ab !important;
    font-size: 0.85rem !important;
}

/* Butonlar */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #4a8cff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(74,140,255,0.3);
}

/* Progress */
.stProgress > div > div { background: #4a8cff !important; }

/* Divider */
hr { border-color: #1e2d4a; }

/* DataFrame */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Info box */
.info-box {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #7a8ab0;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HİSSE LİSTESİ
# ──────────────────────────────────────────────────────────────────────────────
HISSELER = [
    "BINHO.IS","ACSEL.IS","AHSGY.IS","AKYHO.IS","AKFYE.IS","AKHAN.IS",
    "AKSA.IS","ALBRK.IS","ALCTL.IS","ALKIM.IS","ALKA.IS","ALTNY.IS",
    "ALKLC.IS","ALVES.IS","ANGEN.IS","ARDYZ.IS","ARFYE.IS","ASELS.IS",
    "ATAKP.IS","ATATP.IS","AVPGY.IS","AYEN.IS","BAHKM.IS","BAKAB.IS",
    "BNTAS.IS","BANVT.IS","BASGZ.IS","BEGYO.IS","BSOKE.IS","BERA.IS",
    "BRKSN.IS","BESTE.IS","BIENY.IS","BIMAS.IS","BINBN.IS","BRLSM.IS",
    "BMSTL.IS","BORSK.IS","BOSSA.IS","BRISA.IS","BURCE.IS","BURVA.IS",
    "CEMZY.IS","COSMO.IS","CVKMD.IS","CWENE.IS","CANTE.IS","CATES.IS",
    "CELHA.IS","CEMTS.IS","CMBTN.IS","CIMSA.IS","DAPGM.IS","DARDL.IS",
    "DGATE.IS","DCTTR.IS","DMSAS.IS","DENGE.IS","DESPC.IS","DOFER.IS",
    "DOFRB.IS","DGNMO.IS","ARASE.IS","DOGUB.IS","DYOBY.IS","EBEBK.IS",
    "EDATA.IS","EDIP.IS","EFOR.IS","EGGUB.IS","EGPRO.IS","EKSUN.IS",
    "ELITE.IS","EKGYO.IS","ENJSA.IS","EREGL.IS","KIMMR.IS","ESCOM.IS",
    "TEZOL.IS","EUPWR.IS","EYGYO.IS","FADE.IS","FONET.IS","FORMT.IS",
    "FRMPL.IS","FORTE.IS","FZLGY.IS","GEDZA.IS","GENIL.IS","GENTS.IS",
    "GEREL.IS","GESAN.IS","GOODY.IS","GOKNR.IS","GOLTS.IS","GRTHO.IS",
    "GUBRF.IS","GLRMK.IS","GUNDG.IS","GRSEL.IS","HRKET.IS","HATSN.IS",
    "HKTM.IS","HOROZ.IS","IDGYO.IS","IHEVA.IS","IHLGM.IS","IHLAS.IS",
    "IHYAY.IS","IMASM.IS","INTEM.IS","ISDMR.IS","ISSEN.IS","IZFAS.IS",
    "IZINV.IS","JANTS.IS","KRDMA.IS","KRDMB.IS","KRDMD.IS","KARSN.IS",
    "KTLEV.IS","KATMR.IS","KRVGD.IS","KZBGY.IS","KCAER.IS","KOCMT.IS",
    "KLSYN.IS","KNFRT.IS","KONTR.IS","KONYA.IS","KONKA.IS","KRPLS.IS",
    "KOTON.IS","KOPOL.IS","KRGYO.IS","KRSTL.IS","KRONT.IS","KUYAS.IS",
    "KBORU.IS","KUTPO.IS","LMKDC.IS","LOGO.IS","LKMNH.IS","MAKIM.IS",
    "MAGEN.IS","MAVI.IS","MEDTR.IS","MEKAG.IS","MNDRS.IS","MERCN.IS",
    "MEYSU.IS","MPARK.IS","MOBTL.IS","MNDTR.IS","EGEPO.IS","NTGAZ.IS",
    "NETAS.IS","OBAMS.IS","OBASE.IS","OFSYM.IS","ONCSM.IS","ORGE.IS",
    "OSTIM.IS","OZRDN.IS","OZYSR.IS","PNLSN.IS","PAGYO.IS","PARSN.IS",
    "PASEU.IS","PENGD.IS","PENTA.IS","PETKM.IS","PETUN.IS","PKART.IS",
    "PLTUR.IS","POLHO.IS","QUAGR.IS","RNPOL.IS","RODRG.IS","RGYAS.IS",
    "RUBNS.IS","SAFKR.IS","SANEL.IS","SNICA.IS","SANKO.IS","SAMAT.IS",
    "SARKYS.IS","SAYAS.IS","SEKUR.IS","SELEC.IS","SELVA.IS","SRVGY.IS",
    "SILVR.IS","SNGYO.IS","SMRTG.IS","SMART.IS","SOKE.IS","SUNTK.IS",
    "SURGY.IS","SUWEN.IS","TNZTP.IS","TARKM.IS","TKNSA.IS","TDGYO.IS",
    "TUCLK.IS","TUKAS.IS","TUREX.IS","MARBL.IS","TMSN.IS","TUPRS.IS",
    "ULAS.IS","ULUSE.IS","USAK.IS","UCAYM.IS","VAKKO.IS","VANGD.IS",
    "VRGYO.IS","VESBE.IS","YATAS.IS","YEOTK.IS","YUNSA.IS","ZEDUR.IS",
    "ZERGY.IS",
]

PERIYOT_MAP = {
    "Günlük":  ("1d",  90,  "gün"),
    "Haftalık":("1wk", 52,  "hafta"),
    "Aylık":   ("1mo", 36,  "ay"),
}

# ──────────────────────────────────────────────────────────────────────────────
# STRATEJİ FONKSİYONLARI
# ──────────────────────────────────────────────────────────────────────────────

def calc_emas(closes: pd.Series) -> dict:
    return {p: closes.ewm(span=p, adjust=False).mean() for p in [5, 14, 34, 55]}

def is_downtrend(closes, idx, period=5):
    if idx < period: return False
    seg = closes.iloc[idx - period:idx]
    return float(seg.iloc[-1]) < float(seg.iloc[0])

def check_hammer(o, h, l, c, lo_ratio=2.0, up_ratio=0.1):
    body = abs(c - o)
    rng  = h - l
    if rng == 0 or body == 0: return False
    lo_sh = min(o, c) - l
    up_sh = h - max(o, c)
    return lo_sh >= lo_ratio * body and up_sh <= up_ratio * rng

def check_engulfing(df, i):
    """Boğa Yutan formasyonu: i-1 düşüş mumu, i yükseliş mumu ve tamamen yutar."""
    if i < 1: return False
    p = df.iloc[i-1]
    c = df.iloc[i]
    prev_bear = float(p["Close"]) < float(p["Open"])
    curr_bull = float(c["Close"]) > float(c["Open"])
    engulfs   = float(c["Open"]) <= float(p["Close"]) and float(c["Close"]) >= float(p["Open"])
    return prev_bear and curr_bull and engulfs

def check_morning_star(df, i):
    """Sabah Yıldızı: 3 mumlu dip formasyon."""
    if i < 2: return False
    m1 = df.iloc[i-2]
    m2 = df.iloc[i-1]
    m3 = df.iloc[i]
    # 1. mum: büyük düşüş
    body1 = abs(float(m1["Close"]) - float(m1["Open"]))
    bear1 = float(m1["Close"]) < float(m1["Open"])
    # 2. mum: küçük gövde (doji/küçük)
    body2 = abs(float(m2["Close"]) - float(m2["Open"]))
    # 3. mum: güçlü yükseliş
    body3 = abs(float(m3["Close"]) - float(m3["Open"]))
    bull3 = float(m3["Close"]) > float(m3["Open"])
    # Orta mum küçük olmalı, 3. mum ilk mumun yarısını kapatmalı
    mid_close = float(m1["Open"]) + (float(m1["Close"]) - float(m1["Open"])) * 0.5
    closes_gap = float(m3["Close"]) >= mid_close
    return (bear1 and bull3 and body2 < body1 * 0.4
            and body3 >= body1 * 0.5 and closes_gap)

def check_ema_alignment(closes, idx):
    """EMA 5>14>34>55 tam dizilim (yükseliş trendi hizalaması)."""
    if idx < 55: return False, {}
    seg = closes.iloc[:idx+1]
    emas = {p: seg.ewm(span=p, adjust=False).mean().iloc[-1] for p in [5,14,34,55]}
    aligned = emas[5] > emas[14] > emas[34] > emas[55]
    return aligned, emas

# ──────────────────────────────────────────────────────────────────────────────
# VERİ ÇEKME & TARAMA
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, interval, lookback):
    end   = datetime.today()
    delta = timedelta(days=lookback*2)
    start = end - delta
    try:
        df = yf.download(ticker, start=start, end=end,
                         interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = {"Open","High","Low","Close"}
        if not required.issubset(df.columns): return None
        return df.tail(lookback + 10)
    except:
        return None

def scan_all(strategies, interval, lookback, son_n, trend_period, progress_cb=None):
    results = []
    total   = len(HISSELER)

    for idx, ticker in enumerate(HISSELER):
        if progress_cb:
            progress_cb(idx / total, ticker)

        df = fetch_data(ticker, interval, lookback)
        if df is None or len(df) < max(trend_period + 2, 57):
            continue

        closes = df["Close"]
        n      = len(df)
        start_i = max(trend_period + 2, n - son_n)

        for i in range(start_i, n):
            o = float(df["Open"].iloc[i])
            h = float(df["High"].iloc[i])
            l = float(df["Low"].iloc[i])
            c = float(df["Close"].iloc[i])
            date_str = df.index[i].strftime("%Y-%m-%d")

            found = []

            if "🔨 Çekiç" in strategies:
                if check_hammer(o, h, l, c) and is_downtrend(closes, i, trend_period):
                    body = abs(c-o)
                    lo   = min(o,c) - l
                    found.append(("🔨 Çekiç", "hammer", round(lo/body,2) if body else 0))

            if "🟢 Yutan (Engulfing)" in strategies:
                if check_engulfing(df, i) and is_downtrend(closes, i, trend_period):
                    found.append(("🟢 Yutan", "engulf", "—"))

            if "⭐ Sabah Yıldızı" in strategies:
                if check_morning_star(df, i) and is_downtrend(closes, i, trend_period):
                    found.append(("⭐ Sabah Yıldızı", "morning", "—"))

            if "📊 EMA 5-14-34-55 Dizilim" in strategies:
                aligned, emas = check_ema_alignment(closes, i)
                if aligned:
                    found.append(("📊 EMA Dizilim", "ema", f"5:{emas[5]:.2f}"))

            for (label, cls, extra) in found:
                results.append({
                    "Hisse"    : ticker.replace(".IS",""),
                    "Ticker"   : ticker,
                    "Tarih"    : date_str,
                    "Formasyon": label,
                    "Sınıf"    : cls,
                    "Kapanış"  : round(c, 2),
                    "Açılış"   : round(o, 2),
                    "Yüksek"   : round(h, 2),
                    "Düşük"    : round(l, 2),
                    "Ek Bilgi" : extra,
                    "Hafta Farkı": f"{'Bu periyot' if n-1-i==0 else f'{n-1-i} periyot önce'}",
                })

        time.sleep(0.05)

    return pd.DataFrame(results)

# ──────────────────────────────────────────────────────────────────────────────
# MUM GRAFİĞİ
# ──────────────────────────────────────────────────────────────────────────────

def plot_candles(df, ticker, ema_show=True, signal_dates=None):
    fig = go.Figure()

    # Mumlar
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name=ticker,
        increasing_line_color="#10b981",
        decreasing_line_color="#ef4444",
        increasing_fillcolor="#10b981",
        decreasing_fillcolor="#ef4444",
    ))

    # EMA çizgileri
    if ema_show:
        colors = {5:"#f59e0b", 14:"#4a8cff", 34:"#8b5cf6", 55:"#06b6d4"}
        for p, col in colors.items():
            ema = df["Close"].ewm(span=p, adjust=False).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=ema,
                name=f"EMA{p}",
                line=dict(color=col, width=1.5),
                opacity=0.85,
            ))

    # Sinyal işaretleri
    if signal_dates:
        sig_df = df[df.index.strftime("%Y-%m-%d").isin(signal_dates)]
        if not sig_df.empty:
            fig.add_trace(go.Scatter(
                x=sig_df.index,
                y=sig_df["Low"] * 0.985,
                mode="markers",
                marker=dict(symbol="triangle-up", size=14,
                            color="#f59e0b", line=dict(color="#fff", width=1)),
                name="Sinyal",
            ))

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1220",
        font=dict(family="Syne", color="#9ab"),
        xaxis=dict(
            gridcolor="#1e2d4a", showgrid=True,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(gridcolor="#1e2d4a", showgrid=True),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# ANA UYGULAMA
# ──────────────────────────────────────────────────────────────────────────────

# Başlık
st.markdown("""
<div class="main-header">
  <h1>📈 BIST Formasyon Tarayıcı</h1>
  <p>223 BIST hissesinde mum formasyonları & EMA dizilimi tara</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Tarama Ayarları")
    st.divider()

    periyot = st.selectbox(
        "📅 Periyot",
        options=list(PERIYOT_MAP.keys()),
        index=1,
    )
    interval, lookback_default, periyot_lbl = PERIYOT_MAP[periyot]

    st.markdown("---")

    strategies = st.multiselect(
        "🎯 Stratejiler",
        options=[
            "🔨 Çekiç",
            "🟢 Yutan (Engulfing)",
            "⭐ Sabah Yıldızı",
            "📊 EMA 5-14-34-55 Dizilim",
        ],
        default=["🔨 Çekiç", "📊 EMA 5-14-34-55 Dizilim"],
    )

    st.markdown("---")

    son_n = st.slider(
        f"Son kaç {periyot_lbl} taransın?",
        min_value=1, max_value=10, value=3
    )

    trend_period = st.slider(
        "Trend periyodu (mum sayısı)",
        min_value=3, max_value=15, value=5
    )

    st.markdown("---")

    ema_on_chart = st.checkbox("Grafiklerde EMA göster", value=True)

    st.divider()

    run_btn = st.button("🚀 TARAMAYI BAŞLAT", use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <b>Stratejiler:</b><br>
    🔨 <b>Çekiç</b>: Dipte uzun alt gölge<br>
    🟢 <b>Yutan</b>: Boğa yutma formasyonu<br>
    ⭐ <b>Sabah Yıldızı</b>: 3 mumlu dip<br>
    📊 <b>EMA Dizilim</b>: 5>14>34>55 hizalanması
    </div>
    """, unsafe_allow_html=True)

# ── SONUÇLAR ────────────────────────────────────────────────────────────────
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

if run_btn:
    if not strategies:
        st.warning("⚠️ Lütfen en az bir strateji seçin.")
    else:
        st.session_state.selected_ticker = None
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def update_progress(pct, ticker):
            progress_bar.progress(pct)
            status_text.markdown(
                f'<div style="color:#5a6a90;font-size:0.8rem;">Taraniyor: <b style="color:#4a8cff">{ticker}</b></div>',
                unsafe_allow_html=True
            )

        with st.spinner(""):
            df_result = scan_all(
                strategies, interval, lookback_default,
                son_n, trend_period, update_progress
            )

        progress_bar.empty()
        status_text.empty()
        st.session_state.scan_results = df_result

# ── SONUÇ GÖSTER ─────────────────────────────────────────────────────────────
if st.session_state.scan_results is not None:
    df_r = st.session_state.scan_results

    if df_r.empty:
        st.info("🔍 Seçilen kriterlere uyan hisse bulunamadı. Parametreleri gevşetmeyi deneyin.")
    else:
        df_r = df_r.sort_values("Tarih", ascending=False).reset_index(drop=True)

        # Metrik kartlar
        toplam      = len(df_r)
        uniq_hisse  = df_r["Hisse"].nunique()
        bu_periyot  = len(df_r[df_r["Hafta Farkı"] == "Bu periyot"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{toplam}</div>
                <div class="lbl">Toplam Sinyal</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{uniq_hisse}</div>
                <div class="lbl">Farklı Hisse</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{bu_periyot}</div>
                <div class="lbl">Bu Periyot</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Filtre
        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            filtre_form = st.multiselect(
                "Formasyona göre filtrele",
                options=df_r["Formasyon"].unique().tolist(),
                default=df_r["Formasyon"].unique().tolist(),
            )
        with col_f2:
            sadece_son = st.checkbox(f"Sadece son {son_n} {periyot_lbl}", value=False)

        df_show = df_r[df_r["Formasyon"].isin(filtre_form)]
        if sadece_son:
            df_show = df_show[df_show["Hafta Farkı"] == "Bu periyot"]

        # Sonuç listesi
        cls_map = {"hammer":"hammer","engulf":"engulf","morning":"morning","ema":"ema"}
        tag_map  = {"hammer":"tag-hammer","engulf":"tag-engulf","morning":"tag-morning","ema":"tag-ema"}

        st.markdown(f"#### 📋 Sonuçlar ({len(df_show)} sinyal)")

        for _, row in df_show.iterrows():
            cls  = row["Sınıf"]
            tag  = tag_map.get(cls, "")
            card = cls_map.get(cls, "")
            ticker_disp = row["Hisse"]
            st.markdown(f"""
            <div class="result-card {card}">
                <span class="ticker-badge">{ticker_disp}</span>
                <span class="tag {tag}">{row['Formasyon']}</span>
                <span class="price-info">
                    📅 {row['Tarih']} &nbsp;|&nbsp;
                    ₺{row['Kapanış']} &nbsp;|&nbsp;
                    {row['Hafta Farkı']}
                </span>
                <span style="color:#3a4a6a;font-size:0.8rem;">{row['Ek Bilgi']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── GRAFİK BÖLÜMÜ ──────────────────────────────────────────────────
        st.markdown("#### 📊 Mum Grafiği")

        ticker_options = df_show["Ticker"].unique().tolist()
        if ticker_options:
            sel = st.selectbox(
                "Hisse seç",
                options=ticker_options,
                format_func=lambda x: x.replace(".IS",""),
            )

            df_chart = fetch_data(sel, interval, lookback_default)
            if df_chart is not None and not df_chart.empty:
                signal_dates = df_show[df_show["Ticker"] == sel]["Tarih"].tolist()
                fig = plot_candles(df_chart, sel.replace(".IS",""),
                                   ema_show=ema_on_chart,
                                   signal_dates=signal_dates)
                st.plotly_chart(fig, use_container_width=True)

                # Sinyal detay tablosu
                detail = df_show[df_show["Ticker"] == sel][
                    ["Tarih","Formasyon","Açılış","Yüksek","Düşük","Kapanış","Hafta Farkı"]
                ]
                st.dataframe(detail, use_container_width=True, hide_index=True)
            else:
                st.warning("Bu hisse için grafik verisi alınamadı.")

        # ── CSV İNDİR ──────────────────────────────────────────────────────
        st.divider()
        csv = df_r.drop(columns=["Sınıf"]).to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ Tüm Sonuçları CSV İndir",
            data=csv,
            file_name=f"bist_tarama_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    # Karşılama ekranı
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color:#3a4a6a;">
        <div style="font-size:4rem;">📊</div>
        <h3 style="color:#5a6a90; font-family:'Syne',sans-serif;">Taramaya Hazır</h3>
        <p style="max-width:500px; margin:0 auto; line-height:1.7;">
            Sol panelden <b style="color:#4a8cff">periyot</b> ve <b style="color:#4a8cff">stratejileri</b> seçip
            <b style="color:#f59e0b">TARAMAYI BAŞLAT</b> butonuna bas.<br><br>
            223 BIST hissesi otomatik taranacak.
        </p>
    </div>
    """, unsafe_allow_html=True)
