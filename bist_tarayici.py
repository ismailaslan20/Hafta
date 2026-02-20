"""
BIST Mum Formasyonu & EMA Tarayıcı — Streamlit Uygulaması
==========================================================
Kurulum:
    pip install streamlit yfinance pandas numpy plotly

Çalıştırma:
    streamlit run bist_scanner_app.py
"""
streamlit>=1.32.0
yfinance>=0.2.38
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ─── SAYFA AYARLARI ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIST Tarayıcı",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STİL ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    :root {
        --bg: #0a0e1a;
        --surface: #111827;
        --surface2: #1a2235;
        --accent: #00d4aa;
        --accent2: #ff6b6b;
        --accent3: #ffd166;
        --text: #e2e8f0;
        --muted: #64748b;
        --border: #1e2d40;
    }

    .stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }

    /* Başlık */
    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }

    /* Kart */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: var(--accent); }
    .card-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .card-value { font-family: 'Space Mono', monospace; font-size: 1.6rem; color: var(--accent); font-weight: 700; }

    /* Sinyal rozeti */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
    }
    .badge-hammer   { background: #00d4aa22; color: #00d4aa; border: 1px solid #00d4aa55; }
    .badge-bull-eng { background: #0099ff22; color: #0099ff; border: 1px solid #0099ff55; }
    .badge-morning  { background: #ffd16622; color: #ffd166; border: 1px solid #ffd16655; }
    .badge-ema      { background: #ff6b6b22; color: #ff6b6b; border: 1px solid #ff6b6b55; }

    /* Tablo */
    .result-table { width: 100%; border-collapse: collapse; }
    .result-table th {
        background: var(--surface2);
        color: var(--muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        padding: 0.6rem 1rem;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }
    .result-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--border);
        font-size: 0.88rem;
        color: var(--text);
    }
    .result-table tr:hover td { background: var(--surface2); }
    .ticker-cell { font-family: 'Space Mono', monospace; font-weight: 700; color: var(--accent); }
    .pos { color: #00d4aa; } .neg { color: #ff6b6b; }

    /* Buton */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #0099ff);
        color: #0a0e1a;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.7rem 2rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Divider */
    hr { border-color: var(--border); margin: 1.5rem 0; }

    /* Selectbox & Multiselect */
    .stSelectbox > div, .stMultiSelect > div {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* Progress */
    .stProgress > div > div { background: var(--accent); }

    /* Info box */
    .info-box {
        background: var(--surface);
        border-left: 3px solid var(--accent);
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: var(--muted);
    }
</style>
""", unsafe_allow_html=True)

# ─── HİSSE LİSTESİ ────────────────────────────────────────────────────────────
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
    "Günlük":  ("1d",  90),
    "Haftalık":("1wk", 365),
    "Aylık":   ("1mo", 1460),
}


# ─── PATTERN FONKSİYONLARI ────────────────────────────────────────────────────

def calc_emas(closes: pd.Series):
    return {
        5:  closes.ewm(span=5,  adjust=False).mean(),
        14: closes.ewm(span=14, adjust=False).mean(),
        34: closes.ewm(span=34, adjust=False).mean(),
        55: closes.ewm(span=55, adjust=False).mean(),
    }

def is_downtrend(closes, idx, period=5):
    if idx < period: return False
    seg = closes.iloc[idx - period:idx]
    return float(seg.iloc[-1]) < float(seg.iloc[0])

def is_uptrend(closes, idx, period=5):
    if idx < period: return False
    seg = closes.iloc[idx - period:idx]
    return float(seg.iloc[-1]) > float(seg.iloc[0])

def check_hammer(o, h, l, c, lower_ratio=2.0, upper_ratio=0.1):
    body = abs(c - o)
    rng  = h - l
    if rng == 0 or body == 0: return False
    lower = min(o, c) - l
    upper = h - max(o, c)
    return lower >= lower_ratio * body and upper <= upper_ratio * rng

def check_bullish_engulfing(o1, c1, o2, c2):
    """Önceki mum ayı, mevcut mum boğa ve öncekini yutuyor."""
    prev_bearish = c1 < o1
    curr_bullish = c2 > o2
    engulfs      = o2 < c1 and c2 > o1
    return prev_bearish and curr_bullish and engulfs

def check_morning_star(rows):
    """3 mumlu sabah yıldızı: ayı, doji/küçük, boğa."""
    if len(rows) < 3: return False
    o1,h1,l1,c1 = rows[0]
    o2,h2,l2,c2 = rows[1]
    o3,h3,l3,c3 = rows[2]
    bearish1 = c1 < o1
    small2   = abs(c2 - o2) < 0.3 * abs(c1 - o1)
    bullish3 = c3 > o3 and c3 > (o1 + c1) / 2
    return bearish1 and small2 and bullish3

def check_ema_dizilim(emas, idx):
    """
    EMA 5 > EMA 14 > EMA 34 > EMA 55  (boğa dizilimi)
    veya son mumda EMA 5 EMA 14'ü yukarı kesiyor.
    """
    try:
        e5  = float(emas[5].iloc[idx])
        e14 = float(emas[14].iloc[idx])
        e34 = float(emas[34].iloc[idx])
        e55 = float(emas[55].iloc[idx])
        bull_order = e5 > e14 > e34 > e55

        # EMA 5 yakın geçmişte EMA 14'ü yukarı kesti mi?
        cross = False
        if idx >= 2:
            e5_prev  = float(emas[5].iloc[idx-1])
            e14_prev = float(emas[14].iloc[idx-1])
            cross = e5_prev < e14_prev and e5 > e14

        return bull_order, cross
    except:
        return False, False


# ─── ANA TARAMA FONKSİYONU ────────────────────────────────────────────────────

def scan_ticker(ticker, interval, days_back, strategies, trend_period, son_n_mum):
    end   = datetime.today()
    start = end - timedelta(days=days_back)
    try:
        df = yf.download(ticker, start=start, end=end,
                         interval=interval, progress=False, auto_adjust=True)
    except:
        return []

    if df is None or df.empty or len(df) < max(trend_period, 55) + 3:
        return []

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open","High","Low","Close"}
    if not required.issubset(df.columns):
        return []

    closes = df["Close"].squeeze()
    emas   = calc_emas(closes)
    n      = len(df)
    results = []

    start_idx = max(55, trend_period, n - son_n_mum)

    for i in range(start_idx, n):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])

        signals = []

        if "Çekiç" in strategies:
            if check_hammer(o, h, l, c) and is_downtrend(closes, i, trend_period):
                signals.append("Çekiç")

        if "Yutan (Bullish Engulfing)" in strategies and i >= 1:
            o1 = float(df["Open"].iloc[i-1])
            c1 = float(df["Close"].iloc[i-1])
            if check_bullish_engulfing(o1, c1, o, c) and is_downtrend(closes, i, trend_period):
                signals.append("Yutan")

        if "Sabah Yıldızı" in strategies and i >= 2:
            rows = [
                (float(df["Open"].iloc[i-2]), float(df["High"].iloc[i-2]),
                 float(df["Low"].iloc[i-2]),  float(df["Close"].iloc[i-2])),
                (float(df["Open"].iloc[i-1]), float(df["High"].iloc[i-1]),
                 float(df["Low"].iloc[i-1]),  float(df["Close"].iloc[i-1])),
                (o, h, l, c),
            ]
            if check_morning_star(rows) and is_downtrend(closes, i-2, trend_period):
                signals.append("Sabah Yıldızı")

        if "EMA 5-14-34-55 Dizilimi" in strategies:
            bull_order, cross = check_ema_dizilim(emas, i)
            if bull_order or cross:
                label = "EMA Dizilim"
                if cross: label += " (Kesişim)"
                signals.append(label)

        if signals:
            hafta_farki = n - 1 - i
            e5  = round(float(emas[5].iloc[i]),  2)
            e14 = round(float(emas[14].iloc[i]), 2)
            e34 = round(float(emas[34].iloc[i]), 2)
            e55 = round(float(emas[55].iloc[i]), 2)

            results.append({
                "Hisse"    : ticker.replace(".IS",""),
                "Tarih"    : df.index[i].strftime("%Y-%m-%d"),
                "Sinyal"   : ", ".join(signals),
                "Kapanış"  : round(c, 2),
                "Açılış"   : round(o, 2),
                "Yüksek"   : round(h, 2),
                "Düşük"    : round(l, 2),
                "EMA5"     : e5,
                "EMA14"    : e14,
                "EMA34"    : e34,
                "EMA55"    : e55,
                "Zaman"    : "Bu periyot" if hafta_farki == 0 else f"{hafta_farki} periyot önce",
                "_ticker"  : ticker,
                "_idx"     : i,
                "_interval": interval,
                "_days"    : days_back,
            })

    return results


# ─── MUM GRAFİĞİ ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_chart_data(ticker, interval, days_back):
    end   = datetime.today()
    start = end - timedelta(days=days_back)
    df = yf.download(ticker, start=start, end=end,
                     interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def draw_chart(ticker, interval, days_back, signal_dates=None):
    df = get_chart_data(ticker, interval, days_back)
    if df is None:
        st.error("Grafik verisi alınamadı.")
        return

    closes = df["Close"].squeeze()
    emas   = calc_emas(closes)

    fig = go.Figure()

    # Mum grafiği
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=closes,
        name=ticker,
        increasing_line_color="#00d4aa",
        decreasing_line_color="#ff6b6b",
        increasing_fillcolor="#00d4aa",
        decreasing_fillcolor="#ff6b6b",
    ))

    # EMA çizgileri
    ema_colors = {5:"#ffd166", 14:"#0099ff", 34:"#ff6b6b", 55:"#cc88ff"}
    for period, color in ema_colors.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=emas[period],
            name=f"EMA{period}",
            line=dict(color=color, width=1.5),
            opacity=0.9,
        ))

    # Sinyal işaretleri
    if signal_dates:
        for sd in signal_dates:
            if sd in df.index or pd.Timestamp(sd) in df.index:
                try:
                    ts  = pd.Timestamp(sd)
                    row = df.loc[ts]
                    fig.add_annotation(
                        x=ts,
                        y=float(row["Low"].squeeze()) * 0.98,
                        text="▲",
                        font=dict(color="#ffd166", size=16),
                        showarrow=False,
                    )
                except:
                    pass

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0", family="DM Sans"),
        xaxis=dict(
            gridcolor="#1e2d40",
            showgrid=True,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(gridcolor="#1e2d40", showgrid=True),
        legend=dict(
            bgcolor="#111827",
            bordercolor="#1e2d40",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="main-title">📊 BIST</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Mum Formasyonu & EMA Tarayıcı</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Periyot
    st.markdown("**⏱ Periyot**")
    periyot = st.selectbox("", list(PERIYOT_MAP.keys()), index=1, label_visibility="collapsed")
    interval, days_back = PERIYOT_MAP[periyot]

    st.markdown("**🕯 Son Kaç Mum Taransın?**")
    son_n_mum = st.slider("", min_value=1, max_value=10, value=3, label_visibility="collapsed")

    st.markdown("**📉 Trend Periyodu (mum sayısı)**")
    trend_period = st.slider("", min_value=3, max_value=15, value=5, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**🎯 Stratejiler**")
    strategies = st.multiselect(
        "",
        ["Çekiç", "Yutan (Bullish Engulfing)", "Sabah Yıldızı", "EMA 5-14-34-55 Dizilimi"],
        default=["Çekiç", "EMA 5-14-34-55 Dizilimi"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**📋 Hisse Filtresi**")
    hisse_filtre = st.multiselect(
        "Belirli hisseler seç (boş = tümü):",
        [h.replace(".IS","") for h in HISSELER],
        default=[],
    )

    st.markdown("---")
    tarama_btn = st.button("🔍 TARAMAYI BAŞLAT")

    st.markdown("---")
    st.markdown(f"""
    <div class="info-box">
    📌 <b>{len(HISSELER)}</b> hisse listede<br>
    ⏰ Tarama süresi ~1-3 dk
    </div>
    """, unsafe_allow_html=True)


# ─── ANA İÇERİK ───────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">BIST Tarayıcı</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Periyot: {periyot} | Stratejiler: {", ".join(strategies) if strategies else "—"}</div>',
            unsafe_allow_html=True)

# Session state
if "results" not in st.session_state:
    st.session_state.results = []
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False

# ─── TARAMA ───────────────────────────────────────────────────────────────────
if tarama_btn:
    if not strategies:
        st.warning("⚠️ En az bir strateji seç!")
    else:
        taranacaklar = (
            [h for h in HISSELER if h.replace(".IS","") in hisse_filtre]
            if hisse_filtre else HISSELER
        )

        st.session_state.results = []
        all_results = []

        col_prog, col_stat = st.columns([3, 1])
        with col_prog:
            prog_bar  = st.progress(0)
            prog_text = st.empty()

        with col_stat:
            bulunan_text = st.empty()

        for idx, ticker in enumerate(taranacaklar):
            prog_text.markdown(f"<span style='color:#64748b;font-size:0.82rem'>Taraniyor: **{ticker}**</span>",
                               unsafe_allow_html=True)
            prog_bar.progress((idx + 1) / len(taranacaklar))

            rows = scan_ticker(ticker, interval, days_back, strategies, trend_period, son_n_mum)
            all_results.extend(rows)
            bulunan_text.markdown(
                f"<span style='color:#00d4aa;font-size:1rem;font-weight:700'>✅ {len(all_results)}</span>",
                unsafe_allow_html=True,
            )

            if (idx + 1) % 25 == 0:
                time.sleep(0.5)

        prog_text.empty()
        prog_bar.empty()

        st.session_state.results  = all_results
        st.session_state.scan_done = True

# ─── SONUÇLAR ─────────────────────────────────────────────────────────────────
if st.session_state.scan_done and st.session_state.results:
    results = st.session_state.results
    df_res  = pd.DataFrame(results)
    df_res.sort_values(["Tarih","Hisse"], ascending=[False, True], inplace=True)
    df_res.reset_index(drop=True, inplace=True)

    # ── Özet metrikler ──
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="card-label">Bulunan Sinyal</div>
            <div class="card-value">{len(df_res)}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="card-label">Benzersiz Hisse</div>
            <div class="card-value">{df_res['Hisse'].nunique()}</div></div>""", unsafe_allow_html=True)
    with c3:
        bu_periyot = df_res[df_res["Zaman"] == "Bu periyot"]
        st.markdown(f"""<div class="metric-card">
            <div class="card-label">Bu Periyot</div>
            <div class="card-value">{len(bu_periyot)}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="card-label">Taranan Hisse</div>
            <div class="card-value">{len(HISSELER)}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Filtre ──
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        filtre_sinyal = st.multiselect(
            "Sinyale göre filtrele:",
            df_res["Sinyal"].unique().tolist(),
            default=[],
        )
    with col_f2:
        filtre_zaman = st.multiselect(
            "Zamana göre filtrele:",
            df_res["Zaman"].unique().tolist(),
            default=[],
        )

    df_show = df_res.copy()
    if filtre_sinyal:
        df_show = df_show[df_show["Sinyal"].isin(filtre_sinyal)]
    if filtre_zaman:
        df_show = df_show[df_show["Zaman"].isin(filtre_zaman)]

    # ── Sonuç tablosu ──
    st.markdown(f"### 📋 Sonuçlar ({len(df_show)} sinyal)")

    badge_map = {
        "Çekiç":"badge-hammer", "Yutan":"badge-bull-eng",
        "Sabah Yıldızı":"badge-morning", "EMA":"badge-ema",
    }

    def sinyal_badge(s):
        for k, v in badge_map.items():
            if k in s:
                return f'<span class="badge {v}">{s}</span>'
        return f'<span class="badge badge-ema">{s}</span>'

    tablo_html = '<table class="result-table"><thead><tr>'
    for col in ["Hisse","Tarih","Sinyal","Kapanış","EMA5","EMA14","EMA34","EMA55","Zaman"]:
        tablo_html += f"<th>{col}</th>"
    tablo_html += "</tr></thead><tbody>"

    for _, row in df_show.iterrows():
        tablo_html += "<tr>"
        tablo_html += f'<td class="ticker-cell">{row["Hisse"]}</td>'
        tablo_html += f'<td>{row["Tarih"]}</td>'
        tablo_html += f'<td>{sinyal_badge(row["Sinyal"])}</td>'
        tablo_html += f'<td><b>{row["Kapanış"]}</b></td>'
        tablo_html += f'<td style="color:#ffd166">{row["EMA5"]}</td>'
        tablo_html += f'<td style="color:#0099ff">{row["EMA14"]}</td>'
        tablo_html += f'<td style="color:#ff6b6b">{row["EMA34"]}</td>'
        tablo_html += f'<td style="color:#cc88ff">{row["EMA55"]}</td>'
        tablo_html += f'<td><span style="color:#64748b;font-size:0.8rem">{row["Zaman"]}</span></td>'
        tablo_html += "</tr>"
    tablo_html += "</tbody></table>"

    st.markdown(tablo_html, unsafe_allow_html=True)

    # ── CSV İndir ──
    st.markdown("")
    csv_data = df_show.drop(columns=["_ticker","_idx","_interval","_days"], errors="ignore")
    st.download_button(
        label="💾 CSV Olarak İndir",
        data=csv_data.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"bist_tarama_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    # ── Grafik bölümü ──
    st.markdown("---")
    st.markdown("### 📈 Hisse Grafiği")

    hisse_secim = st.selectbox(
        "Grafik için hisse seç:",
        df_show["Hisse"].unique().tolist(),
    )

    if hisse_secim:
        ticker_full  = hisse_secim + ".IS"
        signal_dates = df_show[df_show["Hisse"] == hisse_secim]["Tarih"].tolist()
        draw_chart(ticker_full, interval, days_back, signal_dates)

        # EMA değerleri
        row = df_show[df_show["Hisse"] == hisse_secim].iloc[0]
        ec1, ec2, ec3, ec4 = st.columns(4)
        for col_w, (label, val, color) in zip(
            [ec1, ec2, ec3, ec4],
            [("EMA 5", row["EMA5"], "#ffd166"),
             ("EMA 14", row["EMA14"], "#0099ff"),
             ("EMA 34", row["EMA34"], "#ff6b6b"),
             ("EMA 55", row["EMA55"], "#cc88ff")]
        ):
            with col_w:
                col_w.markdown(f"""<div class="metric-card">
                    <div class="card-label">{label}</div>
                    <div class="card-value" style="color:{color}">{val}</div>
                </div>""", unsafe_allow_html=True)

elif st.session_state.scan_done and not st.session_state.results:
    st.info("ℹ️ Belirlenen kriterlere uyan sinyal bulunamadı. Parametreleri gevşetmeyi deneyin.")

else:
    # Hoş geldiniz ekranı
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size:4rem;">📊</div>
        <h2 style="color:#e2e8f0; font-family:'DM Sans',sans-serif; font-weight:300; margin-top:1rem;">
            Taramayı başlatmak için sol paneli kullan
        </h2>
        <p style="color:#64748b; max-width:500px; margin: 1rem auto;">
            Periyot ve strateji seç, ardından <b style="color:#00d4aa">TARAMAYI BAŞLAT</b> butonuna bas.
            223 BIST hissesi otomatik olarak taranır.
        </p>
    </div>

    <div style="display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin-top:2rem;">
        <div class="metric-card" style="min-width:180px; text-align:center;">
            <div class="card-label">Çekiç Mumu</div>
            <div style="color:#00d4aa; font-size:1.8rem;">🔨</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:0.3rem;">Alt gölge ≥ 2x gövde</div>
        </div>
        <div class="metric-card" style="min-width:180px; text-align:center;">
            <div class="card-label">Yutan Formasyon</div>
            <div style="color:#0099ff; font-size:1.8rem;">📈</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:0.3rem;">Boğa önceki ayıyı yutar</div>
        </div>
        <div class="metric-card" style="min-width:180px; text-align:center;">
            <div class="card-label">Sabah Yıldızı</div>
            <div style="color:#ffd166; font-size:1.8rem;">⭐</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:0.3rem;">3 mumlu dönüş sinyali</div>
        </div>
        <div class="metric-card" style="min-width:180px; text-align:center;">
            <div class="card-label">EMA 5-14-34-55</div>
            <div style="color:#cc88ff; font-size:1.8rem;">〰️</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:0.3rem;">Fibonacci EMA dizilimi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
