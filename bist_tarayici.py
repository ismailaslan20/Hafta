import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="BIST Tarayici", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp { background: #f0f4f8; color: #1a202c; font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
section[data-testid="stSidebar"] * { color: #2d3748 !important; }

.stButton > button {
    background: linear-gradient(135deg, #0099ff, #00d4aa);
    color: #ffffff; border: none; border-radius: 8px;
    font-weight: 700; width: 100%; padding: 0.7rem;
    font-size: 0.95rem; letter-spacing: 0.5px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0077cc, #00b389);
    transform: translateY(-1px);
}

.stDataFrame { background: #ffffff; border-radius: 10px; }
.stMetric { background: #ffffff; border-radius: 10px; padding: 1rem; border: 1px solid #e2e8f0; }

div[data-testid="stMetricValue"] { color: #0099ff !important; font-weight: 700; }
div[data-testid="stMetricLabel"] { color: #64748b !important; }

.stSelectbox > div, .stMultiSelect > div { background: #ffffff; border-radius: 8px; }
.stSlider > div { color: #2d3748; }
.stExpander { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }

h1, h2, h3 { color: #1a202c !important; }
.stMarkdown p { color: #4a5568; }

.ema-badge {
    display: inline-block;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

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
    "Gunluk":   ("1d",  90),
    "Haftalik": ("1wk", 365),
    "Aylik":    ("1mo", 1460),
}

# ═══════════════════════════════════════════════════════════
# MEVCUT FONKSİYONLAR (degistirilmedi)
# ═══════════════════════════════════════════════════════════

def calc_emas(closes, n=None):
    return {p: closes.ewm(span=p, adjust=False, min_periods=1).mean() for p in [5, 14, 20, 34, 50, 55, 200]}

def is_downtrend(closes, idx, period):
    if idx < period:
        return False
    return float(closes.iloc[idx - period]) > float(closes.iloc[idx - 1])

def check_bolge(closes, emas, rsi, idx, bolge):
    if bolge == "Filtre Yok":
        return True
    try:
        c    = float(closes.iloc[idx])
        e5   = float(emas[5].iloc[idx])
        e14  = float(emas[14].iloc[idx])
        e34  = float(emas[34].iloc[idx])
        e55  = float(emas[55].iloc[idx])
        rsi_val = float(rsi.iloc[idx])
        dip_kosul   = rsi_val < 45 and c < e55
        trend_kosul = e5 > e14 > e34
        if bolge == "Dip Bolgesi":
            return dip_kosul
        elif bolge == "Trend Devami":
            return trend_kosul
        elif bolge == "Ikisi de":
            return dip_kosul or trend_kosul
        return True
    except Exception:
        return True

def check_hammer(o, h, l, c):
    body = abs(c - o)
    rng = h - l
    if rng == 0 or body == 0:
        return False
    lower = min(o, c) - l
    upper = h - max(o, c)
    return lower >= 2.0 * body and upper <= 0.1 * rng

def check_engulfing(o1, c1, o2, c2):
    return c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1

def check_morning_star(r):
    if len(r) < 3:
        return False
    return (
        r[0][3] < r[0][0]
        and abs(r[1][3] - r[1][0]) < 0.3 * abs(r[0][3] - r[0][0])
        and r[2][3] > r[2][0]
        and r[2][3] > (r[0][0] + r[0][3]) / 2
    )

def check_three_inside_up(r):
    if len(r) < 3:
        return False
    o1, h1, l1, c1 = r[0]
    o2, h2, l2, c2 = r[1]
    o3, h3, l3, c3 = r[2]
    bearish_1 = c1 < o1
    bullish_2 = c2 > o2
    bullish_3 = c3 > o3
    inside = o2 >= c1 and c2 <= o1
    confirm = c3 > o1
    return bearish_1 and bullish_2 and bullish_3 and inside and confirm

def check_golden_cross(emas, closes, idx):
    if idx < 1:
        return False
    try:
        e20  = float(emas[20].iloc[idx])
        e50  = float(emas[50].iloc[idx])
        e200 = float(emas[200].iloc[idx])
        e50_prev  = float(emas[50].iloc[idx - 1])
        e200_prev = float(emas[200].iloc[idx - 1])
        c = float(closes.iloc[idx])
        cross = (e50_prev < e200_prev) and (e50 > e200)
        fiyat_ema20_ustu = c > e20
        return cross and fiyat_ema20_ustu
    except Exception:
        return False

def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=1).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def check_duseni_kiran(df, idx, lookback=20):
    if idx < lookback + 1:
        return False
    try:
        highs_window = df["High"].iloc[idx - lookback: idx]
        resistance = float(highs_window.max())
        current_close = float(df["Close"].iloc[idx])
        current_volume = float(df["Volume"].iloc[idx]) if "Volume" in df.columns else 0
        prev_volume = float(df["Volume"].iloc[idx - 1]) if "Volume" in df.columns else 0
        prev_close = float(df["Close"].iloc[idx - 1])
        kiriyor = current_close > resistance and prev_close <= resistance * 1.01
        hacim_artisi = current_volume > prev_volume * 1.2 if prev_volume > 0 else True
        return kiriyor and hacim_artisi
    except Exception:
        return False

def check_pullback(df, emas, rsi, idx, lookback=10, tolerance=2.0, rsi_min=40, rsi_max=60):
    if idx < max(lookback, 20) + 1:
        return False, 0, 0
    try:
        e5  = float(emas[5].iloc[idx])
        e14 = float(emas[14].iloc[idx])
        e20 = float(emas[20].iloc[idx])
        e34 = float(emas[34].iloc[idx])
        current_close = float(df["Close"].iloc[idx])
        current_open  = float(df["Open"].iloc[idx])
        current_rsi   = float(rsi.iloc[idx])
        uptrend = e5 > e14 > e34
        touched_ema = False
        min_distance = float('inf')
        for i in range(max(0, idx - lookback), idx):
            low_i   = float(df["Low"].iloc[i])
            close_i = float(df["Close"].iloc[i])
            ema20_i = float(emas[20].iloc[i])
            threshold = ema20_i * (1 + tolerance / 100)
            if low_i <= threshold or close_i <= threshold:
                touched_ema = True
                distance = min(abs(low_i - ema20_i), abs(close_i - ema20_i))
                min_distance = min(min_distance, distance)
        above_ema20 = current_close > e20
        distance_from_ema20 = ((current_close - e20) / e20) * 100
        rsi_healthy = rsi_min <= current_rsi <= rsi_max
        green_candle = current_close > current_open
        volume_increase = True
        if "Volume" in df.columns:
            current_volume = float(df["Volume"].iloc[idx])
            avg_volume = df["Volume"].iloc[idx - 20: idx].mean()
            volume_increase = current_volume > avg_volume
        pullback_confirmed = (uptrend and touched_ema and above_ema20 and
                             rsi_healthy and green_candle and volume_increase)
        return pullback_confirmed, distance_from_ema20, min_distance
    except Exception:
        return False, 0, 0

def check_bullish_divergence(closes, rsi, idx, lookback=20):
    if idx < lookback + 1:
        return False
    window_closes = closes.iloc[idx - lookback: idx + 1]
    window_rsi    = rsi.iloc[idx - lookback: idx + 1]
    if window_rsi.isna().any():
        return False
    current_low = float(closes.iloc[idx])
    current_rsi = float(rsi.iloc[idx])
    prev_window_closes = window_closes.iloc[:-1]
    prev_low_label = prev_window_closes.idxmin()
    prev_low_pos   = closes.index.get_loc(prev_low_label)
    if prev_low_pos == idx:
        return False
    prev_low = float(closes.iloc[prev_low_pos])
    prev_rsi = float(rsi.iloc[prev_low_pos])
    price_lower_low = current_low  < prev_low
    rsi_higher_low  = current_rsi  > prev_rsi
    rsi_oversold    = current_rsi < 50
    return price_lower_low and rsi_higher_low and rsi_oversold

# ═══════════════════════════════════════════════════════════
# YENİ: EMA14/21 x SMA50 TARAMA FONKSİYONU
# ═══════════════════════════════════════════════════════════

def tarama_ema_sma(df_dict, ema1_len=14, ema2_len=21, sma_len=50, uzaklasma_pct=3.0, bar_limit=3):
    yukari = []
    asagi  = []

    for sembol, df in df_dict.items():
        try:
            close = df["Close"].squeeze().astype(float)
            if len(close) < sma_len + 5:
                continue

            e1  = close.ewm(span=ema1_len, adjust=False).mean()
            e2  = close.ewm(span=ema2_len, adjust=False).mean()
            s50 = close.rolling(window=sma_len).mean()

            for i in range(1, bar_limit + 1):
                once   = float(e1.iloc[-i-1])
                sonra  = float(e1.iloc[-i])
                once2  = float(e2.iloc[-i-1])
                sonra2 = float(e2.iloc[-i])

                # Yukari kesisme
                if once < once2 and sonra > sonra2:
                    son_fiyat = float(close.iloc[-1])
                    son_sma50 = float(s50.iloc[-1])
                    son_e1    = float(e1.iloc[-1])
                    son_e2    = float(e2.iloc[-1])
                    uzaklik   = abs(son_e1 - son_sma50) / son_sma50 * 100
                    yukari.append({
                        "Sembol"     : sembol,
                        "Fiyat"      : round(son_fiyat, 2),
                        "EMA"+ str(ema1_len): round(son_e1, 2),
                        "EMA"+ str(ema2_len): round(son_e2, 2),
                        "SMA"+ str(sma_len) : round(son_sma50, 2),
                        "Uzaklik_%"  : round(uzaklik, 2),
                        "Kesisme_Bar": i,
                        "Durum"      : "YUKARI KESIS" if uzaklik <= uzaklasma_pct else "UZAK (%" + str(round(uzaklik,1)) + ")",
                    })
                    break

                # Asagi kesisme
                if once > once2 and sonra < sonra2:
                    son_fiyat = float(close.iloc[-1])
                    son_sma50 = float(s50.iloc[-1])
                    asagi.append({
                        "Sembol"     : sembol,
                        "Fiyat"      : round(son_fiyat, 2),
                        "EMA"+ str(ema1_len): round(float(e1.iloc[-1]), 2),
                        "EMA"+ str(ema2_len): round(float(e2.iloc[-1]), 2),
                        "SMA"+ str(sma_len) : round(son_sma50, 2),
                        "Kesisme_Bar": i,
                        "Durum"      : "ASAGI KESIS",
                    })
                    break

        except Exception:
            continue

    df_yukari = pd.DataFrame(yukari).sort_values("Uzaklik_%") if yukari else pd.DataFrame()
    df_asagi  = pd.DataFrame(asagi).sort_values("Kesisme_Bar") if asagi else pd.DataFrame()
    return df_yukari, df_asagi


def scan_ticker(ticker, interval, days_back, strategies, trend_period, son_n,
                bolge_filtre="Filtre Yok",
                pullback_lookback=10, pullback_tolerance=2.0,
                pullback_rsi_min=40, pullback_rsi_max=60,
                resistance_lookback=20, min_volume_increase=20):
    end = datetime.today()
    start = end - timedelta(days=days_back)
    try:
        df = yf.download(ticker, start=start, end=end,
                         interval=interval, progress=False, auto_adjust=False)
    except Exception:
        return []

    min_bars = {"1d": 60, "1wk": 30, "1mo": 12}
    min_required = min_bars.get(interval, 30)
    if df is None or df.empty or len(df) < min_required:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return []

    df["Open"]  = df["Open"].squeeze()
    df["High"]  = df["High"].squeeze()
    df["Low"]   = df["Low"].squeeze()
    df["Close"] = df["Close"].squeeze()
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].squeeze()

    closes = df["Close"]
    emas   = calc_emas(closes)
    rsi    = calc_rsi(closes)
    n      = len(df)
    results = []
    min_ema_warmup = min(55, n - son_n - 1)

    for i in range(max(min_ema_warmup, trend_period, n - son_n), n):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        signals = []

        if "Cekic" in strategies:
            if check_hammer(o, h, l, c) and is_downtrend(closes, i, trend_period) and check_bolge(closes, emas, rsi, i, bolge_filtre):
                signals.append("Cekic")

        if "Yutan" in strategies and i >= 1:
            o1 = float(df["Open"].iloc[i - 1])
            c1 = float(df["Close"].iloc[i - 1])
            if check_engulfing(o1, c1, o, c) and is_downtrend(closes, i, trend_period) and check_bolge(closes, emas, rsi, i, bolge_filtre):
                signals.append("Yutan")

        if "Sabah Yildizi" in strategies and i >= 2:
            rows = [(float(df["Open"].iloc[i-2+j]), float(df["High"].iloc[i-2+j]),
                     float(df["Low"].iloc[i-2+j]),  float(df["Close"].iloc[i-2+j])) for j in range(3)]
            if check_morning_star(rows) and check_bolge(closes, emas, rsi, i, bolge_filtre):
                signals.append("Sabah Yildizi")

        if "Three Inside Up" in strategies and i >= 2:
            rows3 = [(float(df["Open"].iloc[i-2+j]), float(df["High"].iloc[i-2+j]),
                      float(df["Low"].iloc[i-2+j]),  float(df["Close"].iloc[i-2+j])) for j in range(3)]
            if check_three_inside_up(rows3) and is_downtrend(closes, i, trend_period) and check_bolge(closes, emas, rsi, i, bolge_filtre):
                signals.append("Three Inside Up")

        if "RSI Uyumsuzlugu" in strategies:
            if check_bullish_divergence(closes, rsi, i):
                signals.append("Pozitif RSI Uyumsuzlugu")

        if "Golden Cross" in strategies:
            if check_golden_cross(emas, closes, i):
                signals.append("Golden Cross [EMA50>EMA200 & Fiyat>EMA20]")

        if "Duseni Kiran" in strategies:
            if check_duseni_kiran(df, i, lookback=resistance_lookback):
                resistance = float(df["High"].iloc[max(0, i - resistance_lookback): i].max())
                signals.append("Duseni Kiran [Direnc: " + str(round(resistance,2)) + "]")

        if "Pullback" in strategies:
            pb_result = check_pullback(df, emas, rsi, i,
                                       lookback=pullback_lookback,
                                       tolerance=pullback_tolerance,
                                       rsi_min=pullback_rsi_min,
                                       rsi_max=pullback_rsi_max)
            if pb_result[0]:
                ema20_val    = float(emas[20].iloc[i])
                distance_pct = pb_result[1]
                signals.append("Pullback [EMA20: " + str(round(ema20_val,2)) + ", Mesafe: " + str(round(distance_pct,1)) + "%]")

        if signals:
            e = {p: round(float(emas[p].iloc[i]), 2) for p in [5, 14, 34, 55]}
            hafta_farki = n - 1 - i
            row = {
                "Hisse":   ticker.replace(".IS", ""),
                "Tarih":   df.index[i].strftime("%Y-%m-%d"),
                "Sinyal":  " | ".join(signals),
                "Kapanis": round(c, 2),
                "RSI14":   round(float(rsi.iloc[i]), 1),
                "EMA5":    e[5],
                "EMA14":   e[14],
                "EMA34":   e[34],
                "EMA55":   e[55],
                "Zaman":   "Bu periyot" if hafta_farki == 0 else str(hafta_farki) + " periyot once",
                "_ticker": ticker,
                "_int":    interval,
                "_days":   days_back,
            }
            results.append(row)

    return results


def draw_chart(ticker, interval, days_back, signal_dates):
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)
    df = yf.download(ticker, start=start_dt, end=end_dt,
                     interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    closes = df["Close"].squeeze()
    df["Open"]  = df["Open"].squeeze()
    df["High"]  = df["High"].squeeze()
    df["Low"]   = df["Low"].squeeze()
    df["Close"] = df["Close"].squeeze()
    emas = calc_emas(closes)
    rsi  = calc_rsi(closes)

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=list(range(len(df))),
        open=df["Open"].tolist(), high=df["High"].tolist(),
        low=df["Low"].tolist(),   close=closes.tolist(),
        name=ticker,
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        increasing_fillcolor="#16a34a",  decreasing_fillcolor="#dc2626",
    ), row=1, col=1)

    colors = {5: "#f59e0b", 14: "#2563eb", 20: "#059669", 34: "#dc2626", 55: "#7c3aed"}
    for p, col in colors.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=emas[p].tolist(),
            name="EMA" + str(p), line=dict(color=col, width=1.5),
        ), row=1, col=1)

    for sd in signal_dates:
        try:
            ts  = pd.Timestamp(sd)
            pos = df.index.get_loc(ts)
            if pos >= 20:
                lookback_start = max(0, pos - 20)
                resistance = float(df["High"].iloc[lookback_start:pos].max())
                fig.add_shape(type="line",
                    x0=lookback_start, x1=pos, y0=resistance, y1=resistance,
                    line=dict(color="#dc2626", width=2, dash="dash"), row=1, col=1)
            low_val = float(df["Low"].iloc[pos]) * 0.98
            fig.add_annotation(x=pos, y=low_val, text="▲",
                font=dict(color="#f59e0b", size=18), showarrow=False, row=1, col=1)
        except Exception:
            pass

    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=rsi.tolist(),
        name="RSI 14", line=dict(color="#7c3aed", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#dc2626", width=1, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#16a34a", width=1, dash="dash"), row=2, col=1)

    step   = max(1, len(df) // 10)
    tvals  = list(range(0, len(df), step))
    tlabels = [df.index[i].strftime("%Y-%m-%d") for i in tvals]

    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font=dict(color="#1a202c", family="Inter"),
        xaxis=dict(gridcolor="#e2e8f0", tickmode="array", tickvals=tvals,
                   ticktext=tlabels, rangeslider=dict(visible=False)),
        xaxis2=dict(gridcolor="#e2e8f0", tickmode="array", tickvals=tvals, ticktext=tlabels),
        yaxis=dict(gridcolor="#e2e8f0"),
        yaxis2=dict(gridcolor="#e2e8f0"),
        legend=dict(bgcolor="#f8fafc", bordercolor="#e2e8f0", borderwidth=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=620,
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## BIST Tarayici")
    st.markdown("---")

    sekme = st.radio("Modul sec:", ["Formasyon Tarama", "EMA/SMA Kesisim"],
                     horizontal=False)

    st.markdown("---")

    if sekme == "Formasyon Tarama":
        periyot = st.selectbox("Periyot", list(PERIYOT_MAP.keys()), index=1)
        interval, days_back = PERIYOT_MAP[periyot]
        son_n      = st.slider("Son kac mum taransin?", 1, 10, 3)
        trend_per  = st.slider("Trend periyodu (mum)", 3, 15, 5)
        bolge_filtre = st.selectbox(
            "Bolge Filtresi:",
            ["Filtre Yok", "Dip Bolgesi", "Trend Devami", "Ikisi de"], index=0)
        st.markdown("---")
        with st.expander("Pullback Ayarlari", expanded=False):
            pullback_lookback  = st.slider("Arama Periyodu", 5, 20, 10)
            pullback_tolerance = st.slider("EMA20 Tolerans (%)", 0.5, 5.0, 2.0, 0.5)
            pullback_rsi_min   = st.slider("RSI Min", 30, 50, 40)
            pullback_rsi_max   = st.slider("RSI Max", 50, 70, 60)
        with st.expander("Duseni Kiran Ayarlari", expanded=False):
            resistance_lookback  = st.slider("Direnc Periyodu", 10, 30, 20)
            min_volume_increase  = st.slider("Min Hacim Artisi (%)", 10, 50, 20)
        st.markdown("---")
        strategies = st.multiselect(
            "Stratejiler",
            ["Cekic","Yutan","Sabah Yildizi","Three Inside Up",
             "RSI Uyumsuzlugu","Golden Cross","Duseni Kiran","Pullback"],
            default=["Duseni Kiran","Pullback"])
        hisse_sec = st.multiselect(
            "Hisse filtresi (bos=tumu):",
            [h.replace(".IS","") for h in HISSELER], default=[])
        st.markdown("---")
        btn_formasyon = st.button("TARAMAYI BASLAT")

    else:  # EMA/SMA Kesisim
        periyot_ema = st.selectbox("Periyot", list(PERIYOT_MAP.keys()), index=0)
        interval_ema, days_back_ema = PERIYOT_MAP[periyot_ema]
        st.markdown("---")
        col1, col2 = st.columns(2)
        ema1_len = col1.number_input("EMA 1", value=14, min_value=1, max_value=200)
        ema2_len = col2.number_input("EMA 2", value=21, min_value=1, max_value=200)
        sma_len  = st.number_input("SMA", value=50, min_value=1, max_value=500)
        uzaklasma = st.slider("Yukari kesis max uzaklik %", 0.5, 15.0, 3.0, 0.5,
                              help="EMA14 ile SMA50 arasindaki max uzaklik")
        bar_limit = st.slider("Son kac barda kesisti?", 1, 10, 3)
        st.markdown("---")
        btn_ema = st.button("TARAMAYI BASLAT")

    st.caption(str(len(HISSELER)) + " hisse listede")


# ═══════════════════════════════════════════════════════════
# ANA SAYFA
# ═══════════════════════════════════════════════════════════
st.markdown("# BIST Tarayici")

if sekme == "Formasyon Tarama":
    st.markdown("**Mod:** Formasyon Tarama")
    st.markdown("---")

    if "results" not in st.session_state:
        st.session_state.results = []
        st.session_state.done    = False

    if btn_formasyon:
        if not strategies:
            st.warning("En az bir strateji sec!")
        else:
            taranacak = [h for h in HISSELER if h.replace(".IS","") in hisse_sec] if hisse_sec else HISSELER
            st.session_state.results = []
            all_res = []
            prog  = st.progress(0)
            durum = st.empty()
            sayi  = st.empty()

            for i, ticker in enumerate(taranacak):
                durum.caption("Taraniyor: " + ticker)
                prog.progress((i + 1) / len(taranacak))
                rows = scan_ticker(ticker, interval, days_back, strategies, trend_per, son_n,
                                   bolge_filtre, pullback_lookback, pullback_tolerance,
                                   pullback_rsi_min, pullback_rsi_max,
                                   resistance_lookback, min_volume_increase)
                all_res.extend(rows)
                sayi.markdown("Bulunan sinyal: **" + str(len(all_res)) + "**")
                if (i + 1) % 25 == 0:
                    time.sleep(0.3)

            prog.empty(); durum.empty()
            st.session_state.results = all_res
            st.session_state.done    = True

    if st.session_state.get("done"):
        res = st.session_state.results
        if not res:
            st.info("Hic sinyal bulunamadi. Parametreleri gevset.")
        else:
            df_r = pd.DataFrame(res)
            df_r.sort_values(["Tarih","Hisse"], ascending=[False,True], inplace=True)
            df_r.reset_index(drop=True, inplace=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Sinyal",    len(df_r))
            c2.metric("Benzersiz Hisse",  df_r["Hisse"].nunique())
            c3.metric("Bu Periyot",       len(df_r[df_r["Zaman"] == "Bu periyot"]))
            c4.metric("Taranan",          len(HISSELER))
            st.markdown("---")

            f1, f2 = st.columns(2)
            fs = f1.multiselect("Sinyale filtrele:", df_r["Sinyal"].unique(), default=[])
            fz = f2.multiselect("Zamana filtrele:",  df_r["Zaman"].unique(),  default=[])

            df_show = df_r.copy()
            if fs: df_show = df_show[df_show["Sinyal"].isin(fs)]
            if fz: df_show = df_show[df_show["Zaman"].isin(fz)]

            show_cols = ["Hisse","Tarih","Sinyal","Kapanis","RSI14","EMA5","EMA14","EMA34","EMA55","Zaman"]
            st.markdown("### Sonuclar: " + str(len(df_show)) + " sinyal")
            st.dataframe(df_show[show_cols], use_container_width=True, hide_index=True)

            csv = df_show.drop(columns=["_ticker","_int","_days"], errors="ignore")
            st.download_button("CSV Indir",
                csv.to_csv(index=False, encoding="utf-8-sig"),
                "bist_" + datetime.now().strftime("%Y%m%d_%H%M") + ".csv", "text/csv")

            st.markdown("---")
            st.markdown("### Grafik")
            secim = st.selectbox("Hisse sec:", df_show["Hisse"].unique().tolist())
            if secim:
                dates = df_show[df_show["Hisse"] == secim]["Tarih"].tolist()
                draw_chart(secim + ".IS", interval, days_back, dates)
                row = df_show[df_show["Hisse"] == secim].iloc[0]
                e1c, e2c, e3c, e4c = st.columns(4)
                e1c.metric("EMA 5",  row["EMA5"])
                e2c.metric("EMA 14", row["EMA14"])
                e3c.metric("EMA 34", row["EMA34"])
                e4c.metric("EMA 55", row["EMA55"])
    else:
        st.info("Sol panelden strateji sec ve TARAMAYI BASLAT butonuna bas.")


else:  # EMA/SMA Kesisim modulu
    st.markdown("**Mod:** EMA/SMA Kesisim Tarama")
    st.markdown("---")

    if "ema_done" not in st.session_state:
        st.session_state.ema_done    = False
        st.session_state.ema_yukari  = pd.DataFrame()
        st.session_state.ema_asagi   = pd.DataFrame()

    if btn_ema:
        prog  = st.progress(0)
        durum = st.empty()
        df_dict = {}

        for i, sembol in enumerate(HISSELER):
            durum.text("Cekiliyor: " + sembol + " (" + str(i+1) + "/" + str(len(HISSELER)) + ")")
            try:
                period_map = {"1d": "6mo", "1wk": "2y", "1mo": "5y"}
                veri_period = period_map.get(interval_ema, "1y")
                raw = yf.download(sembol, period=veri_period, interval=interval_ema,
                                  progress=False, auto_adjust=True)
                if not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    df_dict[sembol.replace(".IS","")] = raw
            except Exception:
                pass
            prog.progress((i + 1) / len(HISSELER))

        prog.empty(); durum.empty()

        yukari, asagi = tarama_ema_sma(df_dict, ema1_len, ema2_len, sma_len, uzaklasma, bar_limit)
        st.session_state.ema_yukari = yukari
        st.session_state.ema_asagi  = asagi
        st.session_state.ema_done   = True

    if st.session_state.ema_done:
        yukari = st.session_state.ema_yukari
        asagi  = st.session_state.ema_asagi

        # Metrikler
        yukari_gecen = yukari[yukari["Durum"] == "YUKARI KESIS"] if not yukari.empty else pd.DataFrame()
        c1, c2, c3 = st.columns(3)
        c1.metric("Yukari Kesis (yakin)", len(yukari_gecen))
        c2.metric("Yukari Kesis (toplam)", len(yukari) if not yukari.empty else 0)
        c3.metric("Asagi Kesis", len(asagi) if not asagi.empty else 0)
        st.markdown("---")

        # YUKARI KESİŞİM
        st.markdown("### Yukari Kesisim — SMA" + str(st.session_state.get("sma_len", sma_len)) + " Yakin")

        if not yukari.empty and not yukari_gecen.empty:
            def style_yukari(row):
                return ["background-color:#dcfce7; color:#166534; font-weight:600"] * len(row)
            st.dataframe(
                yukari_gecen.style.apply(style_yukari, axis=1),
                use_container_width=True, hide_index=True)

            st.markdown("**TradingView:**")
            linkler = "  |  ".join(
                "[" + s + "](https://www.tradingview.com/chart/?symbol=BIST:" + s + ")"
                for s in yukari_gecen["Sembol"].tolist()
            )
            st.markdown(linkler)
        else:
            st.info("SMA uzaklik filtresini gecen yukari kesisim bulunamadi.")

        if not yukari.empty:
            uzak = yukari[yukari["Durum"] != "YUKARI KESIS"]
            if not uzak.empty:
                with st.expander("Uzakta kalanlar (" + str(len(uzak)) + " hisse)"):
                    st.dataframe(uzak, use_container_width=True, hide_index=True)

        st.markdown("---")

        # AŞAĞI KESİŞİM
        st.markdown("### Asagi Kesisim")
        if not asagi.empty:
            def style_asagi(row):
                return ["background-color:#fee2e2; color:#991b1b; font-weight:600"] * len(row)
            st.dataframe(
                asagi.style.apply(style_asagi, axis=1),
                use_container_width=True, hide_index=True)
        else:
            st.info("Asagi kesisim bulunamadi.")

    else:
        st.info("Sol panelden ayarlari sec ve TARAMAYI BASLAT butonuna bas.")
        st.markdown("""
        **Nasil Calisir:**
        - **Yukari Kesis:** EMA14 asagidan yukari gecti + SMA50 yakininda (uzaklik % ayarlanabilir)
        - **Asagi Kesis:** EMA14 yukaridan asagi gecti
        - Periyot, EMA/SMA uzunluklari ve uzaklik limiti sidebar'dan ayarlanabilir
        """)
