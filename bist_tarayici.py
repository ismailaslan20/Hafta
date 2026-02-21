import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="BIST Tarayıcı", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp { background: #0a0e1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #111827; }
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0099ff);
    color: #0a0e1a; border: none; border-radius: 8px;
    font-weight: 700; width: 100%; padding: 0.7rem;
}
.metric-box {
    background: #111827; border: 1px solid #1e2d40;
    border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 0.8rem;
}
.metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #00d4aa; }
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
    "Günlük":   ("1d",  90),
    "Haftalık": ("1wk", 365),
    "Aylık":    ("1mo", 1460),
}

def calc_emas(closes, n=None):
    return {p: closes.ewm(span=p, adjust=False, min_periods=1).mean() for p in [5, 14, 34, 55]}

def is_downtrend(closes, idx, period):
    if idx < period:
        return False
    return float(closes.iloc[idx - period]) > float(closes.iloc[idx - 1])

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

def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=1).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ─────────────────────────────────────────────
# YENİ: MACD(100) hesaplama
# fast=12, slow=100, signal=9
# ─────────────────────────────────────────────
def calc_macd(closes, fast=12, slow=100, signal=9):
    ema_fast   = closes.ewm(span=fast,   adjust=False, min_periods=1).mean()
    ema_slow   = closes.ewm(span=slow,   adjust=False, min_periods=1).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

# ─────────────────────────────────────────────
# YENİ: ADX hesaplama (Wilder yöntemi)
# ─────────────────────────────────────────────
def calc_adx(df, period=14):
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=close.index).ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=close.index).ewm(alpha=1/period, min_periods=1, adjust=False).mean()
    tr_s       = tr.ewm(alpha=1/period, min_periods=1, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / tr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / tr_s.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=1, adjust=False).mean()

    return adx, plus_di, minus_di

# ─────────────────────────────────────────────
# YENİ: MACD+ADX sinyal kontrolü
# Koşulların TAMAMI aynı anda gerçekleşmeli:
# 1) MACD histogramı sıfırı yukarı kesti
# 2) MACD çizgisi sinyal hattını yukarı kesti
# 3) ADX 25 üzerinde (güçlü trend)
# 4) Hem MACD hem ADX yukarı dönüyor (momentum artıyor)
# ─────────────────────────────────────────────
def check_macd_adx(macd_line, signal_line, histogram, adx, idx):
    if idx < 2:
        return False, []

    # Mevcut ve önceki değerler
    hist_curr  = float(histogram.iloc[idx])
    hist_prev  = float(histogram.iloc[idx - 1])
    macd_curr  = float(macd_line.iloc[idx])
    macd_prev  = float(macd_line.iloc[idx - 1])
    sig_curr   = float(signal_line.iloc[idx])
    sig_prev   = float(signal_line.iloc[idx - 1])
    adx_curr   = float(adx.iloc[idx])
    adx_prev   = float(adx.iloc[idx - 1])

    # 4 koşul
    hist_cross_zero   = hist_prev <= 0 and hist_curr > 0          # Histogram sıfırı yukarı kesti
    macd_cross_signal = macd_prev <= sig_prev and macd_curr > sig_curr  # MACD sinyal hattını kesti
    adx_above_25      = adx_curr > 25                             # ADX güçlü trend bölgesinde
    both_rising       = macd_curr > macd_prev and adx_curr > adx_prev   # Her ikisi yukarı dönüyor

    details = []
    if hist_cross_zero:
        details.append("Hist>0")
    if macd_cross_signal:
        details.append("MACD×Sig")
    if adx_above_25:
        details.append(f"ADX:{adx_curr:.1f}")
    if both_rising:
        details.append("↑Momentum")

    all_conditions = hist_cross_zero and macd_cross_signal and adx_above_25 and both_rising
    return all_conditions, details


def check_bullish_divergence(closes, rsi, idx, lookback=20):
    if idx < lookback + 1:
        return False
    window_closes = closes.iloc[idx - lookback: idx + 1]
    window_rsi    = rsi.iloc[idx - lookback: idx + 1]
    if window_rsi.isna().any():
        return False
    current_low  = float(closes.iloc[idx])
    current_rsi  = float(rsi.iloc[idx])
    prev_window_closes = window_closes.iloc[:-1]
    prev_low_label = prev_window_closes.idxmin()
    prev_low_pos   = closes.index.get_loc(prev_low_label)
    if prev_low_pos == idx:
        return False
    prev_low = float(closes.iloc[prev_low_pos])
    prev_rsi = float(rsi.iloc[prev_low_pos])
    price_lower_low  = current_low  < prev_low
    rsi_higher_low   = current_rsi  > prev_rsi
    rsi_oversold     = current_rsi < 50
    return price_lower_low and rsi_higher_low and rsi_oversold

def check_ema(emas, idx):
    try:
        e5      = float(emas[5].iloc[idx])
        e14     = float(emas[14].iloc[idx])
        e34     = float(emas[34].iloc[idx])
        e55     = float(emas[55].iloc[idx])
        bull    = e5 > e14 > e34 > e55
        cross     = False
        pre_cross = False
        if idx >= 3:
            e5_prev1  = float(emas[5].iloc[idx - 1])
            e5_prev2  = float(emas[5].iloc[idx - 2])
            e14_prev1 = float(emas[14].iloc[idx - 1])
            cross = e5_prev1 < e14_prev1 and e5 > e14
            ema5_rising = e5 > e5_prev1 > e5_prev2
            still_below = e5 < e14
            gap_pct     = (e14 - e5) / e14 * 100
            pre_cross   = ema5_rising and still_below and gap_pct < 1.5
        return bull, cross, pre_cross
    except Exception:
        return False, False, False


def scan_ticker(ticker, interval, days_back, strategies, trend_period, son_n):
    end = datetime.today()
    start = end - timedelta(days=days_back)
    try:
        df = yf.download(
            ticker, start=start, end=end,
            interval=interval, progress=False, auto_adjust=True
        )
    except Exception:
        return []

    min_bars = {"1d": 60, "1wk": 30, "1mo": 12}
    min_required = min_bars.get(interval, 30)

    # MACD(100) için en az 120 bar gerekli
    if "MACD+ADX" in strategies:
        min_required = max(min_required, 120)

    if df is None or df.empty or len(df) < min_required:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return []

    closes = df["Close"].squeeze()
    emas   = calc_emas(closes)
    rsi    = calc_rsi(closes)
    n      = len(df)
    results = []

    # MACD ve ADX hesapla (sadece strateji seçildiyse)
    macd_line = signal_line = histogram = adx_series = plus_di = minus_di = None
    if "MACD+ADX" in strategies:
        macd_line, signal_line, histogram = calc_macd(closes, fast=12, slow=100, signal=9)
        adx_series, plus_di, minus_di     = calc_adx(df, period=14)

    min_ema_warmup = min(55, n - son_n - 1)

    for i in range(max(min_ema_warmup, trend_period, n - son_n), n):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        signals = []

        if "Cekic" in strategies:
            if check_hammer(o, h, l, c) and is_downtrend(closes, i, trend_period):
                signals.append("Çekiç")

        if "Yutan" in strategies and i >= 1:
            o1 = float(df["Open"].iloc[i - 1])
            c1 = float(df["Close"].iloc[i - 1])
            if check_engulfing(o1, c1, o, c) and is_downtrend(closes, i, trend_period):
                signals.append("Yutan")

        if "Sabah Yildizi" in strategies and i >= 2:
            rows = [
                (
                    float(df["Open"].iloc[i - 2 + j]),
                    float(df["High"].iloc[i - 2 + j]),
                    float(df["Low"].iloc[i - 2 + j]),
                    float(df["Close"].iloc[i - 2 + j]),
                )
                for j in range(3)
            ]
            if check_morning_star(rows):
                signals.append("Sabah Yıldızı")

        if "Three Inside Up" in strategies and i >= 2:
            rows3 = [
                (
                    float(df["Open"].iloc[i - 2 + j]),
                    float(df["High"].iloc[i - 2 + j]),
                    float(df["Low"].iloc[i - 2 + j]),
                    float(df["Close"].iloc[i - 2 + j]),
                )
                for j in range(3)
            ]
            if check_three_inside_up(rows3) and is_downtrend(closes, i, trend_period):
                signals.append("Three Inside Up")

        if "RSI Uyumsuzlugu" in strategies:
            if check_bullish_divergence(closes, rsi, i):
                signals.append("Pozitif RSI Uyumsuzluğu")

        if "EMA Dizilimi" in strategies:
            bull, cross, pre_cross = check_ema(emas, i)
            if bull:
                signals.append("EMA Dizilim")
            if cross:
                signals.append("EMA Kesisim")
            if pre_cross:
                signals.append("EMA Yaklasim")

        # ── YENİ: MACD(100) + ADX Stratejisi ──────────────────────────────
        if "MACD+ADX" in strategies and macd_line is not None:
            triggered, details = check_macd_adx(macd_line, signal_line, histogram, adx_series, i)
            if triggered:
                signals.append("MACD+ADX [" + " | ".join(details) + "]")
        # ───────────────────────────────────────────────────────────────────

        if signals:
            e = {p: round(float(emas[p].iloc[i]), 2) for p in [5, 14, 34, 55]}
            hafta_farki = n - 1 - i

            # MACD+ADX ek bilgileri
            extra = {}
            if macd_line is not None:
                extra["MACD"]  = round(float(macd_line.iloc[i]), 4)
                extra["ADX"]   = round(float(adx_series.iloc[i]), 1)
                extra["+DI"]   = round(float(plus_di.iloc[i]), 1)
                extra["-DI"]   = round(float(minus_di.iloc[i]), 1)

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
            row.update(extra)
            results.append(row)

    return results


def draw_chart(ticker, interval, days_back, signal_dates, show_macd=False):
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)
    df = yf.download(
        ticker, start=start_dt, end=end_dt,
        interval=interval, progress=False, auto_adjust=True
    )
    if df is None or df.empty:
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    closes = df["Close"].squeeze()
    emas   = calc_emas(closes)
    rsi    = calc_rsi(closes)

    from plotly.subplots import make_subplots

    if show_macd:
        # 3 panel: Mum | MACD | RSI
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.03,
            subplot_titles=("", "MACD (12/100/9)", "RSI 14"),
        )
        macd_line, signal_line, histogram = calc_macd(closes, fast=12, slow=100, signal=9)
        adx_series, plus_di, minus_di     = calc_adx(df, period=14)
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.68, 0.32],
            vertical_spacing=0.03,
        )

    # Mum grafiği
    fig.add_trace(
        go.Candlestick(
            x=list(range(len(df))),
            open=df["Open"].squeeze().tolist(),
            high=df["High"].squeeze().tolist(),
            low=df["Low"].squeeze().tolist(),
            close=closes.tolist(),
            name=ticker,
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff6b6b",
            increasing_fillcolor="#00d4aa",
            decreasing_fillcolor="#ff6b6b",
        ),
        row=1, col=1,
    )

    colors = {5: "#ffd166", 14: "#0099ff", 34: "#ff6b6b", 55: "#cc88ff"}
    for p, col in colors.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=emas[p].tolist(),
                name="EMA" + str(p),
                line=dict(color=col, width=1.5),
            ),
            row=1, col=1,
        )

    for sd in signal_dates:
        try:
            ts  = pd.Timestamp(sd)
            pos = df.index.get_loc(ts)
            low_val = float(df["Low"].iloc[pos]) * 0.98
            fig.add_annotation(
                x=pos, y=low_val,
                text="▲",
                font=dict(color="#ffd166", size=18),
                showarrow=False,
                row=1, col=1,
            )
        except Exception:
            pass

    if show_macd:
        # MACD paneli
        colors_hist = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in histogram.tolist()]
        fig.add_trace(
            go.Bar(
                x=list(range(len(df))),
                y=histogram.tolist(),
                name="Histogram",
                marker_color=colors_hist,
                opacity=0.7,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=macd_line.tolist(),
                name="MACD(100)",
                line=dict(color="#0099ff", width=1.5),
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=signal_line.tolist(),
                name="Sinyal",
                line=dict(color="#ffd166", width=1.5),
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line=dict(color="#64748b", width=1, dash="dash"), row=2, col=1)

        # RSI paneli (3. satır)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi.tolist(),
                name="RSI 14",
                line=dict(color="#a78bfa", width=1.5),
            ),
            row=3, col=1,
        )
        fig.add_hline(y=70, line=dict(color="#ff6b6b", width=1, dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#00d4aa", width=1, dash="dash"), row=3, col=1)
        rsi_row = 3
    else:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi.tolist(),
                name="RSI 14",
                line=dict(color="#a78bfa", width=1.5),
            ),
            row=2, col=1,
        )
        fig.add_hline(y=70, line=dict(color="#ff6b6b", width=1, dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#00d4aa", width=1, dash="dash"), row=2, col=1)
        rsi_row = 2

    step   = max(1, len(df) // 10)
    tvals  = list(range(0, len(df), step))
    tlabels = [df.index[i].strftime("%Y-%m-%d") for i in tvals]

    height = 750 if show_macd else 620
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e2d40", tickmode="array", tickvals=tvals, ticktext=tlabels, rangeslider=dict(visible=False)),
        xaxis2=dict(gridcolor="#1e2d40", tickmode="array", tickvals=tvals, ticktext=tlabels),
        yaxis=dict(gridcolor="#1e2d40"),
        yaxis2=dict(gridcolor="#1e2d40"),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d40", borderwidth=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
    )
    if show_macd:
        fig.update_layout(xaxis3=dict(gridcolor="#1e2d40", tickmode="array", tickvals=tvals, ticktext=tlabels))
        fig.update_yaxes(range=[0, 100], row=3, col=1)
    else:
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## BIST Tarayici")
    st.markdown("---")
    periyot = st.selectbox("Periyot", list(PERIYOT_MAP.keys()), index=1)
    interval, days_back = PERIYOT_MAP[periyot]

    # MACD(100) için günlük 1 yıl+ gerekli → otomatik uzat
    if interval == "1d":
        days_back = max(days_back, 400)

    son_n = st.slider("Son kac mum taransin?", 1, 10, 3)
    trend_per = st.slider("Trend periyodu (mum)", 3, 15, 5)
    st.markdown("---")
    strategies = st.multiselect(
        "Stratejiler",
        ["Cekic", "Yutan", "Sabah Yildizi", "Three Inside Up",
         "RSI Uyumsuzlugu", "EMA Dizilimi", "MACD+ADX"],
        default=["MACD+ADX"],
    )
    hisse_sec = st.multiselect(
        "Hisse filtresi (bos=tumu):",
        [h.replace(".IS", "") for h in HISSELER],
        default=[],
    )
    st.markdown("---")
    btn = st.button("TARAMAYI BASLAT")
    st.caption(str(len(HISSELER)) + " hisse listede")

    # MACD+ADX seçildiyse bilgi notu
    if "MACD+ADX" in strategies:
        st.info(
            "**MACD(12/100/9) + ADX(14)**\n\n"
            "Tüm koşullar aynı anda:\n"
            "- Histogram 0'ı yukarı kesti\n"
            "- MACD sinyal hattını yukarı kesti\n"
            "- ADX > 25 (güçlü trend)\n"
            "- MACD & ADX yukarı dönüyor"
        )

st.markdown("# BIST Formasyon Tarayici")
st.markdown("Periyot: " + periyot + " | Stratejiler: " + (", ".join(strategies) if strategies else "-"))
st.markdown("---")

if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.done = False

if btn:
    if not strategies:
        st.warning("En az bir strateji sec!")
    else:
        if hisse_sec:
            taranacak = [h for h in HISSELER if h.replace(".IS", "") in hisse_sec]
        else:
            taranacak = HISSELER

        st.session_state.results = []
        all_res = []
        prog = st.progress(0)
        durum = st.empty()
        sayi = st.empty()

        for i, ticker in enumerate(taranacak):
            durum.caption("Taraniyor: " + ticker)
            prog.progress((i + 1) / len(taranacak))
            rows = scan_ticker(ticker, interval, days_back, strategies, trend_per, son_n)
            all_res.extend(rows)
            sayi.markdown("Bulunan sinyal: " + str(len(all_res)))
            if (i + 1) % 25 == 0:
                time.sleep(0.3)

        prog.empty()
        durum.empty()
        st.session_state.results = all_res
        st.session_state.done = True

if st.session_state.done:
    res = st.session_state.results
    if not res:
        st.info("Hic sinyal bulunamadi. Parametreleri gevset.")
    else:
        df_r = pd.DataFrame(res)
        df_r.sort_values(["Tarih", "Hisse"], ascending=[False, True], inplace=True)
        df_r.reset_index(drop=True, inplace=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Toplam Sinyal", len(df_r))
        with c2:
            st.metric("Benzersiz Hisse", df_r["Hisse"].nunique())
        with c3:
            st.metric("Bu Periyot", len(df_r[df_r["Zaman"] == "Bu periyot"]))
        with c4:
            st.metric("Taranan", len(HISSELER))

        st.markdown("---")

        f1, f2 = st.columns(2)
        with f1:
            fs = st.multiselect("Sinyale filtrele:", df_r["Sinyal"].unique(), default=[])
        with f2:
            fz = st.multiselect("Zamana filtrele:", df_r["Zaman"].unique(), default=[])

        df_show = df_r.copy()
        if fs:
            df_show = df_show[df_show["Sinyal"].isin(fs)]
        if fz:
            df_show = df_show[df_show["Zaman"].isin(fz)]

        # Gösterilecek kolonlar (MACD+ADX varsa ekstra kolonlar da göster)
        base_cols = ["Hisse", "Tarih", "Sinyal", "Kapanis", "RSI14", "EMA5", "EMA14", "EMA34", "EMA55", "Zaman"]
        extra_cols = [c for c in ["MACD", "ADX", "+DI", "-DI"] if c in df_show.columns]
        show_cols = base_cols + extra_cols

        st.markdown("### Sonuclar: " + str(len(df_show)) + " sinyal")
        st.dataframe(
            df_show[show_cols],
            use_container_width=True,
            hide_index=True,
        )

        csv = df_show.drop(columns=["_ticker", "_int", "_days"], errors="ignore")
        st.download_button(
            "CSV Indir",
            csv.to_csv(index=False, encoding="utf-8-sig"),
            "bist_" + datetime.now().strftime("%Y%m%d_%H%M") + ".csv",
            "text/csv",
        )

        st.markdown("---")
        st.markdown("### Grafik")
        secim = st.selectbox("Hisse sec:", df_show["Hisse"].unique().tolist())
        if secim:
            ticker_f = secim + ".IS"
            dates = df_show[df_show["Hisse"] == secim]["Tarih"].tolist()

            # MACD+ADX seçiliyse MACD panelini göster
            show_macd_panel = "MACD+ADX" in strategies
            draw_chart(ticker_f, interval, days_back, dates, show_macd=show_macd_panel)

            row = df_show[df_show["Hisse"] == secim].iloc[0]
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("EMA 5", row["EMA5"])
            with e2:
                st.metric("EMA 14", row["EMA14"])
            with e3:
                st.metric("EMA 34", row["EMA34"])
            with e4:
                st.metric("EMA 55", row["EMA55"])

            # MACD+ADX varsa ek metrikler
            if "ADX" in row and pd.notna(row.get("ADX")):
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("MACD(100)", row.get("MACD", "-"))
                with m2:
                    st.metric("ADX", row.get("ADX", "-"))
                with m3:
                    st.metric("+DI", row.get("+DI", "-"))
                with m4:
                    st.metric("-DI", row.get("-DI", "-"))
else:
    st.info("Sol panelden periyot ve strateji sec, ardından TARAMAYI BASLAT butonuna bas.")
