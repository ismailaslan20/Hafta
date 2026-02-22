import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# ─── GÖSTERGE FONKSİYONLARI ──────────────────────────────────────────────────

def calc_emas(closes):
    return {p: closes.ewm(span=p, adjust=False, min_periods=1).mean() for p in [5, 14, 34, 55]}

def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(com=period - 1, min_periods=1).mean()
    al    = loss.ewm(com=period - 1, min_periods=1).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def is_downtrend(closes, idx, period):
    if idx < period:
        return False
    return float(closes.iloc[idx - period]) > float(closes.iloc[idx - 1])

# ─── MUM FORMASYON FONKSİYONLARI ─────────────────────────────────────────────

def check_hammer(o, h, l, c):
    body  = abs(c - o)
    rng   = h - l
    if rng == 0 or body == 0:
        return False
    lower = min(o, c) - l
    upper = h - max(o, c)
    return lower >= 2.0 * body and upper <= 0.3 * rng

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
    o1, _, _, c1 = r[0]
    o2, _, _, c2 = r[1]
    o3, _, _, c3 = r[2]
    bearish_1 = c1 < o1
    bullish_2 = c2 > o2
    bullish_3 = c3 > o3
    inside    = o2 >= c1 and c2 <= o1
    confirm   = c3 > o1
    return bearish_1 and bullish_2 and bullish_3 and inside and confirm

def check_inside_bar_breakout(df, idx):
    if idx < 2:
        return False
    try:
        h_mother = float(df["High"].iloc[idx - 2])
        l_mother = float(df["Low"].iloc[idx - 2])
        h_inside = float(df["High"].iloc[idx - 1])
        l_inside = float(df["Low"].iloc[idx - 1])
        c_curr   = float(df["Close"].iloc[idx])
        o_curr   = float(df["Open"].iloc[idx])
        is_inside = h_inside < h_mother and l_inside > l_mother
        breakout  = c_curr > h_mother and c_curr > o_curr
        return is_inside and breakout
    except Exception:
        return False

# ─── RSI FONKSİYONLARI ───────────────────────────────────────────────────────

def check_rsi_bounce(closes, rsi, idx):
    if idx < 2:
        return False
    try:
        rsi_prev = float(rsi.iloc[idx - 1])
        rsi_curr = float(rsi.iloc[idx])
        c_prev   = float(closes.iloc[idx - 1])
        c_curr   = float(closes.iloc[idx])
        return rsi_prev < 30 and rsi_curr > 30 and c_curr > c_prev
    except Exception:
        return False

def find_pivot_lows(series, left=3):
    pivots = []
    vals   = series.values
    n      = len(vals)
    for i in range(left, n):
        if all(vals[i] <= vals[i - j] for j in range(1, left + 1)):
            pivots.append(i)
    return pivots

def check_bullish_divergence(closes, rsi, idx, lookback=60, left=3):
    # Her zaman (bool, info_or_None) döndürür
    if idx < left + 4:
        return False, None
    try:
        win_start      = max(0, idx - lookback)
        window_closes  = closes.iloc[win_start: idx + 1]
        window_rsi     = rsi.iloc[win_start: idx + 1]

        if window_rsi.isna().sum() > len(window_rsi) * 0.3:
            return False, None

        local_pivots = find_pivot_lows(window_closes, left=left)
        if len(local_pivots) < 2:
            return False, None

        p2_local = local_pivots[-1]
        p1_local = local_pivots[-2]

        # En yeni pivot son 5 mum içinde olsun
        if len(window_closes) - 1 - p2_local > 5:
            return False, None
        # Pivotlar arası en az 5 mum
        if p2_local - p1_local < 5:
            return False, None

        p2 = win_start + p2_local
        p1 = win_start + p1_local

        if p2 >= len(closes) or p1 >= len(closes):
            return False, None

        price_p1 = float(closes.iloc[p1])
        price_p2 = float(closes.iloc[p2])
        rsi_p1   = float(rsi.iloc[p1])
        rsi_p2   = float(rsi.iloc[p2])

        if np.isnan(rsi_p1) or np.isnan(rsi_p2):
            return False, None

        ok = (
            price_p2 < price_p1
            and rsi_p2 > rsi_p1
            and rsi_p2 < 60
            and abs(price_p2 - price_p1) / price_p1 > 0.005
            and abs(rsi_p2 - rsi_p1) > 1.0
        )
        if ok:
            return True, (p1, p2, price_p1, price_p2, rsi_p1, rsi_p2)
        return False, None
    except Exception:
        return False, None

# ─── BOLLINGER + EMA FORMASYON ───────────────────────────────────────────────

def calc_bb_width(closes, idx, window=20, num_std=2.0):
    """Bollinger Bant Genişliği hesaplar (upper-lower / middle)"""
    if idx < window:
        return None
    pw  = closes.iloc[idx - window: idx + 1]
    ma  = float(pw.mean())
    std = float(pw.std())
    if ma == 0:
        return None
    return (2 * num_std * std) / ma  # normalize genişlik

def check_bollinger_squeeze_rsi(closes, rsi, idx, window=20, num_std=2.0, squeeze_lookback=10):
    """
    Bollinger Squeeze + RSI 50 Kesişimi:
    1. BB bantları daralmış (son squeeze_lookback mumun en dar noktasına yakın)
    2. RSI önceki mumda 50 altında, bu mumda 50 üstüne çıktı (50 kesişimi)
    3. Fiyat orta bandın (MA) üzerinde kapandı
    """
    if idx < window + squeeze_lookback + 2:
        return False
    try:
        curr_rsi = float(rsi.iloc[idx])
        prev_rsi = float(rsi.iloc[idx - 1])

        # RSI 50 kesişimi: önceki mum altında, bu mum üstünde
        rsi_cross_50 = prev_rsi < 50 and curr_rsi >= 50

        if not rsi_cross_50:
            return False

        # Mevcut BB genişliği
        curr_width = calc_bb_width(closes, idx, window, num_std)
        if curr_width is None:
            return False

        # Son squeeze_lookback mumun BB genişliklerini hesapla
        widths = []
        for j in range(1, squeeze_lookback + 1):
            w = calc_bb_width(closes, idx - j, window, num_std)
            if w is not None:
                widths.append(w)

        if not widths:
            return False

        # Sıkışma: mevcut genişlik, geçmiş genişliklerin ortalamasının %70'inden az
        avg_width = sum(widths) / len(widths)
        is_squeeze = curr_width < avg_width * 0.70

        # Fiyat orta bandın üzerinde kapandı
        pw = closes.iloc[idx - window: idx + 1]
        middle_band  = float(pw.mean())
        curr_close   = float(closes.iloc[idx])
        above_middle = curr_close > middle_band

        return is_squeeze and above_middle
    except Exception:
        return False

def check_ema_squeeze_volume(df, closes, emas, idx, vol_mult=1.5):
    if idx < 20:
        return False
    try:
        e5       = float(emas[5].iloc[idx])
        e5_prev  = float(emas[5].iloc[idx - 1])
        e34      = float(emas[34].iloc[idx])
        e55      = float(emas[55].iloc[idx])
        e5_p3    = float(emas[5].iloc[idx - 3])
        e34_p3   = float(emas[34].iloc[idx - 3])
        c_curr   = float(closes.iloc[idx])
        c_prev   = float(closes.iloc[idx - 1])
        c_5ago   = float(closes.iloc[idx - 5])

        # Tepe filtresi: EMA55'in %10 üzerinde değil
        not_too_high    = c_curr < e55 * 1.10
        # Dip filtresi: 5 mum önce daha yüksekteydi
        came_from_above = c_5ago > c_curr * 1.02
        # EMA sıkışması: mesafe daralıyor
        squeeze         = abs(e5 - e34) < abs(e5_p3 - e34_p3) * 0.85
        # Yükseliş kırılımı: önceki mum EMA5 altındaydı, bu mum üstünde kapandı
        bullish_break   = c_prev < e5_prev and c_curr > e5

        if "Volume" not in df.columns:
            return False
        vol_curr  = float(df["Volume"].iloc[idx])
        vol_avg   = float(df["Volume"].iloc[idx - 20:idx].mean())
        vol_break = vol_curr > vol_avg * vol_mult

        return not_too_high and came_from_above and squeeze and vol_break and bullish_break
    except Exception:
        return False

def check_ema(emas, idx):
    try:
        e5   = float(emas[5].iloc[idx])
        e14  = float(emas[14].iloc[idx])
        e34  = float(emas[34].iloc[idx])
        e55  = float(emas[55].iloc[idx])
        bull = e5 > e14 > e34 > e55

        cross = pre_cross = False
        if idx >= 3:
            e5_p1  = float(emas[5].iloc[idx - 1])
            e5_p2  = float(emas[5].iloc[idx - 2])
            e14_p1 = float(emas[14].iloc[idx - 1])
            cross     = e5_p1 < e14_p1 and e5 > e14
            pre_cross = (e5 > e5_p1 > e5_p2) and e5 < e14 and (e14 - e5) / e14 * 100 < 1.5

        return bull, cross, pre_cross
    except Exception:
        return False, False, False

# ─── ANA TARAMA FONKSİYONU ───────────────────────────────────────────────────

def scan_ticker(ticker, interval, days_back, strategies, trend_period, son_n):
    end   = datetime.today()
    start = end - timedelta(days=days_back)
    try:
        df = yf.download(ticker, start=start, end=end,
                         interval=interval, progress=False, auto_adjust=True)
    except Exception:
        return []

    min_bars = {"1d": 60, "1wk": 30, "1mo": 12}
    if df is None or df.empty or len(df) < min_bars.get(interval, 30):
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

    warmup = min(55, n - son_n - 1)
    for i in range(max(warmup, trend_period, n - son_n), n):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        signals    = []
        div_pivots = None

        # Mum formasyonları
        if "Cekic" in strategies:
            if check_hammer(o, h, l, c) and is_downtrend(closes, i, trend_period):
                signals.append("Çekiç")

        if "Yutan" in strategies and i >= 1:
            o1 = float(df["Open"].iloc[i - 1])
            c1 = float(df["Close"].iloc[i - 1])
            if check_engulfing(o1, c1, o, c) and is_downtrend(closes, i, trend_period):
                signals.append("Yutan")

        if "Sabah Yildizi" in strategies and i >= 2:
            rows = [(float(df["Open"].iloc[i-2+j]), float(df["High"].iloc[i-2+j]),
                     float(df["Low"].iloc[i-2+j]),  float(df["Close"].iloc[i-2+j]))
                    for j in range(3)]
            if check_morning_star(rows):
                signals.append("Sabah Yıldızı")

        if "Three Inside Up" in strategies and i >= 2:
            rows3 = [(float(df["Open"].iloc[i-2+j]), float(df["High"].iloc[i-2+j]),
                      float(df["Low"].iloc[i-2+j]),  float(df["Close"].iloc[i-2+j]))
                     for j in range(3)]
            if check_three_inside_up(rows3) and is_downtrend(closes, i, trend_period):
                signals.append("Three Inside Up")

        if "Inside Bar" in strategies:
            if check_inside_bar_breakout(df, i):
                signals.append("Inside Bar Kırılım")

        # RSI stratejileri
        if "RSI Uyumsuzlugu" in strategies:
            lb  = min(30, n - 10) if interval == "1wk" else (min(18, n - 6) if interval == "1mo" else 60)
            lft = 2 if interval == "1wk" else (1 if interval == "1mo" else 3)
            div_ok, div_info = check_bullish_divergence(closes, rsi, i, lookback=lb, left=lft)
            if div_ok:
                signals.append("Pozitif RSI Uyumsuzluğu")
                div_pivots = div_info

        if "RSI Donus" in strategies:
            if check_rsi_bounce(closes, rsi, i):
                signals.append("RSI 30 Dönüşü")

        # Bollinger + EMA
        if "BB Squeeze" in strategies:
            if check_bollinger_squeeze_rsi(closes, rsi, i):
                signals.append("BB Squeeze+RSI50")

        if "EMA Hacim" in strategies:
            if check_ema_squeeze_volume(df, closes, emas, i):
                signals.append("EMA Sıkışma+Hacim")

        if "EMA Dizilimi" in strategies:
            bull, cross, pre_cross = check_ema(emas, i)
            if bull:       signals.append("EMA Dizilim")
            if cross:      signals.append("EMA Kesisim")
            if pre_cross:  signals.append("EMA Yaklasim")

        if signals:
            e = {p: round(float(emas[p].iloc[i]), 2) for p in [5, 14, 34, 55]}
            hf = n - 1 - i
            # div_pivots'u string olarak sakla — DataFrame bozulmasın
            dp_str = str(div_pivots) if div_pivots else ""
            results.append({
                "Hisse":    ticker.replace(".IS", ""),
                "Tarih":    df.index[i].strftime("%Y-%m-%d"),
                "Sinyal":   " | ".join(signals),
                "Kapanis":  round(c, 2),
                "RSI14":    round(float(rsi.iloc[i]), 1),
                "EMA5":     e[5],
                "EMA14":    e[14],
                "EMA34":    e[34],
                "EMA55":    e[55],
                "Zaman":    "Bu periyot" if hf == 0 else f"{hf} periyot once",
                "_ticker":  ticker,
                "_int":     interval,
                "_days":    days_back,
                "_dp":      dp_str,
            })
    return results

# ─── GRAFİK ──────────────────────────────────────────────────────────────────

def draw_chart(ticker, interval, days_back, signal_dates, div_pivots=None):
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)
    df = yf.download(ticker, start=start_dt, end=end_dt,
                     interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    closes = df["Close"].squeeze()
    emas   = calc_emas(closes)
    rsi    = calc_rsi(closes)
    xs     = list(range(len(df)))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.03)

    # Mum grafiği
    fig.add_trace(go.Candlestick(
        x=xs,
        open=df["Open"].squeeze().tolist(),
        high=df["High"].squeeze().tolist(),
        low=df["Low"].squeeze().tolist(),
        close=closes.tolist(),
        name=ticker,
        increasing_line_color="#00d4aa", decreasing_line_color="#ff6b6b",
        increasing_fillcolor="#00d4aa",  decreasing_fillcolor="#ff6b6b",
    ), row=1, col=1)

    # EMA çizgileri
    for p, col in {5:"#ffd166", 14:"#0099ff", 34:"#ff6b6b", 55:"#cc88ff"}.items():
        fig.add_trace(go.Scatter(x=xs, y=emas[p].tolist(),
                                 name=f"EMA{p}", line=dict(color=col, width=1.5)),
                      row=1, col=1)

    # Sinyal okları
    for sd in signal_dates:
        try:
            pos     = df.index.get_loc(pd.Timestamp(sd))
            low_val = float(df["Low"].iloc[pos]) * 0.98
            fig.add_annotation(x=pos, y=low_val, text="▲",
                                font=dict(color="#ffd166", size=18),
                                showarrow=False, row=1, col=1)
        except Exception:
            pass

    # RSI paneli
    fig.add_trace(go.Scatter(x=xs, y=rsi.tolist(), name="RSI 14",
                              line=dict(color="#a78bfa", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#ff6b6b", width=1, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#00d4aa", width=1, dash="dash"), row=2, col=1)

    # RSI uyumsuzluk çizgileri
    if div_pivots:
        try:
            p1_idx, p2_idx, price_p1, price_p2, rsi_p1, rsi_p2 = div_pivots
            low_p1 = float(df["Low"].iloc[p1_idx]) if p1_idx < len(df) else price_p1
            low_p2 = float(df["Low"].iloc[p2_idx]) if p2_idx < len(df) else price_p2
            fig.add_trace(go.Scatter(x=[p1_idx, p2_idx], y=[low_p1, low_p2],
                                     mode="lines+markers", name="Fiyat Diverjans",
                                     line=dict(color="#ff6b6b", width=2, dash="dot"),
                                     marker=dict(size=8, color="#ff6b6b")), row=1, col=1)
            fig.add_trace(go.Scatter(x=[p1_idx, p2_idx], y=[rsi_p1, rsi_p2],
                                     mode="lines+markers", name="RSI Diverjans",
                                     line=dict(color="#00d4aa", width=2, dash="dot"),
                                     marker=dict(size=8, color="#00d4aa")), row=2, col=1)
        except Exception:
            pass

    # X eksen etiketleri
    step    = max(1, len(df) // 10)
    tvals   = list(range(0, len(df), step))
    tlabels = [df.index[i].strftime("%Y-%m-%d") for i in tvals]

    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e2d40", tickmode="array",
                   tickvals=tvals, ticktext=tlabels, rangeslider=dict(visible=False)),
        xaxis2=dict(gridcolor="#1e2d40", tickmode="array",
                    tickvals=tvals, ticktext=tlabels),
        yaxis=dict(gridcolor="#1e2d40"),
        yaxis2=dict(gridcolor="#1e2d40", range=[0, 100]),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d40", borderwidth=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## BIST Tarayici")
    st.markdown("---")
    periyot  = st.selectbox("Periyot", list(PERIYOT_MAP.keys()), index=1)
    interval, days_back = PERIYOT_MAP[periyot]
    son_n    = st.slider("Son kac mum taransin?", 1, 10, 3)
    trend_per = st.slider("Trend periyodu (mum)", 3, 15, 5)
    st.markdown("---")
    strategies = st.multiselect(
        "Stratejiler",
        ["Cekic", "Yutan", "Sabah Yildizi", "Three Inside Up",
         "RSI Uyumsuzlugu", "RSI Donus", "BB Squeeze",
         "EMA Hacim", "Inside Bar", "EMA Dizilimi"],
        default=["RSI Donus", "BB Squeeze", "EMA Hacim", "Inside Bar"],
    )
    hisse_sec = st.multiselect(
        "Hisse filtresi (bos=tumu):",
        [h.replace(".IS", "") for h in HISSELER],
        default=[],
    )
    st.markdown("---")
    btn = st.button("TARAMAYI BASLAT")
    st.caption(f"{len(HISSELER)} hisse listede")

st.markdown("# BIST Formasyon Tarayici")
st.markdown(f"Periyot: {periyot} | Stratejiler: {', '.join(strategies) if strategies else '-'}")
st.warning("⚠️ Bu uygulama yalnızca teknik analiz amaçlıdır. Yatırım tavsiyesi değildir. Gösterilen sinyaller alım/satım önerisi içermez. Tüm yatırım kararları kullanıcının kendi sorumluluğundadır.")
st.markdown("---")

if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.done    = False

if btn:
    if not strategies:
        st.warning("En az bir strateji sec!")
    else:
        taranacak = ([h for h in HISSELER if h.replace(".IS", "") in hisse_sec]
                     if hisse_sec else HISSELER)

        st.session_state.results = []
        all_res = []
        prog    = st.progress(0)
        durum   = st.empty()
        sayi    = st.empty()

        for i, ticker in enumerate(taranacak):
            durum.caption(f"Taraniyor: {ticker}")
            prog.progress((i + 1) / len(taranacak))
            try:
                rows = scan_ticker(ticker, interval, days_back, strategies, trend_per, son_n)
                all_res.extend(rows)
            except Exception:
                pass
            sayi.markdown(f"Bulunan sinyal: {len(all_res)}")
            if (i + 1) % 25 == 0:
                time.sleep(0.3)

        prog.empty()
        durum.empty()
        st.session_state.results = all_res
        st.session_state.done    = True

if st.session_state.done:
    res = st.session_state.results
    if not res:
        st.info("Hic sinyal bulunamadi. Parametreleri gevset.")
    else:
        df_r = pd.DataFrame(res)
        df_r.sort_values(["Tarih", "Hisse"], ascending=[False, True], inplace=True)
        df_r.reset_index(drop=True, inplace=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Toplam Sinyal",    len(df_r))
        with c2: st.metric("Benzersiz Hisse",  df_r["Hisse"].nunique())
        with c3: st.metric("Bu Periyot",        len(df_r[df_r["Zaman"] == "Bu periyot"]))
        with c4: st.metric("Taranan",           len(HISSELER))

        st.markdown("---")
        f1, f2 = st.columns(2)
        with f1: fs = st.multiselect("Sinyale filtrele:",  df_r["Sinyal"].unique(), default=[])
        with f2: fz = st.multiselect("Zamana filtrele:", df_r["Zaman"].unique(),  default=[])

        df_show = df_r.copy()
        if fs: df_show = df_show[df_show["Sinyal"].isin(fs)]
        if fz: df_show = df_show[df_show["Zaman"].isin(fz)]

        st.markdown(f"### Sonuclar: {len(df_show)} sinyal")
        goster_cols = ["Hisse","Tarih","Sinyal","Kapanis","RSI14","EMA5","EMA14","EMA34","EMA55","Zaman"]
        st.dataframe(df_show[goster_cols], use_container_width=True, hide_index=True)

        csv_df = df_show[goster_cols].copy()
        st.download_button(
            "CSV Indir",
            csv_df.to_csv(index=False, encoding="utf-8-sig"),
            f"bist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )

        st.markdown("---")
        st.markdown("### Grafik")
        secim = st.selectbox("Hisse sec:", df_show["Hisse"].unique().tolist())
        if secim:
            secim_rows = df_show[df_show["Hisse"] == secim]
            dates      = secim_rows["Tarih"].tolist()

            # div_pivots string'den geri parse et
            div_pivots = None
            for _, srow in secim_rows.iterrows():
                if "Pozitif RSI Uyumsuzluğu" in str(srow.get("Sinyal","")) and srow.get("_dp"):
                    try:
                        div_pivots = eval(srow["_dp"])
                    except Exception:
                        pass
                    break

            draw_chart(secim + ".IS", interval, days_back, dates, div_pivots=div_pivots)

            row = secim_rows.iloc[0]
            e1, e2, e3, e4 = st.columns(4)
            with e1: st.metric("EMA 5",  row["EMA5"])
            with e2: st.metric("EMA 14", row["EMA14"])
            with e3: st.metric("EMA 34", row["EMA34"])
            with e4: st.metric("EMA 55", row["EMA55"])
else:
    st.info("Sol panelden periyot ve strateji sec, ardından TARAMAYI BASLAT butonuna bas.")
