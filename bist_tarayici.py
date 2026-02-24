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
    return {p: closes.ewm(span=p, adjust=False, min_periods=1).mean() for p in [5, 14, 20, 34, 50, 55, 200]}

def is_downtrend(closes, idx, period):
    if idx < period:
        return False
    return float(closes.iloc[idx - period]) > float(closes.iloc[idx - 1])

def check_bolge(closes, emas, rsi, idx, bolge):
    """
    Bölge filtresi:
    - Dip     : RSI < 45 ve fiyat EMA55 altında
    - Trend Devamı : EMA5 > EMA14 > EMA34 (yukarı dizilim)
    - İkisi de: her iki koşul da sağlanmalı
    - Filtre Yok: her zaman True
    """
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

        if bolge == "Dip Bölgesi":
            return dip_kosul
        elif bolge == "Trend Devamı":
            return trend_kosul
        elif bolge == "İkisi de":
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

        # EMA50 EMA200'ü yukarı kesti
        cross = (e50_prev < e200_prev) and (e50 > e200)
        # Fiyat EMA20 üstünde
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


# ─────────────────────────────────────────────
# YENİ: Düşeni Kıran Stratejisi
# Son 20 mum içindeki en yüksek fiyatı (direnci) yukarı kırma
# ─────────────────────────────────────────────
def check_duseni_kiran(df, idx, lookback=20):
    """
    Düşeni Kıran: Fiyat, son N mumdaki en yüksek seviyeyi (direnç) yukarı kırıyor
    Koşullar:
    1. Son lookback mumdaki en yüksek high seviyesini bul (direnç)
    2. Şimdiki kapanış, bu direncin üzerinde
    3. Hacim artışı var (önceki mumdan %20+ fazla)
    4. RSI 50-70 arası (güçlü ama aşırı alım değil)
    """
    if idx < lookback + 1:
        return False
    
    try:
        # Son lookback mumdaki en yüksek seviye (direnç)
        highs_window = df["High"].iloc[idx - lookback: idx]
        resistance = float(highs_window.max())
        
        # Şimdiki fiyat bilgileri
        current_close = float(df["Close"].iloc[idx])
        current_volume = float(df["Volume"].iloc[idx]) if "Volume" in df.columns else 0
        prev_volume = float(df["Volume"].iloc[idx - 1]) if "Volume" in df.columns else 0
        
        # Bir önceki mumun kapanışı direncin altında mıydı?
        prev_close = float(df["Close"].iloc[idx - 1])
        
        # Kırılma koşulları
        kiriyor = current_close > resistance and prev_close <= resistance * 1.01  # %1 tolerans
        hacim_artisi = current_volume > prev_volume * 1.2 if prev_volume > 0 else True
        
        return kiriyor and hacim_artisi
    except Exception:
        return False


# ─────────────────────────────────────────────
# YENİ: Pullback Stratejisi
# Fiyat yükselişten sonra destek seviyesine (EMA) geri çekilip tekrar yükseliyor
# ─────────────────────────────────────────────
def check_pullback(df, emas, rsi, idx, lookback=10):
    """
    Pullback: Yükseliş trendinde fiyat EMA'ya geri çekilip tekrar yükseliyor
    Koşullar:
    1. Genel trend yükseliş (EMA5 > EMA14 > EMA34)
    2. Son lookback mum içinde fiyat EMA20'ye yaklaştı veya dokundu
    3. Şimdiki mum EMA20'nin üstünde kapandı
    4. RSI 40-60 arası (sağlıklı geri çekilme)
    5. Şimdiki mum yeşil (kapanış > açılış)
    """
    if idx < lookback + 1:
        return False
    
    try:
        e5  = float(emas[5].iloc[idx])
        e14 = float(emas[14].iloc[idx])
        e20 = float(emas[20].iloc[idx])
        e34 = float(emas[34].iloc[idx])
        
        current_close = float(df["Close"].iloc[idx])
        current_open = float(df["Open"].iloc[idx])
        current_low = float(df["Low"].iloc[idx])
        current_rsi = float(rsi.iloc[idx])
        
        # 1. Genel trend yükseliş
        uptrend = e5 > e14 > e34
        
        # 2. Son lookback mum içinde EMA20'ye yaklaştı mı?
        touched_ema = False
        for i in range(max(0, idx - lookback), idx):
            low_i = float(df["Low"].iloc[i])
            close_i = float(df["Close"].iloc[i])
            ema20_i = float(emas[20].iloc[i])
            # EMA20'ye %2 içinde yaklaştı veya dokundu
            if low_i <= ema20_i * 1.02 or close_i <= ema20_i * 1.02:
                touched_ema = True
                break
        
        # 3. Şimdiki mum EMA20 üstünde kapandı
        above_ema20 = current_close > e20
        
        # 4. RSI sağlıklı aralıkta
        rsi_healthy = 40 <= current_rsi <= 60
        
        # 5. Yeşil mum
        green_candle = current_close > current_open
        
        return uptrend and touched_ema and above_ema20 and rsi_healthy and green_candle
    except Exception:
        return False


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


def scan_ticker(ticker, interval, days_back, strategies, trend_period, son_n, bolge_filtre="Filtre Yok"):
    end = datetime.today()
    start = end - timedelta(days=days_back)
    try:
        df = yf.download(
            ticker, start=start, end=end,
            interval=interval, progress=False, auto_adjust=False
        )
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

    # Tüm kolonları squeeze et - MultiIndex sorununu önle
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
                signals.append("Çekiç")

        if "Yutan" in strategies and i >= 1:
            o1 = float(df["Open"].iloc[i - 1])
            c1 = float(df["Close"].iloc[i - 1])
            if check_engulfing(o1, c1, o, c) and is_downtrend(closes, i, trend_period) and check_bolge(closes, emas, rsi, i, bolge_filtre):
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
            if check_morning_star(rows) and check_bolge(closes, emas, rsi, i, bolge_filtre):
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
            if check_three_inside_up(rows3) and is_downtrend(closes, i, trend_period) and check_bolge(closes, emas, rsi, i, bolge_filtre):
                signals.append("Three Inside Up")

        if "RSI Uyumsuzlugu" in strategies:
            if check_bullish_divergence(closes, rsi, i):
                signals.append("Pozitif RSI Uyumsuzluğu")

        # ── Golden Cross: EMA50 EMA200'ü yukarı kesiyor, fiyat EMA20 üstünde ──
        if "Golden Cross" in strategies:
            if check_golden_cross(emas, closes, i):
                signals.append("Golden Cross [EMA50>EMA200 & Fiyat>EMA20]")

        # ── YENİ: Düşeni Kıran ──
        if "Duseni Kiran" in strategies:
            if check_duseni_kiran(df, i):
                resistance = float(df["High"].iloc[i - 20: i].max())
                signals.append(f"Düşeni Kıran [Direnç: {resistance:.2f}]")

        # ── YENİ: Pullback ──
        if "Pullback" in strategies:
            if check_pullback(df, emas, rsi, i):
                ema20_val = float(emas[20].iloc[i])
                signals.append(f"Pullback [EMA20: {ema20_val:.2f}]")

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
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)
    df = yf.download(
        ticker, start=start_dt, end=end_dt,
        interval=interval, progress=False, auto_adjust=False
    )
    if df is None or df.empty:
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    closes = df["Close"].squeeze()
    df["Open"]  = df["Open"].squeeze()
    df["High"]  = df["High"].squeeze()
    df["Low"]   = df["Low"].squeeze()
    df["Close"] = df["Close"].squeeze()
    
    emas   = calc_emas(closes)
    rsi    = calc_rsi(closes)

    from plotly.subplots import make_subplots

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
            open=df["Open"].tolist(),
            high=df["High"].tolist(),
            low=df["Low"].tolist(),
            close=closes.tolist(),
            name=ticker,
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff6b6b",
            increasing_fillcolor="#00d4aa",
            decreasing_fillcolor="#ff6b6b",
        ),
        row=1, col=1,
    )

    colors = {5: "#ffd166", 14: "#0099ff", 20: "#00d4aa", 34: "#ff6b6b", 55: "#cc88ff"}
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

    # Düşeni Kıran ve Pullback sinyal işaretleri
    for sd in signal_dates:
        try:
            ts  = pd.Timestamp(sd)
            pos = df.index.get_loc(ts)
            
            # Düşeni Kıran: Son 20 mumdaki en yüksek direnci çiz
            if pos >= 20:
                lookback_start = max(0, pos - 20)
                resistance = float(df["High"].iloc[lookback_start:pos].max())
                
                # Direnç çizgisi (kırmızı kesikli)
                fig.add_shape(
                    type="line",
                    x0=lookback_start, x1=pos,
                    y0=resistance, y1=resistance,
                    line=dict(color="#ff6b6b", width=2, dash="dash"),
                    row=1, col=1,
                )
                
                # Direnç etiketi
                fig.add_annotation(
                    x=pos - 10, y=resistance,
                    text=f"Direnç: {resistance:.2f}",
                    showarrow=False,
                    font=dict(color="#ff6b6b", size=10),
                    bgcolor="rgba(0,0,0,0.5)",
                    row=1, col=1,
                )
            
            # Pullback: EMA20'ye dokunma bölgesini vurgula
            if pos >= 10:
                lookback_start = max(0, pos - 10)
                ema20_val = float(emas[20].iloc[pos])
                
                # EMA20 geri çekilme bölgesini yeşil arka plan ile vurgula
                for i in range(lookback_start, pos):
                    low_i = float(df["Low"].iloc[i])
                    if low_i <= ema20_val * 1.02:  # EMA20'ye %2 içinde yaklaştı
                        fig.add_vrect(
                            x0=i-0.5, x1=i+0.5,
                            fillcolor="rgba(0, 212, 170, 0.1)",
                            line_width=0,
                            row=1, col=1,
                        )
            
            # Sinyal ok işareti
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

    # RSI paneli
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

    step   = max(1, len(df) // 10)
    tvals  = list(range(0, len(df), step))
    tlabels = [df.index[i].strftime("%Y-%m-%d") for i in tvals]

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
        height=620,
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## BIST Tarayici")
    st.markdown("---")
    periyot = st.selectbox("Periyot", list(PERIYOT_MAP.keys()), index=1)
    interval, days_back = PERIYOT_MAP[periyot]

    son_n = st.slider("Son kac mum taransin?", 1, 10, 3)
    trend_per = st.slider("Trend periyodu (mum)", 3, 15, 5)
    bolge_filtre = st.selectbox(
        "Mum Formasyonu Bölge Filtresi:",
        ["Filtre Yok", "Dip Bölgesi", "Trend Devamı", "İkisi de"],
        index=0,
    )
    st.markdown("---")
    strategies = st.multiselect(
        "Stratejiler",
        ["Cekic", "Yutan", "Sabah Yildizi", "Three Inside Up",
         "RSI Uyumsuzlugu", "Golden Cross", "Duseni Kiran", "Pullback"],
        default=["Duseni Kiran", "Pullback"],
    )
    hisse_sec = st.multiselect(
        "Hisse filtresi (bos=tumu):",
        [h.replace(".IS", "") for h in HISSELER],
        default=[],
    )
    st.markdown("---")
    btn = st.button("TARAMAYI BASLAT")
    st.caption(str(len(HISSELER)) + " hisse listede")

    st.markdown("---")
    st.info(
        "**Yeni Stratejiler:**\n\n"
        "🔥 **Düşeni Kıran**: Son 20 mumdaki en yüksek seviyeyi (direnç) yukarı kırma. Hacim artışı ile doğrulanır.\n\n"
        "📉 **Pullback**: Yükseliş trendinde fiyat EMA20'ye geri çekilip tekrar yükseliyor. Sağlıklı düzeltme fırsatı."
    )

st.markdown("# BIST Formasyon Tarayici")
st.markdown("Periyot: " + periyot + " | Stratejiler: " + (", ".join(strategies) if strategies else "-"))
st.markdown("---")

if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.done = False
    st.session_state.bolge_filtre = "Filtre Yok"

if btn:
    if not strategies:
        st.warning("En az bir strateji sec!")
    else:
        st.session_state.bolge_filtre = bolge_filtre
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
            rows = scan_ticker(ticker, interval, days_back, strategies, trend_per, son_n, st.session_state.bolge_filtre)
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

        show_cols = ["Hisse", "Tarih", "Sinyal", "Kapanis", "RSI14", "EMA5", "EMA14", "EMA34", "EMA55", "Zaman"]

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

            draw_chart(ticker_f, interval, days_back, dates)

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
else:
    st.info("Sol panelden periyot ve strateji sec, ardından TARAMAYI BASLAT butonuna bas.")
