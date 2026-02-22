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
    # Mum sayısı azsa EMA periyotları kendiliğinden kısalır (ewm min_periods=1)
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
    """
    Three Inside Up:
    1. Mum: Büyük bearish (düşüş) mumu
    2. Mum: 1. mumun gövdesi içinde kalan küçük bullish mumu (harami)
    3. Mum: 1. mumun kapanışının üzerinde kapanan bullish mumu
    """
    if len(r) < 3:
        return False
    o1, h1, l1, c1 = r[0]  # Büyük bearish mum
    o2, h2, l2, c2 = r[1]  # Küçük bullish mum (harami)
    o3, h3, l3, c3 = r[2]  # Onaylayıcı bullish mum

    bearish_1 = c1 < o1
    bullish_2 = c2 > o2
    bullish_3 = c3 > o3

    # 2. mum 1. mumun gövdesi içinde kalmalı (harami)
    inside = o2 >= c1 and c2 <= o1

    # 3. mum 1. mumun açılışının üzerinde kapanmalı
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

def find_pivot_lows(series, left=3, right=3):
    """
    Gerçek pivot dipleri bulur: solunda ve sağında 'left'/'right' kadar
    mumdan daha düşük olan noktalar. TradingView mantığıyla aynı.
    """
    pivots = []
    vals = series.values
    for i in range(left, len(vals) - right):
        window = list(vals[i - left:i]) + list(vals[i + 1:i + right + 1])
        if all(vals[i] <= w for w in window):
            pivots.append(i)
    return pivots

def check_bullish_divergence(closes, rsi, idx, lookback=60, left=3, right=3):
    """
    Pivot tabanlı Pozitif RSI Uyumsuzluğu (TradingView tarzı):
    1. Son mum bir pivot dip mi? (sağ taraf için right mum gerekir → idx en az right kadar geride olmalı)
    2. Lookback penceresi içinde daha önceki bir pivot dip var mı?
    3. Fiyat: son pivot < önceki pivot (lower low)
    4. RSI:   son pivot RSI > önceki pivot RSI (higher low)
    5. İkinci pivot RSI < 60 (aşırı satım veya orta bölge)
    """
    if idx < lookback + left + right:
        return False

    # Tüm seri üzerinde pivot dipleri hesapla, sonra pencereye filtrele
    window_closes = closes.iloc[idx - lookback: idx + 1]
    window_rsi    = rsi.iloc[idx - lookback: idx + 1]

    if window_rsi.isna().sum() > lookback * 0.3:
        return False

    local_pivots = find_pivot_lows(window_closes, left=left, right=right)

    # En az 2 pivot gerekli
    if len(local_pivots) < 2:
        return False

    # Son iki pivot
    p2_local = local_pivots[-1]  # daha yeni pivot (pencere içi indeks)
    p1_local = local_pivots[-2]  # daha eski pivot

    # Pencere içi → gerçek indeks dönüşümü
    base = idx - lookback
    p2 = base + p2_local
    p1 = base + p1_local

    if p2 >= len(closes) or p1 >= len(closes):
        return False

    price_p1 = float(closes.iloc[p1])
    price_p2 = float(closes.iloc[p2])
    rsi_p1   = float(rsi.iloc[p1])
    rsi_p2   = float(rsi.iloc[p2])

    if np.isnan(rsi_p1) or np.isnan(rsi_p2):
        return False

    # Koşullar
    price_lower_low = price_p2 < price_p1        # fiyat daha düşük dip
    rsi_higher_low  = rsi_p2   > rsi_p1          # RSI daha yüksek dip
    rsi_not_overbought = rsi_p2 < 60             # aşırı alım bölgesinde değil
    pivots_close_enough = (p2 - p1) <= lookback  # çok uzak olmasın

    return price_lower_low and rsi_higher_low and rsi_not_overbought and pivots_close_enough

def check_ema(emas, idx):
    """
    3 farklı EMA sinyali döndürür:
    - bull:          Tam dizilim (EMA5>14>34>55) — trend onayı
    - cross:         EMA5, EMA14'ü bu mumda yukarı kesti — kesişim anı
    - pre_cross:     Henüz kesmedi ama EMA5 eğimi pozitife döndü
                     ve EMA14'e %1.5'ten az kaldı — erken uyarı
    """
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

            # Tam kesişim: önceki mumda EMA5 < EMA14, bu mumda EMA5 > EMA14
            cross = e5_prev1 < e14_prev1 and e5 > e14

            # Erken uyarı: henüz kesmedi ama
            #   1) EMA5 son 2 mumda yükseliyor (eğim pozitif)
            #   2) EMA5 hâlâ EMA14 altında
            #   3) Aradaki fark EMA14'ün %1.5'inden az (yaklaşıyor)
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

    # Periyota göre dinamik minimum mum eşiği
    min_bars = {"1d": 60, "1wk": 30, "1mo": 12}
    min_required = min_bars.get(interval, 30)

    if df is None or df.empty or len(df) < min_required:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return []

    closes = df["Close"].squeeze()
    emas = calc_emas(closes)
    rsi  = calc_rsi(closes)
    n = len(df)
    results = []

    # EMA55 için yeterli mum yoksa EMA periyodunu mum sayısına göre sınırla
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
            if check_bullish_divergence(closes, rsi, i, lookback=60, left=3, right=3):
                signals.append("Pozitif RSI Uyumsuzluğu")

        if "EMA Dizilimi" in strategies:
            bull, cross, pre_cross = check_ema(emas, i)
            if bull:
                signals.append("EMA Dizilim")
            if cross:
                signals.append("EMA Kesisim")
            if pre_cross:
                signals.append("EMA Yaklasim")

        if signals:
            e = {p: round(float(emas[p].iloc[i]), 2) for p in [5, 14, 34, 55]}
            hafta_farki = n - 1 - i
            results.append({
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
            })
    return results

def draw_chart(ticker, interval, days_back, signal_dates):
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

    # ── ÜST PANEL: Mum grafik + EMA ──────────────────────────────────────────
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
            x=list(range(len(df))),          # integer index → boşluk yok
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

    # EMA çizgileri
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

    # Sinyal okları
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

    # ── ALT PANEL: RSI çizgisi ────────────────────────────────────────────────
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

    # X eksen etiketleri: her 10 mumda bir tarih göster
    step   = max(1, len(df) // 10)
    tvals  = list(range(0, len(df), step))
    tlabels = [df.index[i].strftime("%Y-%m-%d") for i in tvals]

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"),
        xaxis=dict(
            gridcolor="#1e2d40",
            tickmode="array",
            tickvals=tvals,
            ticktext=tlabels,
            rangeslider=dict(visible=False),
        ),
        xaxis2=dict(
            gridcolor="#1e2d40",
            tickmode="array",
            tickvals=tvals,
            ticktext=tlabels,
        ),
        yaxis=dict(gridcolor="#1e2d40"),
        yaxis2=dict(gridcolor="#1e2d40", range=[0, 100]),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d40", borderwidth=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)


with st.sidebar:
    st.markdown("## BIST Tarayici")
    st.markdown("---")
    periyot = st.selectbox("Periyot", list(PERIYOT_MAP.keys()), index=1)
    interval, days_back = PERIYOT_MAP[periyot]
    son_n = st.slider("Son kac mum taransin?", 1, 10, 3)
    trend_per = st.slider("Trend periyodu (mum)", 3, 15, 5)
    st.markdown("---")
    strategies = st.multiselect(
        "Stratejiler",
        ["Cekic", "Yutan", "Sabah Yildizi", "Three Inside Up", "RSI Uyumsuzlugu", "EMA Dizilimi"],
        default=["Cekic", "EMA Dizilimi"],
    )
    hisse_sec = st.multiselect(
        "Hisse filtresi (bos=tumu):",
        [h.replace(".IS", "") for h in HISSELER],
        default=[],
    )
    st.markdown("---")
    btn = st.button("TARAMAYI BASLAT")
    st.caption(str(len(HISSELER)) + " hisse listede")

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

        st.markdown("### Sonuclar: " + str(len(df_show)) + " sinyal")
        st.dataframe(
            df_show[["Hisse", "Tarih", "Sinyal", "Kapanis", "RSI14", "EMA5", "EMA14", "EMA34", "EMA55", "Zaman"]],
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
