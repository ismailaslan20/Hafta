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

def calc_emas(closes):
    return {p: closes.ewm(span=p, adjust=False).mean() for p in [5, 14, 34, 55]}

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

def check_ema(emas, idx):
    try:
        e5  = float(emas[5].iloc[idx])
        e14 = float(emas[14].iloc[idx])
        e34 = float(emas[34].iloc[idx])
        e55 = float(emas[55].iloc[idx])
        bull = e5 > e14 > e34 > e55
        cross = False
        if idx >= 2:
            cross = (
                float(emas[5].iloc[idx - 1]) < float(emas[14].iloc[idx - 1])
                and e5 > e14
            )
        return bull, cross
    except Exception:
        return False, False

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
    if df is None or df.empty or len(df) < 60:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return []

    closes = df["Close"].squeeze()
    emas = calc_emas(closes)
    n = len(df)
    results = []

    for i in range(max(55, trend_period, n - son_n), n):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        signals = []

        if "Çekiç" in strategies:
            if check_hammer(o, h, l, c) and is_downtrend(closes, i, trend_period):
                signals.append("Çekiç")

        if "Yutan" in strategies and i >= 1:
            o1 = float(df["Open"].iloc[i - 1])
            c1 = float(df["Close"].iloc[i - 1])
            if check_engulfing(o1, c1, o, c) and is_downtrend(closes, i, trend_period):
                signals.append("Yutan")

        if "Sabah Yıldızı" in strategies and i >= 2:
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

        if "EMA Dizilimi" in strategies:
            bull, cross = check_ema(emas, i)
            if bull:
                signals.append("EMA Dizilim")
            if cross:
                signals.append("EMA Kesisim")

        if signals:
            e = {p: round(float(emas[p].iloc[i]), 2) for p in [5, 14, 34, 55]}
            hafta_farki = n - 1 - i
            results.append({
                "Hisse":   ticker.replace(".IS", ""),
                "Tarih":   df.index[i].strftime("%Y-%m-%d"),
                "Sinyal":  " | ".join(signals),
                "Kapanis": round(c, 2),
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
    end = datetime.today()
    start = end - timedelta(days=days_back)
    df = yf.download(
        ticker, start=start, end=end,
        interval=interval, progress=False, auto_adjust=True
    )
    if df is None or df.empty:
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    closes = df["Close"].squeeze()
    emas = calc_emas(closes)

    fig = go.Figure()
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

    colors = {5: "#ffd166", 14: "#0099ff", 34: "#ff6b6b", 55: "#cc88ff"}
    for p, col in colors.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=emas[p],
            name="EMA" + str(p),
            line=dict(color=col, width=1.5)
        ))

    for sd in signal_dates:
        try:
            ts = pd.Timestamp(sd)
            row = df.loc[ts]
            fig.add_annotation(
                x=ts,
                y=float(row["Low"].squeeze()) * 0.98,
                text="▲",
                font=dict(color="#ffd166", size=18),
                showarrow=False,
            )
        except Exception:
            pass

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e2d40", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#1e2d40"),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d40", borderwidth=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=500,
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
        ["Cekic", "Yutan", "Sabah Yildizi", "EMA Dizilimi"],
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
            df_show[["Hisse", "Tarih", "Sinyal", "Kapanis", "EMA5", "EMA14", "EMA34", "EMA55", "Zaman"]],
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




**requirements.txt içeriği:**
```
streamlit==1.32.2
yfinance
pandas
numpy
plotly
altair==4.2.2
