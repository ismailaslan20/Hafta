import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime

st.set_page_config(page_title="Portföy Takip", page_icon="💼", layout="wide")

st.markdown("""
<style>
.stApp { background: #0a0e1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #111827; }
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0099ff);
    color: #0a0e1a; border: none; border-radius: 8px;
    font-weight: 700; width: 100%; padding: 0.6rem;
}
.kar   { color: #00d4aa; font-weight: 700; }
.zarar { color: #ff6b6b; font-weight: 700; }
.kart {
    background: #111827; border: 1px solid #1e2d40;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
    text-align: center;
}
.kart-baslik { font-size: 0.75rem; color: #64748b; text-transform: uppercase; margin-bottom: 0.3rem; }
.kart-deger  { font-size: 1.8rem; font-weight: 700; color: #00d4aa; }
</style>
""", unsafe_allow_html=True)

DOSYA = "portfolyo.json"

# ── Veri yükle / kaydet ───────────────────────
def yukle():
    if os.path.exists(DOSYA):
        with open(DOSYA, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"islemler": [], "satislar": []}

def kaydet(veri):
    with open(DOSYA, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)

# ── Anlık fiyat çek ───────────────────────────
def fiyat_cek(ticker):
    try:
        df = yf.download(ticker + ".IS", period="1d", interval="1m", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except:
        pass
    return None

# ── Ortalama maliyet hesapla ──────────────────
def ozet_hesapla(islemler):
    """Her hisse için kalan adet ve ortalama maliyet hesapla (FIFO)"""
    portfoy = {}
    for isl in islemler:
        ticker = isl["ticker"]
        if ticker not in portfoy:
            portfoy[ticker] = {"adet": 0, "toplam_maliyet": 0.0}
        portfoy[ticker]["adet"]          += isl["adet"]
        portfoy[ticker]["toplam_maliyet"] += isl["adet"] * isl["fiyat"]

    # Satışları uygula
    for satis in st.session_state.veri.get("satislar", []):
        ticker = satis["ticker"]
        if ticker in portfoy:
            portfoy[ticker]["adet"] -= satis["adet"]
            ort = portfoy[ticker]["toplam_maliyet"] / max(1, portfoy[ticker]["adet"] + satis["adet"])
            portfoy[ticker]["toplam_maliyet"] -= satis["adet"] * ort

    # Sıfır veya negatif adet olanları çıkar
    portfoy = {k: v for k, v in portfoy.items() if v["adet"] > 0.001}

    for ticker, v in portfoy.items():
        v["ort_maliyet"] = v["toplam_maliyet"] / v["adet"] if v["adet"] > 0 else 0

    return portfoy

# ── Session state ─────────────────────────────
if "veri" not in st.session_state:
    st.session_state.veri = yukle()
if "fiyatlar" not in st.session_state:
    st.session_state.fiyatlar = {}
if "son_guncelleme" not in st.session_state:
    st.session_state.son_guncelleme = None

veri = st.session_state.veri

# ── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 Portföy Takip")
    st.markdown("---")

    islem_turu = st.radio("İşlem Türü", ["📈 Alış", "📉 Satış"], horizontal=True)
    st.markdown("---")

    ticker_gir = st.text_input("Hisse Kodu (örn: EREGL)", "").upper().strip()
    adet_gir   = st.number_input("Adet", min_value=0.01, value=1.0, step=1.0)
    fiyat_gir  = st.number_input("Fiyat (TL)", min_value=0.01, value=1.0, step=0.01, format="%.2f")
    tarih_gir  = st.date_input("Tarih", value=datetime.today())
    not_gir    = st.text_input("Not (opsiyonel)", "")

    ekle_btn = st.button("✅ İşlemi Kaydet")

    st.markdown("---")
    guncelle_btn = st.button("🔄 Fiyatları Güncelle")

    st.markdown("---")
    if st.button("🗑️ Tüm Veriyi Sil", type="secondary"):
        st.session_state.veri = {"islemler": [], "satislar": []}
        kaydet(st.session_state.veri)
        st.session_state.fiyatlar = {}
        st.rerun()

# ── İşlem kaydet ──────────────────────────────
if ekle_btn and ticker_gir:
    portfoy = ozet_hesapla(veri["islemler"])

    if "Alış" in islem_turu:
        veri["islemler"].append({
            "ticker":  ticker_gir,
            "adet":    adet_gir,
            "fiyat":   fiyat_gir,
            "tarih":   str(tarih_gir),
            "not":     not_gir,
            "tur":     "alis"
        })
        st.sidebar.success(f"✅ {ticker_gir} alış kaydedildi!")
    else:
        # Satış kontrolü
        mevcut_adet = portfoy.get(ticker_gir, {}).get("adet", 0)
        if adet_gir > mevcut_adet:
            st.sidebar.error(f"❌ Yetersiz adet! Elimizde {mevcut_adet:.0f} adet var.")
        else:
            veri["satislar"].append({
                "ticker":     ticker_gir,
                "adet":       adet_gir,
                "satis_fiyat": fiyat_gir,
                "tarih":      str(tarih_gir),
                "not":        not_gir,
            })
            st.sidebar.success(f"✅ {ticker_gir} satış kaydedildi!")

    kaydet(veri)
    st.session_state.veri = veri
    st.rerun()

# ── Fiyat güncelle ────────────────────────────
portfoy = ozet_hesapla(veri["islemler"])

if guncelle_btn or not st.session_state.fiyatlar:
    with st.spinner("Fiyatlar güncelleniyor..."):
        for ticker in portfoy.keys():
            f = fiyat_cek(ticker)
            if f:
                st.session_state.fiyatlar[ticker] = f
        st.session_state.son_guncelleme = datetime.now().strftime("%H:%M:%S")

# ── ANA SAYFA ─────────────────────────────────
st.markdown("# 💼 Portföy Takip")
if st.session_state.son_guncelleme:
    st.caption(f"Son güncelleme: {st.session_state.son_guncelleme}")
st.markdown("---")

if not portfoy:
    st.info("Henüz hisse yok. Sol panelden alış ekleyin!")
else:
    # ── Özet kartlar ──────────────────────────
    toplam_maliyet  = 0.0
    toplam_deger    = 0.0
    toplam_kar      = 0.0

    satirlar = []
    for ticker, v in portfoy.items():
        adet       = v["adet"]
        ort_mal    = v["ort_maliyet"]
        gun_fiyat  = st.session_state.fiyatlar.get(ticker, None)
        maliyet    = adet * ort_mal
        deger      = adet * gun_fiyat if gun_fiyat else None
        kar_tl     = (deger - maliyet)         if deger else None
        kar_pct    = (kar_tl / maliyet * 100)  if deger else None

        toplam_maliyet += maliyet
        if deger:
            toplam_deger += deger
            toplam_kar   += kar_tl

        satirlar.append({
            "Hisse":        ticker,
            "Adet":         adet,
            "Ort. Maliyet": ort_mal,
            "Güncel Fiyat": gun_fiyat,
            "Toplam Maliyet (TL)": maliyet,
            "Güncel Değer (TL)":   deger,
            "Kar/Zarar (TL)":      kar_tl,
            "Kar/Zarar (%)":       kar_pct,
        })

    # Kartlar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kart">
            <div class="kart-baslik">Toplam Maliyet</div>
            <div class="kart-deger">{toplam_maliyet:,.0f} ₺</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kart">
            <div class="kart-baslik">Güncel Değer</div>
            <div class="kart-deger">{toplam_deger:,.0f} ₺</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        renk = "kar" if toplam_kar >= 0 else "zarar"
        isaret = "+" if toplam_kar >= 0 else ""
        st.markdown(f"""
        <div class="kart">
            <div class="kart-baslik">Toplam Kar/Zarar</div>
            <div class="kart-deger {renk}">{isaret}{toplam_kar:,.0f} ₺</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        pct = (toplam_kar / toplam_maliyet * 100) if toplam_maliyet > 0 else 0
        renk = "kar" if pct >= 0 else "zarar"
        isaret = "+" if pct >= 0 else ""
        st.markdown(f"""
        <div class="kart">
            <div class="kart-baslik">Toplam Getiri</div>
            <div class="kart-deger {renk}">{isaret}{pct:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Pozisyonlar")

    # Tablo
    df_goster = pd.DataFrame(satirlar)

    def kar_renk(val):
        try:
            temiz = str(val).replace("₺","").replace("%","").replace("+","").replace(" ","").replace(",","")
            return "color: #00d4aa; font-weight: bold" if float(temiz) >= 0 else "color: #ff6b6b; font-weight: bold"
        except:
            return ""

    def fmt(val, suffix=""):
        if val is None:
            return "-"
        if isinstance(val, float):
            return f"{val:,.2f}{suffix}"
        return str(val)

    df_display = df_goster.copy()
    df_display["Adet"]                 = df_display["Adet"].apply(lambda x: fmt(x))
    df_display["Ort. Maliyet"]         = df_display["Ort. Maliyet"].apply(lambda x: fmt(x, " ₺"))
    df_display["Güncel Fiyat"]         = df_display["Güncel Fiyat"].apply(lambda x: fmt(x, " ₺"))
    df_display["Toplam Maliyet (TL)"]  = df_display["Toplam Maliyet (TL)"].apply(lambda x: fmt(x, " ₺"))
    df_display["Güncel Değer (TL)"]    = df_display["Güncel Değer (TL)"].apply(lambda x: fmt(x, " ₺"))
    df_display["Kar/Zarar (TL)"]       = df_display["Kar/Zarar (TL)"].apply(lambda x: ("+" if x and x >= 0 else "") + fmt(x, " ₺") if x is not None else "-")
    df_display["Kar/Zarar (%)"]        = df_display["Kar/Zarar (%)"].apply(lambda x: ("+" if x and x >= 0 else "") + fmt(x, "%") if x is not None else "-")

    styled = df_display.style.applymap(kar_renk, subset=["Kar/Zarar (TL)", "Kar/Zarar (%)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Satış geçmişi ─────────────────────────
    if veri.get("satislar"):
        st.markdown("---")
        st.markdown("### 📤 Satış Geçmişi")

        satis_rows = []
        for s in veri["satislar"]:
            # Alış ortalamasını bul
            ticker = s["ticker"]
            # O tarihe kadarki alışları bul
            alinan = [i for i in veri["islemler"] if i["ticker"] == ticker]
            if alinan:
                top_mal = sum(i["adet"] * i["fiyat"] for i in alinan)
                top_adet = sum(i["adet"] for i in alinan)
                ort = top_mal / top_adet if top_adet > 0 else 0
            else:
                ort = 0

            satis_kar_tl  = (s["satis_fiyat"] - ort) * s["adet"]
            satis_kar_pct = ((s["satis_fiyat"] - ort) / ort * 100) if ort > 0 else 0

            satis_rows.append({
                "Tarih":           s["tarih"],
                "Hisse":           ticker,
                "Adet":            s["adet"],
                "Alış Ort.":       f"{ort:.2f} ₺",
                "Satış Fiyatı":    f"{s['satis_fiyat']:.2f} ₺",
                "Kar/Zarar (TL)":  f"{'+'if satis_kar_tl>=0 else ''}{satis_kar_tl:,.2f} ₺",
                "Kar/Zarar (%)":   f"{'+'if satis_kar_pct>=0 else ''}{satis_kar_pct:.2f}%",
                "Not":             s.get("not", ""),
            })

        df_satis = pd.DataFrame(satis_rows)
        st.dataframe(df_satis, use_container_width=True, hide_index=True)

    # ── İşlem geçmişi ─────────────────────────
    with st.expander("📋 Tüm İşlem Geçmişi"):
        if veri["islemler"]:
            df_isl = pd.DataFrame(veri["islemler"])
            df_isl.columns = ["Hisse", "Adet", "Fiyat", "Tarih", "Not", "Tür"]
            st.dataframe(df_isl, use_container_width=True, hide_index=True)
        else:
            st.info("Henüz işlem yok.")

    # CSV indir
    st.markdown("---")
    csv = df_display.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 Portföyü CSV İndir",
        csv,
        f"portfolyo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv"
    )
