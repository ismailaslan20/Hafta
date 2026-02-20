import streamlit as st
import pandas as pd
import numpy as np

# Sayfa ayarlarını mobil için optimize ediyoruz
st.set_page_config(page_title="Pusu Tarama Motoru", layout="wide")

st.title("🚀 Pusu Tarama Paneli")
st.write("Teknik analiz ve sinyal tarayıcı")

# Üst Menü - Ayarlar
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        # Hata aldığın o meşhur 46. satırı burada düzelttim
        periyot = st.selectbox("Tarama Periyodu", ["1h", "4h", "1d", "1w"])
    with col2:
        hisse_tipi = st.selectbox("Pazar/Borsa", ["Kripto", "BIST100", "Nasdaq"])

# Tarama Filtreleri
st.markdown("---")
st.subheader("🔍 Tarama Kriterleri")
st.info("Aşağıdaki kriterlere göre piyasa taranacaktır.")

c1, c2, c3 = st.columns(3)
with c1:
    rsi_filtre = st.checkbox("RSI (Aşırı Satım)", value=True)
with c2:
    golden_cross = st.checkbox("Golden Cross", value=True)
with c3:
    vol_artisi = st.checkbox("Hacim Artışı", value=True)

# Tarama Butonu (Mobil uyumlu genişlikte)
if st.button("TARAMAYI BAŞLAT", use_container_width=True):
    st.success(f"✅ {hisse_tipi} pazarı {periyot} periyodunda taranıyor...")
    
    # Örnek Tarama Sonuçları (Gerçek veriye bağlandığında burası dolacak)
    results = {
        "Sembol": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"],
        "Sinyal": ["AL", "GÜÇLÜ AL", "BEKLE", "AL"],
        "RSI": [32, 28, 45, 35],
        "Fiyat": ["52,400", "2,850", "110.5", "38.2"]
    }
    df = pd.DataFrame(results)
    
    st.markdown("### 📊 Tarama Sonuçları")
    st.dataframe(df, use_container_width=True)
    
    st.balloons() # Tarama bittiğinde görsel efekt
else:
    st.warning("Henüz tarama yapılmadı. Yukarıdaki butona basarak başlayabilirsin.")

st.markdown("---")
st.caption("Pusu (Ambush) v1.0 - Yiğit için özel geliştirilmiştir.")
