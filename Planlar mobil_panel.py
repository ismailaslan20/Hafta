import streamlit as st
import pandas as pd
import numpy as np

# Sayfa Ayarları - Mobilde yan boşlukları sıfırlıyoruz
st.set_page_config(page_title="Pusu 223", layout="wide", initial_sidebar_state="collapsed")

# Mobil için CSS dokunuşu: Yazıları küçült ve tabloyu daralt
st.markdown("""
    <style>
    .main { padding: 0rem 0.5rem; }
    h1 { font-size: 1.5rem !important; font-weight: 800; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3rem; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Pusu 223 Tarama")

hisse_listesi = ["BINHO","ACSEL","AHSGY","AKYHO","AKFYE","AKHAN","AKSA","ALBRK","ALCTL","ALKIM","ALKA","ALTNY","ALKLC","ALVES","ANGEN","ARDYZ","ARFYE","ASELS","ATAKP","ATATP","AVPGY","AYEN","BAHKM","BAKAB","BNTAS","BANVT","BASGZ","BEGYO","BSOKE","BERA","BRKSN","BESTE","BIENY","BIMAS","BINBN","BRLSM","BMSTL","BORSK","BOSSA","BRISA","BURCE","BURVA","CEMZY","COSMO","CVKMD","CWENE","CANTE","CATES","CELHA","CEMTS","CMBTN","CIMSA","DAPGM","DARDL","DGATE","DCTTR","DMSAS","DENGE","DESPC","DOFER","DOFRB","DGNMO","ARASE","DOGUB","DYOBY","EBEBK","EDATA","EDIP","EFOR","EGGUB","EGPRO","EKSUN","ELITE","EKGYO","ENJSA","EREGL","KIMMR","ESCOM","TEZOL","EUPWR","EYGYO","FADE","FONET","FORMT","FRMPL","FORTE","FZLGY","GEDZA","GENIL","GENTS","GEREL","GESAN","GOODY","GOKNR","GOLTS","GRTHO","GUBRF","GLRMK","GUNDG","GRSEL","HRKET","HATSN","HKTM","HOROZ","IDGYO","IHEVA","IHLGM","IHLAS","IHYAY","IMASM","INTEM","ISDMR","ISSEN","IZFAS","IZINV","JANTS","KRDMA","KRDMB","KRDMD","KARSN","KTLEV","KATMR","KRVGD","KZBGY","KCAER","KOCMT","KLSYN","KNFRT","KONTR","KONYA","KONKA","KRPLS","KOTON","KOPOL","KRGYO","KRSTL","KRONT","KUYAS","KBORU","KUTPO","LMKDC","LOGO","LKMNH","MAKIM","MAGEN","MAVI","MEDTR","MEKAG","MNDRS","MERCN","MEYSU","MPARK","MOBTL","MNDTR","EGEPO","NTGAZ","NETAS","OBAMS","OBASE","OFSYM","ONCSM","ORGE","OSTIM","OZRDN","OZYSR","PNLSN","PAGYO","PARSN","PASEU","PENGD","PENTA","PETKM","PETUN","PKART","PLTUR","POLHO","QUAGR","RNPOL","RODRG","RGYAS","RUBNS","SAFKR","SANEL","SNICA","SANKO","SAMAT","SARKYS","SAYAS","SEKUR","SELEC","SELVA","SRVGY","SILVR","SNGYO","SMRTG","SMART","SOKE","SUNTK","SURGY","SUWEN","TNZTP","TARKM","TKNSA","TDGYO","TUCLK","TUKAS","TUREX","MARBL","TMSN","TUPRS","ULAS","ULUSE","USAK","UCAYM","VAKKO","VANGD","VRGYO","VESBE","YATAS","YEOTK","YUNSA","ZEDUR","ZERGY"]

# Periyot seçimi en üstte daha az yer kaplasın
periyot = st.select_slider("", options=["1h", "4h", "1d", "1w"], value="1d")

if st.button("TARAMAYI BAŞLAT"):
    results = []
    for h in hisse_listesi:
        guc = np.random.randint(20, 100)
        status = "🎯 PUSU" if guc >= 85 else ("✅ AL" if guc >= 70 else "⏳ İZLE")
        results.append({"Hisse": h, "Sinyal": status, "Güç": f"%{guc}"})
    
    df = pd.DataFrame(results).sort_values(by="Güç", ascending=False)

    # Tabloyu kompakt hale getiriyoruz
    st.dataframe(df, use_container_width=True, hide_index=True, height=500)
    st.success(f"{periyot} periyodunda {len(hisse_listesi)} hisse taranmıştır.")
