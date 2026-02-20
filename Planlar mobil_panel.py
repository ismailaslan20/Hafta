import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Pusu 223", layout="wide", initial_sidebar_state="collapsed")

# Mobil stil ayarı
st.markdown("<style>h1{font-size: 1.5rem !important;} .stButton>button{width:100%;}</style>", unsafe_allow_html=True)

st.title("🚀 Teknik Pusu 223")

hisse_listesi = ["BINHO","ACSEL","AHSGY","AKYHO","AKFYE","AKHAN","AKSA","ALBRK","ALCTL","ALKIM","ALKA","ALTNY","ALKLC","ALVES","ANGEN","ARDYZ","ARFYE","ASELS","ATAKP","ATATP","AVPGY","AYEN","BAHKM","BAKAB","BNTAS","BANVT","BASGZ","BEGYO","BSOKE","BERA","BRKSN","BESTE","BIENY","BIMAS","BINBN","BRLSM","BMSTL","BORSK","BOSSA","BRISA","BURCE","BURVA","CEMZY","COSMO","CVKMD","CWENE","CANTE","CATES","CELHA","CEMTS","CMBTN","CIMSA","DAPGM","DARDL","DGATE","DCTTR","DMSAS","DENGE","DESPC","DOFER","DOFRB","DGNMO","ARASE","DOGUB","DYOBY","EBEBK","EDATA","EDIP","EFOR","EGGUB","EGPRO","EKSUN","ELITE","EKGYO","ENJSA","EREGL","KIMMR","ESCOM","TEZOL","EUPWR","EYGYO","FADE","FONET","FORMT","FRMPL","FORTE","FZLGY","GEDZA","GENIL","GENTS","GEREL","GESAN","GOODY","GOKNR","Golts","GRTHO","GUBRF","GLRMK","GUNDG","GRSEL","HRKET","HATSN","HKTM","HOROZ","IDGYO","IHEVA","IHLGM","IHLAS","IHYAY","IMASM","INTEM","ISDMR","ISSEN","IZFAS","IZINV","JANTS","KRDMA","KRDMB","KRDMD","KARSN","KTLEV","KATMR","KRVGD","KZBGY","KCAER","KOCMT","KLSYN","KNFRT","KONTR","KONYA","KONKA","KRPLS","KOTON","KOPOL","KRGYO","KRSTL","KRONT","KUYAS","KBORU","KUTPO","LMKDC","LOGO","LKMNH","MAKIM","MAGEN","MAVI","MEDTR","MEKAG","MNDRS","MERCN","MEYSU","MPARK","MOBTL","MNDTR","EGEPO","NTGAZ","NETAS","OBAMS","OBASE","OFSYM","ONCSM","ORGE","OSTIM","OZRDN","OZYSR","PNLSN","PAGYO","PARSN","PASEU","PENGD","PENTA","PETKM","PETUN","PKART","PLTUR","POLHO","QUAGR","RNPOL","RODRG","RGYAS","RUBNS","SAFKR","SANEL","SNICA","SANKO","SAMAT","SARKYS","SAYAS","SEKUR","SELEC","SELVA","SRVGY","SILVR","SNGYO","SMRTG","SMART","SOKE","SUNTK","SURGY","SUWEN","TNZTP","TARKM","TKNSA","TDGYO","TUCLK","TUKAS","TUREX","MARBL","TMSN","TUPRS","ULAS","ULUSE","USAK","UCAYM","VAKKO","VANGD","VRGYO","VESBE","YATAS","YEOTK","YUNSA","ZEDUR","ZERGY"]

periyot = st.select_slider("Zaman Dilimi", options=["1h", "4h", "1d", "1w"], value="1d")

if st.button("STRATEJİK TARAMAYI BAŞLAT"):
    results = []
    for h in hisse_listesi:
        # Teknik Analiz Simülasyonu (Gerçek veri çekilene kadar mantıksal kurgu)
        # 0: EMA Altı (Ayı), 1: EMA Üstü (Boğa)
        ema200_durum = np.random.choice([0, 1], p=[0.4, 0.6])
        ema50_durum = np.random.choice([0, 1], p=[0.5, 0.5])
        
        if ema200_durum == 0:
            guc = np.random.randint(5, 40)
            status = "❌ Trend Altı"
        elif ema200_durum == 1 and ema50_durum == 0:
            guc = np.random.randint(40, 70)
            status = "⏳ EMA50 Bekliyor"
        else:
            guc = np.random.randint(75, 99)
            status = "🎯 PUSU (Trend Üstü)"
            
        results.append({"Hisse": h, "Durum": status, "Güç": f"%{guc}", "Puan": guc})
    
    # Güce göre sırala
    df = pd.DataFrame(results).sort_values(by="Puan", ascending=False)

    st.dataframe(df[["Hisse", "Durum", "Güç"]], use_container_width=True, hide_index=True, height=500)
    st.balloons()
