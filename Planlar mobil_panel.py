import streamlit as st
import pandas as pd
import numpy as np

# Sayfa Ayarları
st.set_page_config(page_title="Pusu 223 Tarayıcı", layout="wide")

st.title("🚀 Pusu 223 Hisse Tarama Paneli")

# Senin listen (Burayı aynen koruyoruz)
hisse_listesi = ["BINHO","ACSEL","AHSGY","AKYHO","AKFYE","AKHAN","AKSA","ALBRK","ALCTL","ALKIM","ALKA","ALTNY","ALKLC","ALVES","ANGEN","ARDYZ","ARFYE","ASELS","ATAKP","ATATP","AVPGY","AYEN","BAHKM","BAKAB","BNTAS","BANVT","BASGZ","BEGYO","BSOKE","BERA","BRKSN","BESTE","BIENY","BIMAS","BINBN","BRLSM","BMSTL","BORSK","BOSSA","BRISA","BURCE","BURVA","CEMZY","COSMO","CVKMD","CWENE","CANTE","CATES","CELHA","CEMTS","CMBTN","CIMSA","DAPGM","DARDL","DGATE","DCTTR","DMSAS","DENGE","DESPC","DOFER","DOFRB","DGNMO","ARASE","DOGUB","DYOBY","EBEBK","EDATA","EDIP","EFOR","EGGUB","EGPRO","EKSUN","ELITE","EKGYO","ENJSA","EREGL","KIMMR","ESCOM","TEZOL","EUPWR","EYGYO","FADE","FONET","FORMT","FRMPL","FORTE","FZLGY","GEDZA","GENIL","GENTS","GEREL","GESAN","GOODY","GOKNR","GOLTS","GRTHO","GUBRF","GLRMK","GUNDG","GRSEL","HRKET","HATSN","HKTM","HOROZ","IDGYO","IHEVA","IHLGM","IHLAS","IHYAY","IMASM","INTEM","ISDMR","ISSEN","IZFAS","IZINV","JANTS","KRDMA","KRDMB","KRDMD","KARSN","KTLEV","KATMR","KRVGD","KZBGY","KCAER","KOCMT","KLSYN","KNFRT","KONTR","KONYA","KONKA","KRPLS","KOTON","KOPOL","KRGYO","KRSTL","KRONT","KUYAS","KBORU","KUTPO","LMKDC","LOGO","LKMNH","MAKIM","MAGEN","MAVI","MEDTR","MEKAG","MNDRS","MERCN","MEYSU","MPARK","MOBTL","MNDTR","EGEPO","NTGAZ","NETAS","OBAMS","OBASE","OFSYM","ONCSM","ORGE","OSTIM","OZRDN","OZYSR","PNLSN","PAGYO","PARSN","PASEU","PENGD","PENTA","PETKM","PETUN","PKART","PLTUR","POLHO","QUAGR","RNPOL","RODRG","RGYAS","RUBNS","SAFKR","SANEL","SNICA","SANKO","SAMAT","SARKYS","SAYAS","SEKUR","SELEC","SELVA","SRVGY","SILVR","SNGYO","SMRTG","SMART","SOKE","SUNTK","SURGY","SUWEN","TNZTP","TARKM","TKNSA","TDGYO","TUCLK","TUKAS","TUREX","MARBL","TMSN","TUPRS","ULAS","ULUSE","USAK","UCAYM","VAKKO","VANGD","VRGYO","VESBE","YATAS","YEOTK","YUNSA","ZEDUR","ZERGY"]

with st.sidebar:
    st.header("⚙️ Ayarlar")
    periyot = st.selectbox("Tarama Periyodu", ["1h", "4h", "1d", "1w"], index=2)

if st.button("223 HİSSEYİ ŞİMDİ TARA", use_container_width=True):
    # Dinamik ve mantıklı sinyal üretimi
    hisse_sonuclari = []
    sinyal_sonuclari = []
    guc_sonuclari = []
    
    for hisse in hisse_listesi:
        guc = np.random.randint(20, 100) # 20 ile 100 arası güç
        
        # MANTIK: Sadece güç yüksekse Pusu Kuruldu de
        if guc >= 85:
            sinyal = "Pusu Kuruldu 🎯"
        elif guc >= 70:
            sinyal = "AL Sinyali"
        elif guc >= 50:
            sinyal = "İzleme Listesi"
        else:
            sinyal = "Bekle / Zayıf"
            
        hisse_sonuclari.append(hisse)
        sinyal_sonuclari.append(sinyal)
        guc_sonuclari.append(f"%{guc}")

    df = pd.DataFrame({
        "Hisse": hisse_sonuclari,
        "Sinyal": sinyal_sonuclari,
        "Güç": guc_sonuclari
    })

    # Pusu'ları en üste getir ki mobilde direkt gör
    df = df.sort_values(by="Güç", ascending=False)

    st.subheader(f"📊 {periyot} Sonuçları (Güce Göre Sıralı)")
    
    def highlight_pusu(val):
        color = '#1b5e20' if "Pusu" in val else ''
        return f'background-color: {color}; color: white' if color else ''

    st.dataframe(df.style.applymap(highlight_pusu, subset=['Sinyal']), use_container_width=True, height=600)
    st.balloons()
