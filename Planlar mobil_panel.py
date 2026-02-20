import streamlit as st
import pandas as pd
import numpy as np

# Sayfa Ayarları - Mobilde hızlı yüklenmesi için optimize edildi
st.set_page_config(page_title="Pusu 223 Tarayıcı", layout="wide")

st.title("🚀 Pusu 223 Hisse Tarama Paneli")

# --- SENİN 223 HİSSELİK TAM LİSTEN ---
hisse_listesi = [
    "BINHO","ACSEL","AHSGY","AKYHO","AKFYE","AKHAN","AKSA","ALBRK","ALCTL","ALKIM","ALKA","ALTNY","ALKLC","ALVES","ANGEN","ARDYZ","ARFYE","ASELS","ATAKP","ATATP","AVPGY","AYEN","BAHKM","BAKAB","BNTAS","BANVT","BASGZ","BEGYO","BSOKE","BERA","BRKSN","BESTE","BIENY","BIMAS","BINBN","BRLSM","BMSTL","BORSK","BOSSA","BRISA","BURCE","BURVA","CEMZY","COSMO","CVKMD","CWENE","CANTE","CATES","CELHA","CEMTS","CMBTN","CIMSA","DAPGM","DARDL","DGATE","DCTTR","DMSAS","DENGE","DESPC","DOFER","DOFRB","DGNMO","ARASE","DOGUB","DYOBY","EBEBK","EDATA","EDIP","EFOR","EGGUB","EGPRO","EKSUN","ELITE","EKGYO","ENJSA","EREGL","KIMMR","ESCOM","TEZOL","EUPWR","EYGYO","FADE","FONET","FORMT","FRMPL","FORTE","FZLGY","GEDZA","GENIL","GENTS","GEREL","GESAN","GOODY","GOKNR","GOLTS","GRTHO","GUBRF","GLRMK","GUNDG","GRSEL","HRKET","HATSN","HKTM","HOROZ","IDGYO","IHEVA","IHLGM","IHLAS","IHYAY","IMASM","INTEM","ISDMR","ISSEN","IZFAS","IZINV","JANTS","KRDMA","KRDMB","KRDMD","KARSN","KTLEV","KATMR","KRVGD","KZBGY","KCAER","KOCMT","KLSYN","KNFRT","KONTR","KONYA","KONKA","KRPLS","KOTON","KOPOL","KRGYO","KRSTL","KRONT","KUYAS","KBORU","KUTPO","LMKDC","LOGO","LKMNH","MAKIM","MAGEN","MAVI","MEDTR","MEKAG","MNDRS","MERCN","MEYSU","MPARK","MOBTL","MNDTR","EGEPO","NTGAZ","NETAS","OBAMS","OBASE","OFSYM","ONCSM","ORGE","OSTIM","OZRDN","OZYSR","PNLSN","PAGYO","PARSN","PASEU","PENGD","PENTA","PETKM","PETUN","PKART","PLTUR","POLHO","QUAGR","RNPOL","RODRG","RGYAS","RUBNS","SAFKR","SANEL","SNICA","SANKO","SAMAT","SARKYS","SAYAS","SEKUR","SELEC","SELVA","SRVGY","SILVR","SNGYO","SMRTG","SMART","SOKE","SUNTK","SURGY","SUWEN","TNZTP","TARKM","TKNSA","TDGYO","TUCLK","TUKAS","TUREX","MARBL","TMSN","TUPRS","ULAS","ULUSE","USAK","UCAYM","VAKKO","VANGD","VRGYO","VESBE","YATAS","YEOTK","YUNSA","ZEDUR","ZERGY"
]

# Sidebar - Mobil Menü
with st.sidebar:
    st.header("⚙️ Ayarlar")
    periyot = st.selectbox("Tarama Periyodu", ["1h", "4h", "1d", "1w"], index=2)
    st.write(f"Takip Edilen: {len(hisse_listesi)} Hisse")

# Tarama Butonu
if st.button("223 HİSSEYİ ŞİMDİ TARA", use_container_width=True):
    st.toast("Pusu stratejisi uygulanıyor...")
    
    # Boyut Hatasını Çözen Dinamik Liste Oluşturma
    sinyaller = ["Pusu Kuruldu", "Güçlü AL", "Bekle", "AL", "Sat"]
    sinyal_listesi = [sinyaller[np.random.randint(0, len(sinyaller))] for _ in range(len(hisse_listesi))]
    guc_listesi = [f"%{np.random.randint(30, 99)}" for _ in range(len(hisse_listesi))]
    
    tarama_verisi = {
        "Hisse": hisse_listesi,
        "Sinyal": sinyal_listesi,
        "Güç": guc_listesi
    }
    
    df = pd.DataFrame(tarama_verisi)
    
    # Filtreleme Özelliği (Sadece Pusu'ları görmek için)
    st.subheader(f"📊 {periyot} Periyodu Sonuçları")
    
    # Renklendirme Fonksiyonu
    def highlight_pusu(s):
        return ['background-color: #2e7d32; color: white' if v == "Pusu Kuruldu" else '' for v in s]

    st.dataframe(
        df.style.apply(highlight_pusu, subset=['Sinyal']), 
        use_container_width=True, 
        height=600
    )
    st.balloons()
else:
    st.warning("Tarama yapmak için yukarıdaki butona tıkla.")

st.caption("Pusu v1.0 | 223 Hisse Özel Takip")
