import streamlit as st
import pandas as pd

# Sayfa Ayarları
st.set_page_config(page_title="Pusu 223 Tarayıcı", layout="wide")

st.title("🚀 Özel 223 Hisse Tarama Motoru")

# --- SENİN 223 HİSSELİK LİSTEN ---
# Buradaki listenin içine tüm hisselerini aralarına virgül koyarak ekleyebilirsin
hisse_listesi = ["BINHO","ACSEL","AHSGY","AKYHO","AKFYE","AKHAN","AKSA","ALBRK","ALCTL","ALKIM","ALKA","ALTNY","ALKLC","ALVES","ANGEN","ARDYZ","ARFYE","ASELS","ATAKP","ATATP","AVPGY","AYEN","BAHKM","BAKAB","BNTAS","BANVT","BASGZ","BEGYO","BSOKE","BERA","BRKSN","BESTE","BIENY","BIMAS","BINBN","BRLSM","BMSTL","BORSK","BOSSA","BRISA","BURCE","BURVA","CEMZY","COSMO","CVKMD","CWENE","CANTE","CATES","CELHA","CEMTS","CMBTN","CIMSA","DAPGM","DARDL","DGATE","DCTTR","DMSAS","DENGE","DESPC","DOFER","DOFRB","DGNMO","ARASE","DOGUB","DYOBY","EBEBK","EDATA","EDIP","EFOR","EGGUB","EGPRO","EKSUN","ELITE","EKGYO","ENJSA","EREGL","KIMMR","ESCOM","TEZOL","EUPWR","EYGYO","FADE","FONET","FORMT","FRMPL","FORTE","FZLGY","GEDZA","GENIL","GENTS","GEREL","GESAN","GOODY","GOKNR","GOLTS","GRTHO","GUBRF","GLRMK","GUNDG","GRSEL","HRKET","HATSN","HKTM","HOROZ","IDGYO","IHEVA","IHLGM","IHLAS","IHYAY","IMASM","INTEM","ISDMR","ISSEN","IZFAS","IZINV","JANTS","KRDMA","KRDMB","KRDMD","KARSN","KTLEV","KATMR","KRVGD","KZBGY","KCAER","KOCMT","KLSYN","KNFRT","KONTR","KONYA","KONKA","KRPLS","KOTON","KOPOL","KRGYO","KRSTL","KRONT","KUYAS","KBORU","KUTPO","LMKDC","LOGO","LKMNH","MAKIM","MAGEN","MAVI","MEDTR","MEKAG","MNDRS","MERCN","MEYSU","MPARK","MOBTL","MNDTR","EGEPO","NTGAZ","NETAS","OBAMS","OBASE","OFSYM","ONCSM","ORGE","OSTIM","OZRDN","OZYSR","PNLSN","PAGYO","PARSN","PASEU","PENGD","PENTA","PETKM","PETUN","PKART","PLTUR","POLHO","QUAGR","RNPOL","RODRG","RGYAS","RUBNS","SAFKR","SANEL","SNICA","SANKO","SAMAT","SARKYS","SAYAS","SEKUR","SELEC","SELVA","SRVGY","SILVR","SNGYO","SMRTG","SMART","SOKE","SUNTK","SURGY","SUWEN","TNZTP","TARKM","TKNSA","TDGYO","TUCLK","TUKAS","TUREX","MARBL","TMSN","TUPRS","ULAS","ULUSE","USAK","UCAYM","VAKKO","VANGD","VRGYO","VESBE","YATAS","YEOTK","YUNSA","ZEDUR","ZERGY"] # Burayı 223 hisseye tamamlayabilirsin

with st.sidebar:
    st.header("⚙️ Tarama Ayarları")
    # 46. satır hatasını burada kökten çözdük:
    periyot = st.selectbox("Tarama Periyodu", ["1h", "4h", "1d", "1w"])
    st.write(f"Toplam Takip Edilen: {len(hisse_listesi)} Hisse")

# Ana Ekran
col1, col2 = st.columns(2)
with col1:
    st.info(f"Şu an seçili periyot: **{periyot}**")
with col2:
    st.success(f"Tarama Listesi: **Özel 223 Hisse**")

# Tarama Butonu
if st.button("223 HİSSEYİ ŞİMDİ TARA", use_container_width=True):
    st.write("🔄 Liste taranıyor, indikatörler hesaplanıyor...")
    
    # Burası senin listeni filtreleyen kısımdır
    # Şimdilik sana nasıl görüneceğini göstermek için bir simülasyon yapıyorum:
    tarama_sonuclari = {
        "Hisse Adı": hisse_listesi[:10], # Listenin ilk 10 tanesini örnek gösterir
        "Sinyal": ["Pusu Kuruldu", "AL", "Bekle", "Pusu Kuruldu", "AL", "Sat", "Bekle", "AL", "Pusu", "AL"],
        "Güç": ["%85", "%70", "%40", "%92", "%65", "%20", "%50", "%75", "%88", "%80"]
    }
    
    df = pd.DataFrame(tarama_sonuclari)
    st.dataframe(df, use_container_width=True)
    st.balloons()
else:
    st.warning("Tarama başlatmak için yukarıdaki butona bas.")

st.markdown("---")
st.caption("Yiğit'e özel 223 hisselik takip paneli v1.0")
