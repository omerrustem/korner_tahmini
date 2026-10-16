import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import poisson

st.set_page_config(page_title="Korner Tahmin Motoru PRO", layout="centered")
st.title("⚽ Korner Beklentisi (Poisson) PRO")
st.markdown("Bu model; takımların **son 5 maçlık güncel formunu** ve **isabetli şut** (kalite) istatistiklerini baz alarak canlı veri üzerinden çalışır.")

# Lig URL Veritabanı
ligler = {
    "İngiltere Premier Lig": "E0",
    "Türkiye Süper Lig": "T1",
    "İspanya La Liga": "SP1",
    "İtalya Serie A": "I1",
    "Almanya Bundesliga": "D1",
    "Fransa Ligue 1": "F1"
}

st.sidebar.header("1. Lig Seçimi")
secilen_lig_adi = st.sidebar.selectbox("Ligi Belirleyin:", list(ligler.keys()))
kod = ligler[secilen_lig_adi]

url_gecen = f"https://www.football-data.co.uk/mmz4281/2526/{kod}.csv"
url_bu = f"https://www.football-data.co.uk/mmz4281/2627/{kod}.csv"

# Veriyi canlı URL'den çek ve hız için 1 saat (3600 sn) cache'de tut
@st.cache_data(ttl=3600)
def veri_cek_ve_egit(url1, url2):
    df_gecen = pd.read_csv(url1)
    df_bu = pd.read_csv(url2)
    df = pd.concat([df_gecen, df_bu], ignore_index=True)
    
    # İsabetli şutları da (HST, AST) dahil ediyoruz
    gerekli = df[['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HS', 'AS', 'HST', 'AST']].dropna()
    
    ev = pd.DataFrame({'Takim': gerekli['HomeTeam'], 'Rakip': gerekli['AwayTeam'], 'Saha': 'Ev', 'Korner': gerekli['HC'], 'Sut': gerekli['HS'], 'IsabetliSut': gerekli['HST']})
    dep = pd.DataFrame({'Takim': gerekli['AwayTeam'], 'Rakip': gerekli['HomeTeam'], 'Saha': 'Dep', 'Korner': gerekli['AC'], 'Sut': gerekli['AS'], 'IsabetliSut': gerekli['AST']})
    matris = pd.concat([ev, dep], ignore_index=True)
    
    # Model artık hem şut hem isabetli şutu dikkate alıyor
    egitilmis_model = smf.poisson("Korner ~ Takim + Rakip + Saha + Sut + IsabetliSut", data=matris).fit(disp=False)
    takimlar = sorted(gerekli['HomeTeam'].unique())
    return egitilmis_model, takimlar, gerekli

try:
    model, gecerli_takimlar, ham_veri = veri_cek_ve_egit(url_gecen, url_bu)
    st.success(f"✅ {secilen_lig_adi} güncel verileri canlı olarak çekildi ve matris eğitildi.")
    st.divider()

    # Son 5 Maç Formunu Hesaplayan Fonksiyon
    def son_5_mac_formu(df, takim):
        # Takımın oynadığı son 5 maçı kronolojik olarak bul
        takim_maclari = df[(df['HomeTeam'] == takim) | (df['AwayTeam'] == takim)].tail(5)
        
        # Eğer takımın verisi yoksa (örn. yeni çıkmış ve maç yapmamışsa) hata vermesin
        if len(takim_maclari) == 0:
            return 0, 0
            
        sut_toplam, isabet_toplam = 0, 0
        for _, row in takim_maclari.iterrows():
            if row['HomeTeam'] == takim:
                sut_toplam += row['HS']
                isabet_toplam += row['HST']
            else:
                sut_toplam += row['AS']
                isabet_toplam += row['AST']
                
        return sut_toplam / len(takim_maclari), isabet_toplam / len(takim_maclari)

    col1, col2 = st.columns(2)
    with col1:
        h_takim = st.selectbox("🏠 Ev Sahibi (H):", gecerli_takimlar)
    with col2:
        a_takim = st.selectbox("✈️ Deplasman (A):", reversed(gecerli_takimlar))

    if st.button("🚀 Canlı Olasılıkları Hesapla"):
        if h_takim == a_takim:
            st.warning("Ev sahibi ve deplasman takımı aynı olamaz!")
        else:
            # Takımların son 5 maçlık güncel formunu çekiyoruz
            h_sut, h_isabet = son_5_mac_formu(ham_veri, h_takim)
            a_sut, a_isabet = son_5_mac_formu(ham_veri, a_takim)

            ornek_ev = pd.DataFrame({'Takim': [h_takim], 'Rakip': [a_takim], 'Saha': ['Ev'], 'Sut': [h_sut], 'IsabetliSut': [h_isabet]})
            ornek_dep = pd.DataFrame({'Takim': [a_takim], 'Rakip': [h_takim], 'Saha': ['Dep'], 'Sut': [a_sut], 'IsabetliSut': [a_isabet]})

            beklenen_ev = model.predict(ornek_ev)[0]
            beklenen_dep = model.predict(ornek_dep)[0]
            beklenen_toplam = beklenen_ev + beklenen_dep

            st.subheader("📊 Beklenen Değerler (λ)")
            m1, m2, m3 = st.columns(3)
            # Metriklerde takımların isabetli şut formları doğru şekilde ayrıldı
            m1.metric(label=f"{h_takim} Korner", value=f"{beklenen_ev:.2f}", delta=f"Son 5 Maç: {h_isabet:.1f} İsabetli Şut")
            m2.metric(label=f"{a_takim} Korner", value=f"{beklenen_dep:.2f}", delta=f"Son 5 Maç: {a_isabet:.1f} İsabetli Şut", delta_color="off")
            m3.metric(label="Toplam Korner", value=f"{beklenen_toplam:.2f}")

            st.divider()
            # Başlıktan PMF kelimesi çıkarıldı
            st.subheader("📈 Toplam Korner Olasılık Dağılımı")
            
            # X ekseninin düzgün sıralanması için metin (string) yerine sayısal (integer) index kullanıldı
            df_chart = pd.DataFrame({
                "Korner Sayısı": range(5, 16),
                "İhtimal (%)": [poisson.pmf(k, beklenen_toplam) * 100 for k in range(5, 16)]
            }).set_index("Korner Sayısı")
            
            st.bar_chart(df_chart)

except Exception as e:
    st.error(f"Veri çekilirken bir hata oluştu. Sezon başı verisi eksik olabilir. Detay: {e}")
