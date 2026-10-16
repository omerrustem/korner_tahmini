import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import poisson
import os

# Sayfa Ayarları
st.set_page_config(page_title="Korner Tahmin Motoru", layout="centered")
st.title("⚽ Korner Beklentisi (Poisson) Modeli")

# Lig Veritabanı
ligler = {
    "İngiltere Premier Lig": "E0",
    "Türkiye Süper Lig": "T1",
    "İspanya La Liga": "SP1",
    "İtalya Serie A": "I1",
    "Almanya Bundesliga": "D1",
    "Fransa Ligue 1": "F1"
}

# Yan menüden lig seçimi
st.sidebar.header("1. Lig Seçimi")
secilen_lig_adi = st.sidebar.selectbox("Ligi Belirleyin:", list(ligler.keys()))
kod = ligler[secilen_lig_adi]

dosya_gecen = f"{kod}_2526.csv"
dosya_bu = f"{kod}_2627.csv"

# Dosya kontrolü ve Model Eğitimi
if not os.path.exists(dosya_gecen) or not os.path.exists(dosya_bu):
    st.error(f"Hata: {dosya_gecen} veya {dosya_bu} bulunamadı. Lütfen CSV dosyalarının klasörde olduğundan emin olun.")
else:
    # Veriyi yükle ve cache'e al (uygulama hızlansın diye)
    @st.cache_data
    def veri_yukle_ve_egit(kod_gecen, kod_bu):
        df_gecen = pd.read_csv(kod_gecen)
        df_bu = pd.read_csv(kod_bu)
        df = pd.concat([df_gecen, df_bu], ignore_index=True)
        
        gerekli = df[['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HS', 'AS']].dropna()
        ev = pd.DataFrame({'Takim': gerekli['HomeTeam'], 'Rakip': gerekli['AwayTeam'], 'Saha': 'Ev', 'Korner': gerekli['HC'], 'Sut': gerekli['HS']})
        dep = pd.DataFrame({'Takim': gerekli['AwayTeam'], 'Rakip': gerekli['HomeTeam'], 'Saha': 'Dep', 'Korner': gerekli['AC'], 'Sut': gerekli['AS']})
        matris = pd.concat([ev, dep], ignore_index=True)
        
        egitilmis_model = smf.poisson("Korner ~ Takim + Rakip + Saha + Sut", data=matris).fit(disp=False)
        takimlar = sorted(gerekli['HomeTeam'].unique())
        return egitilmis_model, takimlar, ev, dep

    # Modeli çalıştır
    model, gecerli_takimlar, ev_df, dep_df = veri_yukle_ve_egit(dosya_gecen, dosya_bu)

    st.success(f"✅ {secilen_lig_adi} veri seti yüklendi ve Poisson modeli eğitildi.")
    st.divider()

    # Takım Seçimleri (Dropdown)
    col1, col2 = st.columns(2)
    with col1:
        h_takim = st.selectbox("🏠 Ev Sahibi (H):", gecerli_takimlar)
    with col2:
        a_takim = st.selectbox("✈️ Deplasman (A):", reversed(gecerli_takimlar)) # Aynı takım denk gelmesin diye ters çevrildi

    if st.button("🚀 Olasılıkları Hesapla"):
        if h_takim == a_takim:
            st.warning("Ev sahibi ve deplasman takımı aynı olamaz!")
        else:
            # Şut beklentilerini hesapla
            h_ortalama_sut = ev_df[ev_df['Takim'] == h_takim]['Sut'].mean()
            a_ortalama_sut = dep_df[dep_df['Takim'] == a_takim]['Sut'].mean()

            # Lambda değerlerini bul
            ornek_ev = pd.DataFrame({'Takim': [h_takim], 'Rakip': [a_takim], 'Saha': ['Ev'], 'Sut': [h_ortalama_sut]})
            ornek_dep = pd.DataFrame({'Takim': [a_takim], 'Rakip': [h_takim], 'Saha': ['Dep'], 'Sut': [a_ortalama_sut]})

            beklenen_ev = model.predict(ornek_ev)[0]
            beklenen_dep = model.predict(ornek_dep)[0]
            beklenen_toplam = beklenen_ev + beklenen_dep

            # Metrikleri Ekrana Bas
            st.subheader(f"📊 Beklenen Değerler (λ)")
            m1, m2, m3 = st.columns(3)
            m1.metric(label=f"{h_takim} Korner", value=f"{beklenen_ev:.2f}", delta=f"{h_ortalama_sut:.1f} Şut Baskısı")
            m2.metric(label=f"{a_takim} Korner", value=f"{beklenen_dep:.2f}", delta=f"{a_ortalama_sut:.1f} Şut Baskısı", delta_color="off")
            m3.metric(label="Toplam Korner (Beklenen)", value=f"{beklenen_toplam:.2f}")

            # PMF Dağılım Grafiği (Bar Chart)
            st.divider()
            st.subheader("📈 Toplam Korner Olasılık Dağılımı (PMF)")
            
            pmf_verileri = {}
            for k in range(5, 16):
                olasilik = poisson.pmf(k, beklenen_toplam) * 100
                pmf_verileri[f"{k} Korner"] = olasilik

            # Grafiği çizdir (Streamlit'in kendi şık bar chart bileşeni)
            chart_df = pd.DataFrame(list(pmf_verileri.values()), index=list(pmf_verileri.keys()), columns=["İhtimal (%)"])
            st.bar_chart(chart_df)