import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import poisson
import os

st.set_page_config(page_title="Korner Tahmin Motoru PRO", layout="centered")
st.title("⚽ Korner Beklentisi (Poisson) PRO")
st.markdown("Bu model; takımların **son 5 maçlık güncel formunu** ve **şut kalitesini** baz alarak çalışır.")

ligler = {
    "İngiltere Premier Lig": "E0",
    "İspanya La Liga": "SP1",
    "İtalya Serie A": "I1",
    "Almanya Bundesliga": "D1",
    "Fransa Ligue 1": "F1",
    "Türkiye Süper Lig": "T1"
}

st.sidebar.header("1. Lig Seçimi")
secilen_lig_adi = st.sidebar.selectbox("Ligi Belirleyin:", list(ligler.keys()))
kod = ligler[secilen_lig_adi]

# İnternet URL'leri iptal edildi, direkt senin klasöründeki sağlam CSV'ler okunacak
dosya_gecen = f"{kod}_2526.csv"
dosya_bu = f"{kod}_2627.csv"

@st.cache_data
def veri_yukle_ve_egit(d1, d2):
    df_gecen = pd.read_csv(d1)
    df_bu = pd.read_csv(d2)
    df = pd.concat([df_gecen, df_bu], ignore_index=True)
    
    # Türkiye ligi gibi isabetli şut (HST) tutmayan ligler için çökmeyi engelleyen sistem
    if 'HST' in df.columns and 'AST' in df.columns:
        gerekli = df[['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HS', 'AS', 'HST', 'AST']].dropna()
        form_etiket = "İsabetli Şut"
        hesap_formulu = "Korner ~ Takim + Rakip + Saha + Sut + IsabetliSut"
        ev = pd.DataFrame({'Takim': gerekli['HomeTeam'], 'Rakip': gerekli['AwayTeam'], 'Saha': 'Ev', 'Korner': gerekli['HC'], 'Sut': gerekli['HS'], 'IsabetliSut': gerekli['HST']})
        dep = pd.DataFrame({'Takim': gerekli['AwayTeam'], 'Rakip': gerekli['HomeTeam'], 'Saha': 'Dep', 'Korner': gerekli['AC'], 'Sut': gerekli['AS'], 'IsabetliSut': gerekli['AST']})
    else:
        gerekli = df[['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HS', 'AS']].dropna()
        form_etiket = "Toplam Şut"
        hesap_formulu = "Korner ~ Takim + Rakip + Saha + Sut"
        ev = pd.DataFrame({'Takim': gerekli['HomeTeam'], 'Rakip': gerekli['AwayTeam'], 'Saha': 'Ev', 'Korner': gerekli['HC'], 'Sut': gerekli['HS']})
        dep = pd.DataFrame({'Takim': gerekli['AwayTeam'], 'Rakip': gerekli['HomeTeam'], 'Saha': 'Dep', 'Korner': gerekli['AC'], 'Sut': gerekli['AS']})

    matris = pd.concat([ev, dep], ignore_index=True)
    egitilmis_model = smf.poisson(hesap_formulu, data=matris).fit(disp=False)
    takimlar = sorted(gerekli['HomeTeam'].unique())
    
    return egitilmis_model, takimlar, gerekli, form_etiket

# Dosyaların klasörde olup olmadığını kontrol et, hata verirse çökmek yerine uyarı bas
if not os.path.exists(dosya_gecen) or not os.path.exists(dosya_bu):
    st.error(f"⚠️ Klasörde {dosya_gecen} veya {dosya_bu} dosyası bulunamadı! Lütfen CSV dosyalarının projeyle aynı klasörde olduğundan emin ol.")
else:
    try:
        model, gecerli_takimlar, ham_veri, form_turu = veri_yukle_ve_egit(dosya_gecen, dosya_bu)
        st.success(f"✅ {secilen_lig_adi} verileri başarıyla yüklendi.")
        st.divider()

        def son_5_mac_formu(df, takim):
            takim_maclari = df[(df['HomeTeam'] == takim) | (df['AwayTeam'] == takim)].tail(5)
            if len(takim_maclari) == 0:
                return 0, 0
            
            sut_toplam, isabet_toplam = 0, 0
            for _, row in takim_maclari.iterrows():
                if row['HomeTeam'] == takim:
                    sut_toplam += row['HS']
                    isabet_toplam += row.get('HST', 0)
                else:
                    sut_toplam += row['AS']
                    isabet_toplam += row.get('AST', 0)
                    
            return sut_toplam / len(takim_maclari), isabet_toplam / len(takim_maclari)

        col1, col2 = st.columns(2)
        with col1:
            h_takim = st.selectbox("🏠 Ev Sahibi (H):", gecerli_takimlar)
        with col2:
            a_takim = st.selectbox("✈️ Deplasman (A):", reversed(gecerli_takimlar))

        if st.button("🚀 Olasılıkları Hesapla"):
            if h_takim == a_takim:
                st.warning("Ev sahibi ve deplasman takımı aynı olamaz!")
            else:
                h_sut, h_isabet = son_5_mac_formu(ham_veri, h_takim)
                a_sut, a_isabet = son_5_mac_formu(ham_veri, a_takim)
                
                if form_turu == "İsabetli Şut":
                    ornek_ev = pd.DataFrame({'Takim': [h_takim], 'Rakip': [a_takim], 'Saha': ['Ev'], 'Sut': [h_sut], 'IsabetliSut': [h_isabet]})
                    ornek_dep = pd.DataFrame({'Takim': [a_takim], 'Rakip': [h_takim], 'Saha': ['Dep'], 'Sut': [a_sut], 'IsabetliSut': [a_isabet]})
                    gosterilecek_h = h_isabet
                    gosterilecek_a = a_isabet
                else:
                    ornek_ev = pd.DataFrame({'Takim': [h_takim], 'Rakip': [a_takim], 'Saha': ['Ev'], 'Sut': [h_sut]})
                    ornek_dep = pd.DataFrame({'Takim': [a_takim], 'Rakip': [h_takim], 'Saha': ['Dep'], 'Sut': [a_sut]})
