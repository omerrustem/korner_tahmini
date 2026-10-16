import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import poisson
import urllib.request
import io
import ssl

# 1. KESİN ÇÖZÜM: Mac ve Python'un SSL güvenlik duvarını küresel olarak esnetiyoruz
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="Korner Tahmin Motoru PRO", layout="centered")
st.title("⚽ Korner Beklentisi (Poisson) PRO")
st.markdown("Bu model; takımların **son 5 maçlık güncel formunu** ve **şut kalitesini** baz alarak canlı veri üzerinden çalışır.")

# Lig URL Veritabanı
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

url_gecen = f"https://www.football-data.co.uk/mmz4281/2526/{kod}.csv"
url_bu = f"https://www.football-data.co.uk/mmz4281/2627/{kod}.csv"

@st.cache_data(ttl=3600)
def veri_cek_ve_egit(url1, url2):
    # 2. KESİN ÇÖZÜM: Siteye "Ben bir bot değilim, Mac tarayıcısıyım" diyoruz
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    
    # Veriyi direkt pandas ile değil, kimlik göstererek ham metin olarak indiriyoruz
    req1 = urllib.request.Request(url1, headers=headers)
    res1 = urllib.request.urlopen(req1).read().decode('utf-8')
    df_gecen = pd.read_csv(io.StringIO(res1))

    req2 = urllib.request.Request(url2, headers=headers)
    res2 = urllib.request.urlopen(req2).read().decode('utf-8')
    df_bu = pd.read_csv(io.StringIO(res2))
    
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

try:
    model, gecerli_takimlar, ham_veri, form_turu = veri_cek_ve_egit(url_gecen, url_bu)
    st.success(f"✅ {secilen_lig_adi} verileri başarıyla çekildi.")
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

    if st.button("🚀 Canlı Olasılıkları Hesapla"):
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
                gosterilecek_h = h_sut
                gosterilecek_a = a_sut

            beklenen_ev = model.predict(ornek_ev)[0]
            beklenen_dep = model.predict(ornek_dep)[0]
            beklenen_toplam = beklenen_ev + beklenen_dep

            st.subheader("📊 Beklenen Değerler (λ)")
            m1, m2, m3 = st.columns(3)
            m1.metric(label=f"{h_takim} Korner", value=f"{beklenen_ev:.2f}", delta=f"Son 5 Maç: {gosterilecek_h:.1f} {form_turu}")
            m2.metric(label=f"{a_takim} Korner", value=f"{beklenen_dep:.2f}", delta=f"Son 5 Maç: {gosterilecek_a:.1f} {form_turu}", delta_color="off")
            m3.metric(label="Toplam Korner", value=f"{beklenen_toplam:.2f}")

            st.divider()
            st.subheader("📈 Toplam Korner Olasılık Dağılımı")
            
            df_chart = pd.DataFrame({
                "Korner Sayısı": range(5, 16),
                "İhtimal (%)": [poisson.pmf(k, beklenen_toplam) * 100 for k in range(5, 16)]
            }).set_index("Korner Sayısı")
            
            st.bar_chart(df_chart)

except Exception as e:
    st.error(f"Veri çekilirken bir hata oluştu. Detay: {e}")
