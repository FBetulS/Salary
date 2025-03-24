import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

# Streamlit sayfası yapılandırması
st.set_page_config(page_title="Maaş Tahmin Aracı", page_icon="💰")

# Başlık ve açıklama
st.title("🚀 Maaş Tahmin Modeli")
st.markdown("Bu uygulama, çalışma deneyiminize göre tahmini maaşınızı hesaplar.")

# Polinom modelini yükleyen fonksiyon
@st.cache_resource
def load_model():
    # Eğitilmiş modeli ve dönüşüm nesnesini yükle
    model = joblib.load('salary_prediction_model.pkl')
    poly_features = joblib.load('poly_features.pkl')
    return model, poly_features

# Model ve polinom dönüşüm nesnesini yükle
model, poly_features = load_model()

# Yan çubuk
st.sidebar.header("Model Bilgileri")
st.sidebar.info("""
- Model: Polinom Regresyon (Derece 2)
- Performans Metrikleri:
  * R2 Skoru: 0.918
  * Ortalama Mutlak Hata: 5,879
""")

# Kullanıcı girişi
st.header("Tahmin Parametreleri")
years_experience = st.slider(
    "Çalışma Deneyim Yılı", 
    min_value=0.0, 
    max_value=20.0, 
    step=0.5, 
    value=2.0
)

# Tahmin butonu
if st.button("Maaş Tahminini Hesapla"):
    # Girişi hazırla
    X_input = np.array([[years_experience]])
    X_poly = poly_features.transform(X_input)
    
    # Tahmin yap
    predicted_salary = model.predict(X_poly)[0]
    
    # Sonuçları göster
    st.success(f"🎯 Tahmini Maaş: {predicted_salary:,.2f} TL")
    
    # Güven aralığı ve açıklama
    st.info("""
    ### Tahmin Açıklaması
    - Bu tahmin, çalışma deneyiminize dayalı istatistiksel bir tahmindir.
    - Gerçek maaş, şirket, sektör ve bireysel performansa göre değişebilir.
    """)

# Alt bilgi
st.markdown("---")
st.markdown("📊 Model Geliştiren: Veri Bilimi Ekibi")
