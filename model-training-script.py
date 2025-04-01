import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

# Veri Okuma
df = pd.read_csv("Salary.csv")

# Veri Hazırlığı
X = df[['YearsExperience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polinom Özellikleri Oluşturma
poly_features = PolynomialFeatures(degree=2)
X_poly_train = poly_features.fit_transform(X_train)

# Model Eğitimi
lr_poly = LinearRegression()
lr_poly.fit(X_poly_train, y_train)

# Modeli ve Polinom Dönüşüm Nesnesini Kaydetme
joblib.dump(lr_poly, 'salary_prediction_model.pkl')
joblib.dump(poly_features, 'poly_features.pkl')

print("Model ve dönüşüm nesnesi başarıyla kaydedildi!")