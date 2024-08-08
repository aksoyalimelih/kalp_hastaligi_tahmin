from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Model ve scaler'ı yükle
model = joblib.load('heart.pkl')
scaler = joblib.load('heartscaler.pkl')

# Ana sayfayı tanımlayın
@app.route('/')
def index():
    return render_template('index.html')

# Tahmin yapacak API endpoint'ini tanımlayın
@app.route('/predict', methods=['POST'])
def predict():
    # Veriyi formdan alın
    data = {
        'yas': request.form['yas'],
        'cinsiyet': request.form['cinsiyet'],
        'gogus_agrisi_tipi': request.form['gogus_agrisi_tipi'],
        'dinlenme_kan_basinci': request.form['dinlenme_kan_basinci'],
        'kolestrol': request.form['kolestrol'],
        'aclik_kan_sekeri': request.form['aclik_kan_sekeri'],
        'istirahat_EKG_sonuclari': request.form['istirahat_EKG_sonuclari'],
        'maksimum_kalp_hizi': request.form['maksimum_kalp_hizi'],
        'egzersiz_induklu_anjina': request.form['egzersiz_induklu_anjina'],
        'st_depresyonu': request.form['st_depresyonu'],
        'egzersiz_ST_egrisi': request.form['egzersiz_ST_egrisi'],
        'ana_damarlar': request.form['ana_damarlar'],
        'talasemi': request.form['talasemi']
    }
    
    # Veriyi DataFrame'e dönüştürün
    df = pd.DataFrame([data])
    
    # Özellikler
    feature_columns = [
        'yas', 'cinsiyet', 'gogus_agrisi_tipi', 'dinlenme_kan_basinci', 'kolestrol',
        'aclik_kan_sekeri', 'istirahat_EKG_sonuclari', 'maksimum_kalp_hizi',
        'egzersiz_induklu_anjina', 'st_depresyonu', 'egzersiz_ST_egrisi',
        'ana_damarlar', 'talasemi'
    ]
    
    # Test verisini hazırlayın
    X_test = df[feature_columns]
    X_test_scaled = scaler.transform(X_test)
    
    # Tahmin yapın
    prediction = model.predict(X_test_scaled)
    
    # Tahmin sonucu
    result = 'Kalp hastası' if prediction[0] == 1 else 'Kalp hastası değil'
    
    # Sonucu sayfada göstermek için render_template kullanın
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)