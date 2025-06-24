# Makine ve Derin Öğrenme ile Meme Kanseri Sağkalım Tahmini

Bu proje, **meme kanseri hastalarının sağkalım durumunu** makine öğrenmesi ve derin öğrenme modelleriyle tahmin etmeyi amaçlamaktadır. Kullanılan veri seti, [SEER Programı (Surveillance, Epidemiology, and End Results)](https://seer.cancer.gov/) tarafından sunulmuş ve 2006–2010 yılları arasında teşhis edilen hastalara ait klinik ve demografik bilgileri içermektedir.

> **Ders Projesi**  
> *Yapay Zekaya Giriş (Introduction to Artificial Intelligence)*

---

## 📌 Proje Amaçları

- SEER veri seti üzerinde Keşifsel Veri Analizi (EDA) yapmak
- Eksik ve aykırı değerleri temizlemek
- En az **5 makine öğrenmesi modeli** ve **3 derin öğrenme modeli** uygulamak
- Modelleri **Accuracy**, **F1 Skoru**, **Precision**, **Recall**, **AUC (ROC)** ve **Cohen's Kappa** gibi metriklerle değerlendirmek
- Sağkalıma en çok etki eden öznitelikleri belirlemek
- En iyi modeli kaydedip yeni örneklerde tahmin yapabilen bir sistem kurmak

---

## 🗂️ Veri Seti Özeti

- 📁 Kaynak: SEER Meme Kanseri Veri Seti (2006–2010)
- 🧪 Gözlem Sayısı: 4024 hasta
- 🔬 Özellikler:
  - Yaş, Irk, Medeni Durum
  - Tümör Evreleri (T, N, 6. AJCC Evresi)
  - Tümör Boyutu, Derece (Grade)
  - Hormon Reseptör Durumu (Estrojen, Progesteron)
  - İncelenen ve pozitif çıkan lenf nodları
  - Sağkalım Süresi (ay)
  - Sağkalım Durumu (Alive / Dead)

---

## 🧠 Uygulanan Modeller

### 🔹 Makine Öğrenmesi
- Lojistik Regresyon
- Random Forest
- K-En Yakın Komşu (KNN)
- Destek Vektör Makineleri (SVM)
- Karar Ağaçları
- XGBoost

### 🔹 Derin Öğrenme
- Basit Yapay Sinir Ağı (FNN)
- Çok Katmanlı Algılayıcı (MLP)
- Derin Sinir Ağı (DNN)
- Konvolüsyonel Sinir Ağı (CNN)

Her modelde:
- Hiperparametre ayarlamaları yapılmıştır (ör. GridSearchCV)
- Özellik seçimi uygulanmıştır (ör. RFE)
- Değerlendirme metrikleri ve ROC eğrileri analiz edilmiştir

---

## 📊 Özellik Mühendisliği

- Kategorik değişkenler sayısal forma dönüştürüldü
- Tümör evresi ve derecesine göre yeni bir `Risk Group` özniteliği oluşturuldu
- Recursive Feature Elimination (RFE) ile önemli özellikler belirlendi

---

## 📈 Değerlendirme Metrikleri

| Metrik             | Açıklama                                     |
|--------------------|----------------------------------------------|
| Accuracy           | Genel doğruluk oranı                         |
| Precision          | Doğru pozitif / Tüm pozitif tahminler        |
| Recall (Duyarlılık)| Doğru pozitif / Gerçek pozitifler            |
| F1 Skoru           | Precision ve Recall'un harmonik ortalaması   |
| ROC-AUC            | ROC eğrisi altında kalan alan                 |
| Cohen’s Kappa      | Tahmin ve gerçek sınıflar arası uyum         |

---

## 🏆 En İyi Modelin Seçilmesi

Tüm modeller değerlendirildikten sonra, en yüksek **Accuracy** ve **F1 Skoru** değerine sahip model seçilip `best_model.pkl` dosyasına kaydedilir.

---

## 🔮 Yeni Hasta Üzerinden Tahmin

Proje içerisinde aşağıdaki işlemleri gerçekleştiren bir yapı bulunmaktadır:

- Yeni hasta verilerini kullanıcıdan alır
- Gerekli ön işlem ve ölçeklendirme adımlarını uygular
- Eğitilmiş en iyi modelle tahmin yapar

### 🧪 Örnek Kullanım

```python
# 1. Kullanıcıdan yeni hasta bilgilerini al
new_sample = get_input_for_features()

# 2. Veriyi ölçeklendir
new_sample_scaled = scaler.transform(new_sample)

# 3. Tahmin yap
prediction = predict_new_samples(best_model, new_sample_scaled)

# 4. Sonucu yazdır
print("✅ Hasta büyük olasılıkla YAŞAYACAK." if prediction[0] == 1 else "❌ Hasta büyük olasılıkla YAŞAMAYACAK.")
