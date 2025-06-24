import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.utils import class_weight
from keras.layers import LeakyReLU, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import joblib

# Uyarıları gizle
warnings.filterwarnings('ignore')

# Veri setini yükle
vc = pd.read_csv(r'C:\Users\...\Desktop\SEER Breast Cancer Dataset .csv')

# Veri seti bilgileri
print("Sütunlar:", vc.columns)
print(vc.info())
print("Boş değerler:", vc.isnull().sum)
print('Veri Setindeki Satır Sayısı: ', vc.shape[0])
print('Veri Setindeki Sütun Sayısı: ', vc.shape[1])
print('Eksik Değerler:\n', vc.isnull().sum())
print(vc.describe())

plt.figure(figsize=(12, 6))
sns.heatmap(vc.isnull(), cbar=False, cmap='viridis')
plt.title('Eksik Değerlerin Dağılımı')
plt.show()


# Sütun adlarındaki boşlukları temizle
vc.columns = vc.columns.str.strip()
vc = vc.drop(columns=['Unnamed: 3'])

cat_feats = ['Race ','Marital Status','T Stage ','N Stage','6th Stage','Grade','A Stage','Estrogen Status','Progesterone Status']
num_feats = ['Age','Tumor Size','Regional Node Examined','Reginol Node Positive']

plt.figure(figsize=(10,20))
for i in range(len(num_feats)):
    plt.subplot(4,2,2*i+1)
    sns.boxplot(data=vc,y=num_feats[i])
    plt.subplot(4,2,2*i+2)
    sns.histplot(vc, x=num_feats[i], kde=True)
plt.show()

# Sürekli değişkenler için dağılım grafikleri
plt.figure(figsize=(12, 8))
sns.histplot(vc['Age'], bins=30, kde=True, color='blue')
plt.title('Yaş Dağılımı')
plt.show()

# Başka bir kategorik sütun analizi
plt.figure(figsize=(12, 8))
sns.countplot(data=vc, x='Race')
plt.title('Irk Dağılımı')
plt.xticks(rotation=45)
plt.show()

# Örnek olarak, 'Marital Status' analiz edelim
plt.figure(figsize=(12, 8))
sns.countplot(data=vc, x='Marital Status')
plt.title('Medeni Duruma Göre Dağılım')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=vc, x='T Stage', palette='viridis')
plt.title('T Evresi Dağılımı')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=vc, x='N Stage', palette='viridis')
plt.title('N Evresi Dağılımı')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(data=vc, x='6th Stage', palette='viridis')
plt.title('6. Evre Dağılımı')
plt.xticks(rotation=45)
plt.show()


# Kategorik değişkenler arasında ilişki analizi
plt.figure(figsize=(12, 8))
sns.countplot(x='Grade', data=vc)
plt.title('Derece Dağılımı')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=vc, x='A Stage', palette='viridis')
plt.title('A Evresi Dağılımı')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 8))
sns.histplot(vc['Tumor Size'], bins=30, kde=True, color='green')
plt.title('Tümör Boyutu Dağılımı')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=vc, x='Estrogen Status', palette='pastel')
plt.title('Östrojen Durumu Dağılımı')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=vc, x='Progesterone Status', palette='pastel')
plt.title('Progesteron Durumu Dağılımı')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(vc['Regional Node Examined'], bins=20, kde=True, color='purple')
plt.title('İncelenen Bölgesel Lenf Düğümü Dağılımı')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(vc['Reginol Node Positive'], bins=20, kde=True, color='orange')
plt.title('Pozitif Bölgesel Lenf Düğümü Dağılımı')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(vc['Survival Months'], bins=30, kde=True, color='red')
plt.title('Sağkalım Ayları Dağılımı')
plt.show()


# Sınıf dağılımını görselleştirme
plt.figure(figsize=(8, 6))
sns.countplot(data=vc, x='Status')
plt.title('Status Sınıf Dağılımı')
plt.xticks(rotation=45)
plt.show()

# Sayısal sütunların seçilmesi
numeric_cols = vc.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = vc[numeric_cols].corr()


# Korelasyon matrisini görselleştirin
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasyon Matrisi')
plt.show()



# Encoding "Grade"
Vmap = {
    "Well differentiated; Grade I": 0,
    "Moderately differentiated; Grade II": 1,
    "Poorly differentiated; Grade III": 2,
    "Undifferentiated; anaplastic; Grade IV": 3
}
vc['Grade'] = vc['Grade'].replace(Vmap)

# Encoding "6th Stage"
Vmap = {
    "IIA": 0,
    "IIB": 1,
    "IIIA": 2,
    "IIIB": 3,
    "IIIC": 4
}
vc['6th Stage'] = vc['6th Stage'].replace(Vmap)

# Encoding "N Stage"
Vmap = {
    "N1": 0,
    "N2": 1,
    "N3": 2
}

vc['N Stage'] = vc['N Stage'].replace(Vmap)

# Encoding "T Stage"
Vmap = {
    "T1": 0,
    "T2": 1,
    "T3": 2,
    "T4": 3
}
vc['T Stage'] = vc['T Stage'].replace(Vmap)

# Encoding "Estrogen Status" and "Progesterone Status"
Vmap = {
    "Positive": 1,
    "Negative": 0,
    "Regional": 1,
    "Distant": 0
}
vc['Estrogen Status'] = vc['Estrogen Status'].replace(Vmap)
vc['Progesterone Status'] = vc['Progesterone Status'].replace(Vmap)

# Encoding "A Stage"
Vmap = {
    "Regional": 1,
    "Distant": 0
}
vc['A Stage'] = vc['A Stage'].replace(Vmap)

# Encoding "Race"
Vmap = {
    "White": 0,
    "Black": 1,
    "Other (American Indian/AK Native, Asian/Pacific Islander)": 2
}
vc['Race'] = vc['Race'].replace(Vmap)

# Encoding "Marital Status"
Vmap = {
    "Married (including common law)": 0,
    "Single (never married)": 1,
    "Divorced": 2,
    "Widowed": 3,
    "Separated": 4
}
vc['Marital Status'] = vc['Marital Status'].replace(Vmap)

# Encoding "Status"
Vmap = {
    "Alive": 1,
    "Dead": 0
}
vc['Status'] = vc['Status'].replace(Vmap)

vc.head()

# Sayaçlar oluştur
low_risk_count = 0
medium_risk_count = 0
medium_risk_default_count = 0
high_risk_count = 0

# Risk grubu oluşturma fonksiyonu
def risk_group(row):
    global low_risk_count, medium_risk_count, medium_risk_default_count, high_risk_count

    # Sayısal değerler kullanılıyor, bu yüzden strip() fonksiyonuna gerek yok
    t_stage = row['T Stage']  # Sayısal değer kullanılıyor
    n_stage = row['N Stage']
    sixth_stage = row['6th Stage']
    grade = row['Grade']

    # Düşük risk grubu
    if t_stage in [0, 1] and n_stage in [0, 1] and sixth_stage in [0, 1] and grade in [0, 1]:
        low_risk_count += 1
        return 0

    # Orta risk grubu
    elif t_stage in [1, 2] and n_stage in [1, 2] and sixth_stage in [2, 3] and grade in [1, 2]:
        medium_risk_count += 1
        return 1

    # Yüksek risk grubu
    elif t_stage in [2, 3] and n_stage == 2 and sixth_stage in [3, 4] and grade in [2, 3]:
        high_risk_count += 1
        return 2

    else:
        medium_risk_default_count += 1
        return 1  # Orta risk, eğer yukarıdaki koşulların hiçbiri uymazsa

# Risk grubu sütunu oluştur
vc['Risk Group'] = vc.apply(risk_group, axis=1)
print("Sütunlar:", vc.columns)

# Sonuçları gösterme
print("Low Risk:", low_risk_count)
print("Medium Risk:", medium_risk_count)
print("Medium Risk by Default:", medium_risk_default_count)
print("High Risk:", high_risk_count)


# Sonuçları gösterme
print(vc['Risk Group'].value_counts())


# Sınıf dağılımını görselleştirme
plt.figure(figsize=(8, 6))
sns.countplot(data=vc, x='Risk Group')
plt.title('Risk Group Dağılımı')
plt.show()



# Sınıf dağılımını sayma
class_distribution = vc['Status'].value_counts()
print('Sınıf Dağılımı:\n', class_distribution)


# Veri setini X ve y olarak ayır
# Burada 'Status' hedef değişkenimiz olarak alınmıştır.
X = vc.drop(['Status'], axis=1)  # Geriye kalan değişkenler
y = vc['Status']  # Hedef değişken


# Veriyi eğitim ve test setlerine ayır
test_size_value = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, random_state=19)


# Veriyi normalleştir
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# K-NN modelini oluştur
knn = KNeighborsClassifier()
params = {'n_neighbors': [i for i in range(1, 77, 2)]}
modelknn = GridSearchCV(knn, params, cv=10)

# Modeli eğit
modelknn.fit(X_train, y_train)
print('En iyi parametreler:', modelknn.best_params_)

# Tahmin yap
predictknn = modelknn.predict(X_test)

# Doğruluk oranını hesapla
print('Eğitime ayrılan yüzde:%', test_size_value * 100)
print('Doğruluk oranı:', accuracy_score(y_test, predictknn))
print('K-NN algoritmasını kullanarak yapılan modelin tahmin oranı : %',
      round(accuracy_score(y_test, predictknn), 5) * 100)

# Karşılaştırma matrisini göster
cnf_matrixknn = confusion_matrix(y_test, predictknn)
class_names = vc['Status'].unique()  # 'Status' sınıflarını al

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Sıcaklık haritası
sns.heatmap(pd.DataFrame(cnf_matrixknn), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Karşılaştırma Matrisi', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.show()


# Kesinlik, Duyarlılık ve F1 Skorunu hesapla
reportknn = classification_report(y_test, predictknn, target_names=np.unique(y), output_dict=True)
print('Kesinlik, Duyarlılık ve F1 Skoru:\n', reportknn)

# Raporu daha okunabilir hale getirelim
report_df = pd.DataFrame(reportknn).transpose()
print(report_df)


# Cohen's Kappa Değeri
kappaknn = cohen_kappa_score(y_test, predictknn)
print("Cohen's Kappa Değeri:", kappaknn)

# ROC Eğrisi ve AUC Analizi (Sınıf başına)

# Pozitif sınıfa ait olasılıkları alın
y_scoreknn = modelknn.predict_proba(X_test)[:, 1]  # Pozitif sınıf için olasılıklar

# ROC eğrisi ve AUC hesaplama
fpr, tpr, _ = roc_curve(y_test, y_scoreknn)  # Yanlış pozitif oranı (FPR), Doğru pozitif oranı (TPR)
roc_aucknn = auc(fpr, tpr)  # Eğri altındaki alan (AUC)

# ROC eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_aucknn:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Rastgele tahmin
plt.title('K-NN Modeli için ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# Genel Özellikleri Yazdır
print("Accuracy:", accuracy_score(y_test, predictknn))
print("F1-Skoru:", f1_score(y_test, predictknn, average='weighted'))
print("Precision:", precision_score(y_test, predictknn, average='weighted'))
print("Sensitivity/Recall:", recall_score(y_test, predictknn, average='weighted'))



modelRF = RandomForestClassifier(class_weight={0: 3, 1: 1},n_jobs=-1)

# En iyi sonuç veren özellik seçimini arama
for n_features in range(1, len(X.columns) + 1):
    rfe = RFE(modelRF, n_features_to_select=n_features)
    scores = cross_val_score(rfe, X_train, y_train, cv=5)
    print(f"Özellik Sayısı: {n_features}, Ortalama Doğruluk: {scores.mean()}")

# RFE uygulama
rfe = RFE(modelRF, n_features_to_select=11)  # En önemli 11 özelliği seçelim
rfe = rfe.fit(X_train, y_train)

# Seçilen özellikler
selected_features = X.columns[rfe.support_]
print("Seçilen Özellikler:", selected_features)

X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

# Random Forest modelini eğit
modelRF.fit(X_train_selected, y_train)

# Tahmin yap
predict_RF = modelRF.predict(X_test_selected)

# Doğruluk oranını hesapla
print('Eğitime ayrılan yüzde:%', test_size_value * 100)
print('Doğruluk oranı:', accuracy_score(y_test, predict_RF))
print('Random Forest algoritmasını kullanarak yapılan modelin tahmin oranı : %',
      round(accuracy_score(y_test, predict_RF), 5) * 100)

# Karşılaştırma matrisini göster
cnf_matrix_RF = confusion_matrix(y_test, predict_RF)
class_names = vc['Status'].unique()  # 'Status' sınıflarını al

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Sıcaklık haritası
sns.heatmap(pd.DataFrame(cnf_matrix_RF), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Karşılaştırma Matrisi (Random Forest)', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.show()

# Kesinlik, Duyarlılık ve F1 Skorunu hesapla
report_RF = classification_report(y_test, predict_RF, target_names=np.unique(y), output_dict=True)
print('Kesinlik, Duyarlılık ve F1 Skoru (Random Forest):\n', report_RF)

# Raporu daha okunabilir hale getirelim
report_df_RF = pd.DataFrame(report_RF).transpose()
print(report_df_RF)

# Cohen's Kappa Değeri
kapparf = cohen_kappa_score(y_test, predict_RF)
print("Cohen's Kappa Değeri:", kapparf)

# ROC Eğrisi ve AUC Analizi (Sınıf başına)

# Pozitif sınıfa ait olasılıkları alın
y_scorerf = modelRF.predict_proba(X_test_selected)[:, 1]  # Pozitif sınıf için olasılıklar

# ROC eğrisi ve AUC hesaplama
fpr, tpr, _ = roc_curve(y_test, y_scorerf)  # Yanlış pozitif oranı (FPR), Doğru pozitif oranı (TPR)
roc_aucrf = auc(fpr, tpr)  # Eğri altındaki alan (AUC)

# ROC eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_aucrf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Rastgele tahmin
plt.title('RF Modeli için ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# Genel Özellikleri Yazdır
print("Accuracy:", accuracy_score(y_test, predict_RF))
print("F1-Skoru:", f1_score(y_test, predict_RF, average='weighted'))
print("Precision:", precision_score(y_test, predict_RF, average='weighted'))
print("Sensitivity/Recall:", recall_score(y_test, predict_RF, average='weighted'))



#logic regresyon

# Logistic Regression modelini oluştur
modelLR = LogisticRegression(class_weight={0: 3, 1: 1})

rfe = RFE(modelLR, n_features_to_select=11)  # En önemli 11 özelliği seçelim
rfe = rfe.fit(X_train, y_train)

# Seçilen özellikler
selected_features = X.columns[rfe.support_]
print("Seçilen Özellikler:", selected_features)

X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'newton-cg'],
    'penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

# En iyi model ile tahmin yap
best_model = grid_search.best_estimator_
predict_LR = best_model.predict(X_test_selected)

# Doğruluk oranını hesapla
print('Eğitime ayrılan yüzde:%', test_size_value * 100)
print('Doğruluk oranı:', accuracy_score(y_test, predict_LR))
print('Logistic Regression algoritmasını kullanarak yapılan modelin tahmin oranı : %',
      round(accuracy_score(y_test, predict_LR), 5) * 100)

# Karşılaştırma matrisini göster
cnf_matrix_LR = confusion_matrix(y_test, predict_LR)
class_names = vc['Status'].unique()  # 'Status' sınıflarını al

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Sıcaklık haritası
sns.heatmap(pd.DataFrame(cnf_matrix_LR), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Karşılaştırma Matrisi (Logistic Regression)', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.show()

# Kesinlik, Duyarlılık ve F1 Skorunu hesapla
report_LR = classification_report(y_test, predict_LR, target_names=np.unique(y), output_dict=True)
print('Kesinlik, Duyarlılık ve F1 Skoru (Logistic Regression):\n', report_LR)

# Raporu daha okunabilir hale getirelim
report_df_LR = pd.DataFrame(report_LR).transpose()
print(report_df_LR)

# Cohen's Kappa Değeri
kappalr = cohen_kappa_score(y_test, predict_LR)
print("Cohen's Kappa Değeri:", kappalr)

# ROC Eğrisi ve AUC Analizi (Sınıf başına)

# Pozitif sınıfa ait olasılıkları alın
y_scorelr = best_model.predict_proba(X_test_selected)[:, 1]  # Pozitif sınıf için olasılıklar

# ROC eğrisi ve AUC hesaplama
fpr, tpr, _ = roc_curve(y_test, y_scorelr)  # Yanlış pozitif oranı (FPR), Doğru pozitif oranı (TPR)
roc_auclr = auc(fpr, tpr)  # Eğri altındaki alan (AUC)

# ROC eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_auclr:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Rastgele tahmin
plt.title('LR Modeli için ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# Genel Özellikleri Yazdır
print("Accuracy:", accuracy_score(y_test, predict_LR))
print("F1-Skoru:", f1_score(y_test, predict_LR, average='weighted'))
print("Precision:", precision_score(y_test, predict_LR, average='weighted'))
print("Sensitivity/Recall:", recall_score(y_test, predict_LR, average='weighted'))


# svm

# SVM modelini oluştur ve probability=True yap
svm = SVC(class_weight={0: 3, 1: 1}, probability=True)  # probability=True burada eklenmeli

params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# GridSearchCV kullanarak SVM modelini eğit
modelsvm = GridSearchCV(svm, params, cv=10, n_jobs=-1)

# Modeli eğit
modelsvm.fit(X_train, y_train)
print('En iyi parametreler:', modelsvm.best_params_)

# Tahmin yap
predictsvm = modelsvm.predict(X_test)

# Doğruluk oranını hesapla
print('Eğitime ayrılan yüzde:%', test_size_value * 100)
print('Doğruluk oranı:', accuracy_score(y_test, predictsvm))
print('SVM algoritmasını kullanarak yapılan modelin tahmin oranı : %',
      round(accuracy_score(y_test, predictsvm), 5) * 100)

# Karşılaştırma matrisini göster
cnf_matrixsvm = confusion_matrix(y_test, predictsvm)
class_names = vc['Status'].unique()  # 'Status' sınıflarını al

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Sıcaklık haritası
sns.heatmap(pd.DataFrame(cnf_matrixsvm), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Karşılaştırma Matrisi', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.show()

# Kesinlik, Duyarlılık ve F1 Skorunu hesapla
reportsvm = classification_report(y_test, predictsvm, target_names=np.unique(y), output_dict=True)
print('Kesinlik, Duyarlılık ve F1 Skoru:\n', reportsvm)

# Raporu daha okunabilir hale getirelim
report_df_svm = pd.DataFrame(reportsvm).transpose()
print(report_df_svm)

# Cohen's Kappa Değeri
kappasvm = cohen_kappa_score(y_test, predictsvm)
print("Cohen's Kappa Değeri:", kappasvm)

# ROC Eğrisi ve AUC Analizi (Sınıf başına)

# `best_estimator_` kullanarak en iyi modeli al ve pozitif sınıfa ait olasılıkları alın
y_scoresvm = modelsvm.best_estimator_.predict_proba(X_test)[:, 1]  # Pozitif sınıf için olasılıklar

# ROC eğrisi ve AUC hesaplama
fpr, tpr, _ = roc_curve(y_test, y_scoresvm)  # Yanlış pozitif oranı (FPR), Doğru pozitif oranı (TPR)
roc_aucsvm = auc(fpr, tpr)  # Eğri altındaki alan (AUC)

# ROC eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_aucsvm:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Rastgele tahmin
plt.title('SVM Modeli için ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Genel Özellikleri Yazdır
print("Accuracy:", accuracy_score(y_test, predictsvm))
print("F1-Skoru:", f1_score(y_test, predictsvm, average='weighted'))
print("Precision:", precision_score(y_test, predictsvm, average='weighted'))
print("Sensitivity/Recall:", recall_score(y_test, predictsvm, average='weighted'))


# Decision Tree modelini oluştur
dt = DecisionTreeClassifier()


for n_features in range(1, len(X.columns) + 1):
    rfe = RFE(dt, n_features_to_select=n_features)
    scores = cross_val_score(rfe, X_train, y_train, cv=5)
    print(f"Özellik Sayısı: {n_features}, Ortalama Doğruluk: {scores.mean()}")

rfe = RFE(dt, n_features_to_select=11)  # En önemli 11 özelliği seçelim
rfe = rfe.fit(X_train, y_train)

# Seçilen özellikler
selected_features = X.columns[rfe.support_]
print("Seçilen Özellikler:", selected_features)

X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)


# Hiperparametre aralığı tanımla (örneğin, 'max_depth' ve 'min_samples_split')
params = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

# GridSearchCV ile en iyi parametreleri bul
modeldt = GridSearchCV(dt, params, cv=10)
modeldt.fit(X_train_selected, y_train)

# En iyi parametreleri yazdır
print("En iyi parametreler:", modeldt.best_params_)

# Tahmin yap
predictdt = modeldt.predict(X_test_selected)

# Doğruluk oranını hesapla
print('Eğitime ayrılan yüzde:%', test_size_value * 100)
print('Doğruluk oranı:', accuracy_score(y_test, predictdt))
print('Decision Tree algoritmasını kullanarak yapılan modelin tahmin oranı : %',
      round(accuracy_score(y_test, predictdt), 5) * 100)

# Karşılaştırma matrisini göster
cnf_matrixdt = confusion_matrix(y_test, predictdt)
class_names = vc['Status'].unique()  # 'Status' sınıflarını al

# Matris için görselleştirme
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Sıcaklık haritası
sns.heatmap(pd.DataFrame(cnf_matrixdt), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Karşılaştırma Matrisi', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.show()

# Kesinlik, Duyarlılık ve F1 Skorunu hesapla
report = classification_report(y_test, predictdt, target_names=np.unique(y), output_dict=True)
print('Kesinlik, Duyarlılık ve F1 Skoru:\n', report)

# Raporu daha okunabilir hale getirelim
report_df_dt = pd.DataFrame(report).transpose()
print(report_df_dt)


# Cohen's Kappa Değeri
kappadt = cohen_kappa_score(y_test, predictdt)
print("Cohen's Kappa Değeri:", kappadt)

# ROC Eğrisi ve AUC Analizi (Sınıf başına)

# `best_estimator_` kullanarak en iyi modeli al ve pozitif sınıfa ait olasılıkları alın
y_scoresvm = modeldt.predict_proba(X_test_selected)[:, 1]  # Pozitif sınıf için olasılıklar

# ROC eğrisi ve AUC hesaplama
fpr, tpr, _ = roc_curve(y_test, y_scoresvm)  # Yanlış pozitif oranı (FPR), Doğru pozitif oranı (TPR)
roc_aucdt = auc(fpr, tpr)  # Eğri altındaki alan (AUC)

# ROC eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_aucdt:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Rastgele tahmin
plt.title('DT Modeli için ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Genel Özellikleri Yazdır
print("Accuracy:", accuracy_score(y_test, predictdt))
print("F1-Skoru:", f1_score(y_test, predictdt, average='weighted'))
print("Precision:", precision_score(y_test, predictdt, average='weighted'))
print("Sensitivity/Recall:", recall_score(y_test, predictdt, average='weighted'))

# XGBoost modelini oluştur
xgb = XGBClassifier(eval_metric='logloss')  # eval_metric uyarısını önlemek için

# Hiperparametre aralığı tanımla
params = {
    'max_depth': [3, 4, 5],  # Karar ağacı derinliği
    'learning_rate': [0.01, 0.1],  # Öğrenme hızı
    'n_estimators': [34, 35, 36],  # Ağaç sayısı
    'subsample': [0.9, 1],  # Rastgele alt örnekleme oranı
    'colsample_bytree': [0.7, 1],  # Özellik alt örnekleme oranı
    'scale_pos_weight': [1, 2],  # Dengeleme parametresini test etmek
    'reg_alpha': [1],  # L1 düzenlileştirme parametresi
    'reg_lambda': [0]  # L2 düzenlileştirme parametresi
}

# GridSearchCV ile en iyi parametreleri bul
modelxgb = GridSearchCV(xgb, params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
modelxgb.fit(X_train, y_train)

# En iyi parametreleri yazdır
print("En iyi parametreler:", modelxgb.best_params_)

# Tahmin yap
predictxgb = modelxgb.predict(X_test)

# Doğruluk oranını hesapla
print('Eğitime ayrılan yüzde:%', test_size_value * 100)
print('Doğruluk oranı:', accuracy_score(y_test, predictxgb))
print('XGBoost algoritmasını kullanarak yapılan modelin tahmin oranı : %',
      round(accuracy_score(y_test, predictxgb), 5) * 100)

# Karşılaştırma matrisini göster
cnf_matrixxgb = confusion_matrix(y_test, predictxgb)
class_names = np.unique(y_train)  # Hedef değişkenin sınıflarını al

# Matris için görselleştirme
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Sıcaklık haritası
sns.heatmap(pd.DataFrame(cnf_matrixxgb), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Karşılaştırma Matrisi', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.show()

# Kesinlik, Duyarlılık ve F1 Skorunu hesapla
report = classification_report(y_test, predictxgb, target_names=class_names, output_dict=True)
print('Kesinlik, Duyarlılık ve F1 Skoru:\n', report)

# Raporu daha okunabilir hale getirelim
report_df_xgb = pd.DataFrame(report).transpose()
print(report_df_xgb)

# Cohen's Kappa Değeri
kappaxgb = cohen_kappa_score(y_test, predictxgb)
print("Cohen's Kappa Değeri:", kappaxgb)

# ROC Eğrisi ve AUC Analizi (Pozitif sınıf için)
y_scores = modelxgb.predict_proba(X_test)[:, 1]  # Pozitif sınıf için olasılıklar

# ROC eğrisi ve AUC hesaplama
fpr, tpr, _ = roc_curve(y_test, y_scores)  # Yanlış pozitif oranı (FPR), Doğru pozitif oranı (TPR)
roc_aucxgb = auc(fpr, tpr)  # Eğri altındaki alan (AUC)

# ROC eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_aucxgb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Rastgele tahmin
plt.title('XGBoost Modeli için ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# Genel Özellikleri Yazdır
print("Accuracy:", accuracy_score(y_test, predictxgb))
print("F1-Skoru:", f1_score(y_test, predictxgb, average='weighted'))
print("Precision:", precision_score(y_test, predictxgb, average='weighted'))
print("Sensitivity/Recall:", recall_score(y_test, predictxgb, average='weighted'))


# FNN Modeli (basit yapay sinir ağı)
fnn_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # İlk katman
    Dense(1, activation='sigmoid')  # Çıkış katmanı
])

# Modelin derlenmesi
fnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping callback eklemek (doğrulama kaybı iyileşmezse erken durdurma)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğitmek
fnn_history = fnn_model.fit(X_train, y_train,
                            epochs=100,
                            batch_size=16,
                            validation_split=0.2,
                            verbose=1,
                            callbacks=[early_stopping])

# Test verisi üzerinde değerlendirme
fnn_loss, fnn_accuracy = fnn_model.evaluate(X_test, y_test)
print(f"FNN Test Loss: {fnn_loss}")
print(f"FNN Test Accuracy: {fnn_accuracy}")

# Eğitim ve doğrulama kayıplarını görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(fnn_history.history['loss'], label='Training Loss')
plt.plot(fnn_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('FNN Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Eğitim ve doğrulama doğruluğunu görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(fnn_history.history['accuracy'], label='Training Accuracy')
plt.plot(fnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('FNN Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Test verisi ile tahmin yapma
fnn_predictions = fnn_model.predict(X_test)
fnn_predictions = (fnn_predictions > 0.5).astype(int)  # Tahminleri ikili sınıfa çevir (0 veya 1)
print("FNN Örnek Tahminler:", fnn_predictions.flatten())
print("Gerçek Etiketler:", y_test)



# Cohen's Kappa Değeri
fnn_kappa = cohen_kappa_score(y_test, fnn_predictions)
print("FNN Cohen's Kappa Değeri:", fnn_kappa)

# Precision, Recall ve F1 Skorlarını hesapla
fnn_precision = precision_score(y_test, fnn_predictions)
fnn_recall = recall_score(y_test, fnn_predictions)
fnn_f1 = f1_score(y_test, fnn_predictions)
print("FNN Precision:", fnn_precision)
print("FNN Recall:", fnn_recall)
print("FNN F1 Score:", fnn_f1)

# ROC ve AUC hesapla
fpr, tpr, _ = roc_curve(y_test, fnn_model.predict(X_test))
fnn_roc_auc = auc(fpr, tpr)
print(f"FNN AUC: {fnn_roc_auc}")
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'FNN ROC curve (area = {fnn_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('FNN Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Karışıklık Matrisi
fnn_cnf_matrix = confusion_matrix(y_test, fnn_predictions)

# Grafik oluşturuluyor
fig, ax = plt.subplots(figsize=(8, 6))  # Grafik boyutunu ayarlıyoruz

tick_marks = np.arange(len(class_names))  # Sınıf isimleri için tick'ler
plt.xticks(tick_marks, class_names)  # X ekseninde sınıf isimlerini yerleştiriyoruz
plt.yticks(tick_marks, class_names)  # Y ekseninde sınıf isimlerini yerleştiriyoruz

# Sıcaklık haritası (heatmap) çiziliyor
sns.heatmap(fnn_cnf_matrix, annot=True, cmap='YlGnBu', fmt='g', ax=ax)  # 'g' formatı, sayıları tam sayı olarak gösterir
ax.xaxis.set_label_position('top')  # X ekseninin başlığını üstte yerleştiriyoruz

# Başlık ve etiketler
plt.tight_layout()
plt.title('FNN Karışıklık Matrisi', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')

# Grafik gösterimi
plt.show()



# MLP Modeli (gelişmiş yapay sinir ağı)
mlp_model = Sequential([
    # Gizli Katman 1
    Dense(128, input_dim=X_train.shape[1], activation='relu'),  # İlk katman
    Dropout(0.3),  # Aşırı öğrenmeyi engellemek için dropout

    # Gizli Katman 2
    Dense(64, activation='relu'),  # Orta katman
    Dropout(0.3),  # Aşırı öğrenmeyi engellemek için dropout

    # Gizli Katman 3
    Dense(32, activation='relu'),  # Diğer orta katman
    Dropout(0.2),  # Aşırı öğrenmeyi engellemek için dropout

    # Çıkış Katmanı
    Dense(1, activation='sigmoid')  # Çıkış katmanı
])

# Modelin derlenmesi
mlp_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping callback eklemek (doğrulama kaybı iyileşmezse erken durdurma)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğitmek
mlp_history = mlp_model.fit(X_train, y_train,
                            epochs=100,
                            batch_size=16,
                            validation_split=0.2,
                            verbose=1,
                            callbacks=[early_stopping])

# Test verisi üzerinde değerlendirme
mlp_loss, mlp_accuracy = mlp_model.evaluate(X_test, y_test)
print(f"MLP Test Loss: {mlp_loss}")
print(f"MLP Test Accuracy: {mlp_accuracy}")

# Loss grafiği
plt.figure(figsize=(12, 6))
plt.plot(mlp_history.history['loss'], label='Training Loss')
plt.plot(mlp_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('MLP Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Accuracy grafiği
plt.figure(figsize=(12, 6))
plt.plot(mlp_history.history['accuracy'], label='Training Accuracy')
plt.plot(mlp_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('MLP Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Örnek tahmin
mlp_predictions = mlp_model.predict(X_test)
mlp_predictions = (mlp_predictions > 0.5).astype(int)  # Tahminleri ikili sınıfa çevir (0 veya 1)
print("MLP Tahminler:", mlp_predictions.flatten())
print("Gerçek Etiketler:", y_test)

# Test verisi üzerinde değerlendirme
mlp_loss, mlp_accuracy = mlp_model.evaluate(X_test, y_test)
print(f"MLP Test Loss: {mlp_loss}")
print(f"MLP Test Accuracy: {mlp_accuracy}")

# Cohen's Kappa Değeri
mlp_kappa = cohen_kappa_score(y_test, mlp_predictions)
print("MLP Cohen's Kappa Değeri:", mlp_kappa)

# Precision, Recall ve F1 Skorlarını hesapla
mlp_precision = precision_score(y_test, mlp_predictions)
mlp_recall = recall_score(y_test, mlp_predictions)
mlp_f1 = f1_score(y_test, mlp_predictions)
print("MLP Precision:", mlp_precision)
print("MLP Recall:", mlp_recall)
print("MLP F1 Score:", mlp_f1)

# ROC ve AUC hesapla
fpr, tpr, _ = roc_curve(y_test, mlp_model.predict(X_test))
mlp_roc_auc = auc(fpr, tpr)
print(f"MLP AUC: {mlp_roc_auc}")
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'MLP ROC curve (area = {mlp_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLP Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Karışıklık Matrisi
mlp_cnf_matrix = confusion_matrix(y_test, mlp_predictions)

# Grafik oluşturuluyor
fig, ax = plt.subplots(figsize=(8, 6))  # Grafik boyutunu ayarlıyoruz

tick_marks = np.arange(len(class_names))  # Sınıf isimleri için tick'ler
plt.xticks(tick_marks, class_names)  # X ekseninde sınıf isimlerini yerleştiriyoruz
plt.yticks(tick_marks, class_names)  # Y ekseninde sınıf isimlerini yerleştiriyoruz

# Sıcaklık haritası (heatmap) çiziliyor
sns.heatmap(mlp_cnf_matrix, annot=True, cmap='YlGnBu', fmt='g', ax=ax)  # 'g' formatı, sayıları tam sayı olarak gösterir
ax.xaxis.set_label_position('top')  # X ekseninin başlığını üstte yerleştiriyoruz

# Başlık ve etiketler
plt.tight_layout()
plt.title('MLP Karışıklık Matrisi', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')

# Grafik gösterimi
plt.show()



# DNN Modeli (Derin yapay sinir ağı)
dnn_model = Sequential([
    # Gizli Katman 1
    Dense(512, input_dim=X_train.shape[1], activation='relu'),  # İlk katman
    Dropout(0.3),  # Aşırı öğrenmeyi engellemek için dropout

    # Gizli Katman 2
    Dense(256, activation='relu'),  # Orta katman
    Dropout(0.3),  # Aşırı öğrenmeyi engellemek için dropout

    # Gizli Katman 3
    Dense(128, activation='relu'),  # Diğer orta katman
    Dropout(0.2),  # Aşırı öğrenmeyi engellemek için dropout

    # Gizli Katman 4
    Dense(64, activation='relu'),  # Bir başka orta katman
    Dropout(0.2),  # Aşırı öğrenmeyi engellemek için dropout

    # Gizli Katman 5
    Dense(32, activation='relu'),  # Son bir gizli katman
    Dropout(0.1),  # Aşırı öğrenmeyi engellemek için dropout

    # Çıkış Katmanı
    Dense(1, activation='sigmoid')  # Çıkış katmanı
])

# Modelin derlenmesi
dnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping callback eklemek (doğrulama kaybı iyileşmezse erken durdurma)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğitmek
dnn_history = dnn_model.fit(X_train, y_train,
                            epochs=100,
                            batch_size=16,
                            validation_split=0.2,
                            verbose=1,
                            callbacks=[early_stopping])

# Test verisi üzerinde değerlendirme
dnn_loss, dnn_accuracy = dnn_model.evaluate(X_test, y_test)
print(f"DNN Test Loss: {dnn_loss}")
print(f"DNN Test Accuracy: {dnn_accuracy}")

# Loss grafiği
plt.figure(figsize=(12, 6))
plt.plot(dnn_history.history['loss'], label='Training Loss')
plt.plot(dnn_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('DNN Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Accuracy grafiği
plt.figure(figsize=(12, 6))
plt.plot(dnn_history.history['accuracy'], label='Training Accuracy')
plt.plot(dnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('DNN Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Örnek tahmin
dnn_predictions = dnn_model.predict(X_test)
dnn_predictions = (dnn_predictions > 0.5).astype(int)  # Tahminleri ikili sınıfa çevir (0 veya 1)
print("DNN Tahminler:", dnn_predictions.flatten())
print("Gerçek Etiketler:", y_test)

# Test verisi üzerinde değerlendirme
dnn_loss, dnn_accuracy = dnn_model.evaluate(X_test, y_test)
print(f"DNN Test Loss: {dnn_loss}")
print(f"DNN Test Accuracy: {dnn_accuracy}")

# Cohen's Kappa Değeri
dnn_kappa = cohen_kappa_score(y_test, dnn_predictions)
print("DNN Cohen's Kappa Değeri:", dnn_kappa)

# Precision, Recall ve F1 Skorlarını hesapla
dnn_precision = precision_score(y_test, dnn_predictions)
dnn_recall = recall_score(y_test, dnn_predictions)
dnn_f1 = f1_score(y_test, dnn_predictions)
print("DNN Precision:", dnn_precision)
print("DNN Recall:", dnn_recall)
print("DNN F1 Score:", dnn_f1)

# ROC ve AUC hesapla
fpr, tpr, _ = roc_curve(y_test, dnn_model.predict(X_test))
dnn_roc_auc = auc(fpr, tpr)
print(f"DNN AUC: {dnn_roc_auc}")
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DNN ROC curve (area = {dnn_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DNN Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Karışıklık Matrisi
dnn_cnf_matrix = confusion_matrix(y_test, dnn_predictions)

# Grafik oluşturuluyor
fig, ax = plt.subplots(figsize=(8, 6))  # Grafik boyutunu ayarlıyoruz

tick_marks = np.arange(len(class_names))  # Sınıf isimleri için tick'ler
plt.xticks(tick_marks, class_names)  # X ekseninde sınıf isimlerini yerleştiriyoruz
plt.yticks(tick_marks, class_names)  # Y ekseninde sınıf isimlerini yerleştiriyoruz

# Sıcaklık haritası (heatmap) çiziliyor
sns.heatmap(dnn_cnf_matrix, annot=True, cmap='YlGnBu', fmt='g', ax=ax)  # 'g' formatı, sayıları tam sayı olarak gösterir
ax.xaxis.set_label_position('top')  # X ekseninin başlığını üstte yerleştiriyoruz

# Başlık ve etiketler
plt.tight_layout()
plt.title('DNN Karışıklık Matrisi', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')

# Grafik gösterimi
plt.show()




# CNN Modeli

# Pandas DataFrame'i NumPy dizisine çeviriyoruz
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

# Veriyi CNN için şekillendirme
X_train_cnn = X_train_np.reshape(-1, 15, 1, 1)  # (2816, 15, 1, 1)
X_test_cnn = X_test_np.reshape(-1, 15, 1, 1)

# CNN Modeli
model_cnn = Sequential([
    Conv2D(32, (1, 1), activation='relu', input_shape=(15, 1, 1)),  # 1x1 kernel kullanımı
    MaxPooling2D(pool_size=(1, 1)),  # Havuzlama katmanı (1x1 havuz)
    Flatten(),  # Düzleştirme katmanı
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Çıkış katmanı
])

model_cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping callback eklemek (doğrulama kaybı iyileşmezse erken durdurma)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğitmek
history_cnn = model_cnn.fit(X_train_cnn, y_train,
                            epochs=100,
                            batch_size=16,
                            validation_split=0.2,
                            verbose=1,
                            callbacks=[early_stopping])

# Test verisi üzerinde değerlendirme
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_cnn, y_test)
print(f"CNN Test Loss: {loss_cnn}")
print(f"CNN Test Accuracy: {accuracy_cnn}")

# Loss grafiği
plt.figure(figsize=(12, 6))
plt.plot(history_cnn.history['loss'], label='Training Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('CNN Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Accuracy grafiği
plt.figure(figsize=(12, 6))
plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('CNN Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Cohen's Kappa Değeri
y_pred_cnn = (model_cnn.predict(X_test_cnn) > 0.5).astype(int)
kappa_cnn = cohen_kappa_score(y_test, y_pred_cnn)
print("CNN Cohen's Kappa Değeri:", kappa_cnn)

# Precision, Recall ve F1 Skorlarını hesapla
cnn_precision = precision_score(y_test, y_pred_cnn)
cnn_recall = recall_score(y_test, y_pred_cnn)
cnn_f1 = f1_score(y_test, y_pred_cnn)
print("CNN Precision:", cnn_precision)
print("CNN Recall:", cnn_recall)
print("CNN F1 Score:", cnn_f1)

# ROC ve AUC hesapla
fpr, tpr, _ = roc_curve(y_test, model_cnn.predict(X_test_cnn))
cnn_roc_auc = auc(fpr, tpr)
print(f"CNN AUC: {cnn_roc_auc}")
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'CNN ROC curve (area = {cnn_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CNN Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Karışıklık Matrisi
cnf_matrix_cnn = confusion_matrix(y_test, y_pred_cnn)

# Grafik oluşturuluyor
fig, ax = plt.subplots(figsize=(8, 6))  # Grafik boyutunu ayarlıyoruz

tick_marks = np.arange(len(class_names))  # Sınıf isimleri için tick'ler
plt.xticks(tick_marks, class_names)  # X ekseninde sınıf isimlerini yerleştiriyoruz
plt.yticks(tick_marks, class_names)  # Y ekseninde sınıf isimlerini yerleştiriyoruz

# Sıcaklık haritası (heatmap) çiziliyor
sns.heatmap(cnf_matrix_cnn, annot=True, cmap='YlGnBu', fmt='g', ax=ax)  # 'g' formatı, sayıları tam sayı olarak gösterir
ax.xaxis.set_label_position('top')  # X ekseninin başlığını üstte yerleştiriyoruz

# Başlık ve etiketler
plt.tight_layout()
plt.title('CNN Karışıklık Matrisi', y=1.1)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')

# Grafik gösterimi
plt.show()



# Tüm modellerin sonuçlarını bir sözlükte topluyoruz
results = {
    "Model": ["K-NN","Random Forest",'Logistic Regression', 'Support Vector Machine (SVM)', 'Decision Tree', 'XGBoost',"FNN", "MLP", "DNN", "CNN"],
    "Accuracy": [
        fnn_accuracy, mlp_accuracy, dnn_accuracy, accuracy_cnn,
        accuracy_score(y_test, predictknn),
        accuracy_score(y_test, predict_RF),
        accuracy_score(y_test, predict_LR),
        accuracy_score(y_test, predictsvm),
        accuracy_score(y_test, predictdt),
        accuracy_score(y_test, predictxgb)
    ],
    "Precision": [
        fnn_precision, mlp_precision, dnn_precision, cnn_precision,
        precision_score(y_test, predictknn, average='weighted'),
        precision_score(y_test, predict_RF, average='weighted'),
        precision_score(y_test, predict_LR, average='weighted'),
        precision_score(y_test, predictsvm, average='weighted'),
        precision_score(y_test, predictdt, average='weighted'),
        precision_score(y_test, predictxgb, average='weighted')
    ],
    "Recall": [
        fnn_recall, mlp_recall, dnn_recall, cnn_recall,
        recall_score(y_test, predictknn, average='weighted'),
        recall_score(y_test, predict_RF, average='weighted'),
        recall_score(y_test, predict_LR, average='weighted'),
        recall_score(y_test, predictsvm, average='weighted'),
        recall_score(y_test, predictdt, average='weighted'),
        recall_score(y_test, predictxgb, average='weighted')
    ],
    "F1-Score": [
        fnn_f1, mlp_f1, dnn_f1, cnn_f1,
        f1_score(y_test, predictknn, average='weighted'),
        f1_score(y_test, predict_RF, average='weighted'),
        f1_score(y_test, predict_LR, average='weighted'),
        f1_score(y_test, predictsvm, average='weighted'),
        f1_score(y_test, predictdt, average='weighted'),
        f1_score(y_test, predictxgb, average='weighted')
    ],
    "AUC": [
        fnn_roc_auc, mlp_roc_auc, dnn_roc_auc, cnn_roc_auc,
        roc_aucknn, roc_aucrf, roc_auclr, roc_aucsvm,roc_aucdt,roc_aucxgb
    ],
    "Cohen's Kappa": [
        kappaknn, kapparf, kappalr, kappasvm, kappadt, kappaxgb, fnn_kappa, mlp_kappa, dnn_kappa, kappa_cnn
    ],
}

# Sonuçları DataFrame olarak kaydediyoruz
results_df = pd.DataFrame(results)


# Her bir model için metriklerin yan yana çizilmesi
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Cohen's Kappa"]

# Metrikler için renkler
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']

# Grafik boyutu
plt.figure(figsize=(18, 12))

# Barların genişliği
width = 0.1

# X ekseninde her model için bir pozisyon
x = np.arange(len(results_df))

# Her bir model için metrikleri çizmek
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, results_df[metric], width, label=metric, color=colors[i])

# X eksenindeki etiketler (modellerin adları)
plt.xticks(x + width * (len(metrics) - 1) / 2, results_df['Model'], rotation=45, ha="right")

# Başlık, etiketler ve grid
plt.title('Model Performans Metrikleri Karşılaştırması', fontsize=16)
plt.ylabel('Değerler', fontsize=12)
plt.xlabel('Modeller', fontsize=12)
plt.ylim(0, 1)  # Y eksenini 0-1 arası sınırla
plt.legend(title="Metrikler", bbox_to_anchor=(1.05, 1), loc='upper left')

# Grid çizimi
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Grafik gösterimi
plt.tight_layout()
plt.show()

# Accuracy'e göre sıralama yapıyoruz
results_df_sorted = results_df.sort_values(by="Accuracy", ascending=False)

# Sonuçları yazdırma
print("\nModel Performans Karşılaştırma Tablosu:")
print(results_df_sorted)

# Tablonun görselleştirilmesi için Accuracy değerlerini çubuk grafik olarak çizelim
plt.figure(figsize=(12, 8))
plt.bar(results_df_sorted["Model"], results_df_sorted["Accuracy"], color='cyan')
plt.title("Model Accuracy Karşılaştırması")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Adım 1: En İyi Modeli Seçme ve Kaydetme
# En iyi modeli accuracy ve F1 skoru değerine göre seçiyoruz
best_model_name = results_df_sorted.iloc[0]['Model']
best_model_accuracy = results_df_sorted.iloc[0]['Accuracy']
best_model_f1 = results_df_sorted.iloc[0]['F1-Score']
print(f"En iyi model: {best_model_name} (Accuracy: {best_model_accuracy}, F1-Score: {best_model_f1})")


# Seçilen modelin referansına göre model nesnesi
if best_model_name == "K-NN":
    best_model = modelknn  # Daha önce eğittiğiniz K-NN modelini kullanın
elif best_model_name == "Random Forest":
    best_model = modelRF  # Daha önce eğittiğiniz Random Forest modelini kullanın
elif best_model_name == "Logistic Regression":
    best_model = modelLR  # Daha önce eğittiğiniz Logistic Regression modelini kullanın
elif best_model_name == "Support Vector Machine (SVM)":
    best_model = modelsvm  # Daha önce eğittiğiniz SVM modelini kullanın
elif best_model_name == "Decision Tree":
    best_model = modeldt  # Daha önce eğittiğiniz Decision Tree modelini kullanın
elif best_model_name == "XGBoost":
    best_model = modelxgb  # Daha önce eğittiğiniz XGBoost modelini kullanın
elif best_model_name == "FNN":
    best_model = fnn_model  # Daha önce eğittiğiniz FNN modelini kullanın
elif best_model_name == "MLP":
    best_model = mlp_model  # Daha önce eğittiğiniz MLP modelini kullanın
elif best_model_name == "DNN":
    best_model = dnn_model  # Daha önce eğittiğiniz DNN modelini kullanın
elif best_model_name == "CNN":
    best_model = model_cnn  # Daha önce eğittiğiniz CNN modelini kullanın

# Modeli kaydediyoruz
joblib.dump(best_model, 'best_model.pkl')  # Modeli 'best_model.pkl' olarak kaydediyoruz
print(f"{best_model_name} modeli kaydedildi.")


# Adım 2: Yeni Veriler Üzerinde Tahmin Yapma Fonksiyonu
def predict_new_samples(model, new_data):
    """
    Verilen model ile yeni veriler için tahmin yapar.

    :param model: Eğitilmiş model
    :param new_data: Yeni veriler (pandas DataFrame veya numpy array formatında)
    :return: Tahmin edilen sınıflar
    """
    predictions = model.predict(new_data)  # Model ile tahmin yapıyoruz
    return predictions




def get_input_for_features():
    """
    Kullanıcıdan SEER Breast Cancer veri kümesinin tüm özelliklerine dair veri alır ve pandas DataFrame formatında döndürür.
    """
    # Kullanıcıdan sayısal veri alıyoruz
    age = int(input("Yaş (Age) girin: "))
    tumor_size = float(input("Tümör Büyüklüğü (Tumor Size) girin (mm cinsinden): "))
    survival_months = int(input("Yaşama Ayları (Survival Months) girin: "))

    # Kategorik veriler için seçenekler sunuyoruz
    race_options = ["White", "Black", "Other"]
    race = None
    while race not in race_options:
        print(f"Geçerli seçenekler: {race_options}")
        race = input("Yarış (Race) seçin: ")
        if race not in race_options:
            print("Geçersiz seçim! Lütfen geçerli bir seçenek girin.")

    marital_status_options = ["Single", "Married", "Separated", "Divorced", "Widowed"]
    marital_status = None
    while marital_status not in marital_status_options:
        print(f"Geçerli seçenekler: {marital_status_options}")
        marital_status = input("Evlilik Durumu (Marital Status) seçin: ")
        if marital_status not in marital_status_options:
            print("Geçersiz seçim! Lütfen geçerli bir seçenek girin.")

    t_stage_options = ["T1", "T2", "T3", "T4"]
    t_stage = None
    while t_stage not in t_stage_options:
        print(f"Geçerli seçenekler: {t_stage_options}")
        t_stage = input("T Stage seçin: ")
        if t_stage not in t_stage_options:
            print("Geçersiz seçim! Lütfen geçerli bir seçenek girin.")

    n_stage_options = ["N1", "N2", "N3"]
    n_stage = None
    while n_stage not in n_stage_options:
        print(f"Geçerli seçenekler: {n_stage_options}")
        n_stage = input("N Stage seçin: ")
        if n_stage not in n_stage_options:
            print("Geçersiz seçim! Lütfen geçerli bir seçenek girin.")

    stage_6th_options = ["IIA", "IIB", "IIIA", "IIIB", "IIIC"]
    stage_6th = None
    while stage_6th not in stage_6th_options:
        print(f"Geçerli seçenekler: {stage_6th_options}")
        stage_6th = input("6th Stage seçin: ")
        if stage_6th not in stage_6th_options:
            print("Geçersiz seçim! Lütfen geçerli bir seçenek girin.")

    grade_options = ["Grade I", "Grade II", "Grade III", "Grade IV"]
    grade = None
    while grade not in grade_options:
        print(f"Geçerli seçenekler: {grade_options}")
        grade = input("Grade seçin: ")
        if grade not in grade_options:
            print("Geçersiz seçim! Lütfen geçerli bir seçenek girin.")

    a_stage_options = ["Regional", "Distant"]
    a_stage = None
    while a_stage not in a_stage_options:
        print(f"Geçerli seçenekler: {a_stage_options}")
        a_stage = input("A Stage seçin: ")
        if a_stage not in a_stage_options:
            print("Geçersiz seçim! Lütfen geçerli bir seçenek girin.")

    estrogen_status_options = ["Positive", "Negative"]
    estrogen_status = None
    while estrogen_status not in estrogen_status_options:
        print(f"Geçerli seçenekler: {estrogen_status_options}")
        estrogen_status = input("Estrogen Durumu (Estrogen Status) seçin: ")
        if estrogen_status not in estrogen_status_options:
            print("Geçersiz seçim! Lütfen geçerli bir seçenek girin.")

    progesterone_status_options = ["Positive", "Negative"]
    progesterone_status = None
    while progesterone_status not in progesterone_status_options:
        print(f"Geçerli seçenekler: {progesterone_status_options}")
        progesterone_status = input("Progesteron Durumu (Progesterone Status) seçin: ")
        if progesterone_status not in progesterone_status_options:
            print("Geçersiz seçim! Lütfen geçerli bir seçenek girin.")

    regional_nodes_examined = int(input("Bölgesel Nodüller İncelendi (Regional Nodes Examined) sayısını girin: "))
    regional_nodes_positive = int(input("Bölgesel Nodüller Pozitif (Regional Nodes Positive) sayısını girin: "))

    # Verileri bir pandas DataFrame'e dönüştürüyoruz
    new_data = pd.DataFrame({
        'Age': [age],
        'Race': [race],
        'Marital Status': [marital_status],
        'T Stage': [t_stage],
        'N Stage': [n_stage],
        '6th Stage': [stage_6th],
        'Grade': [grade],
        'A Stage': [a_stage],
        'Tumor Size': [tumor_size],
        'Estrogen Status': [estrogen_status],
        'Progesterone Status': [progesterone_status],
        'Regional Node Examined': [regional_nodes_examined],
        'Regional Node Positive': [regional_nodes_positive],
        'Survival Months': [survival_months]
    })

    # Encoding işlemleri
    vmap_grade = {
        "Grade I": 0,
        "Grade II": 1,
        "Grade III": 2,
        "Grade IV": 3
    }
    new_data['Grade'] = new_data['Grade'].replace(vmap_grade)

    vmap_6th_stage = {
        "IIA": 0,
        "IIB": 1,
        "IIIA": 2,
        "IIIB": 3,
        "IIIC": 4
    }
    new_data['6th Stage'] = new_data['6th Stage'].replace(vmap_6th_stage)

    vmap_n_stage = {
        "N1": 0,
        "N2": 1,
        "N3": 2
    }
    new_data['N Stage'] = new_data['N Stage'].replace(vmap_n_stage)

    vmap_t_stage = {
        "T1": 0,
        "T2": 1,
        "T3": 2,
        "T4": 3
    }
    new_data['T Stage'] = new_data['T Stage'].replace(vmap_t_stage)

    vmap_estrogen_status = {
        "Positive": 1,
        "Negative": 0
    }
    new_data['Estrogen Status'] = new_data['Estrogen Status'].replace(vmap_estrogen_status)

    vmap_progesterone_status = {
        "Positive": 1,
        "Negative": 0
    }
    new_data['Progesterone Status'] = new_data['Progesterone Status'].replace(vmap_progesterone_status)

    vmap_a_stage = {
        "Regional": 1,
        "Distant": 0
    }
    new_data['A Stage'] = new_data['A Stage'].replace(vmap_a_stage)

    vmap_race = {
        "White": 0,
        "Black": 1,
        "Other": 2
    }
    new_data['Race'] = new_data['Race'].replace(vmap_race)

    vmap_marital_status = {
        "Married": 0,
        "Single": 1,
        "Divorced": 2,
        "Widowed": 3,
        "Separated": 4
    }
    new_data['Marital Status'] = new_data['Marital Status'].replace(vmap_marital_status)

    return new_data


# Yeni veri alalım
new_data = get_input_for_features()

# Yeni veriye Risk Grubu sütununu ekle
new_data['Risk Group'] = new_data.apply(risk_group, axis=1)

# Tüm satırları göstermek için
pd.set_option('display.max_rows', None)

# Tüm sütunları göstermek için
pd.set_option('display.max_columns', None)

# Genişliği sınırsız yapmak için
pd.set_option('display.width', None)

# Yeni verileri gösterelim
print("\nYeni Veri:")
print(new_data)

# Veri türlerini kontrol et
print(new_data.dtypes)

new_data_np = new_data

if best_model_name == "CNN":
    # Veriyi CNN için uygun formatta şekillendiriyoruz (4D: [samples, 15, 1, 1])
    new_data_cnn = new_data_np.reshape(-1, 15, 1, 1)  # (n_samples, 15, 1, 1)


# Modeli yükleyelim (CNN modelini veya XGBoost gibi bir model olabilir)
loaded_model = joblib.load('best_model.pkl')  # Kaydedilen modeli yüklüyoruz

# Yeni veriye tahmin yapalım
predictions = predict_new_samples(loaded_model, new_data)

for prediction in predictions:
    if prediction == 1:
        print("Tahmin:[1] Alive")
    else:
        print("Tahmin:[0] Dead")









