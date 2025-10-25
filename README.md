# MLflow Eğitimi — Uçtan Uca Titanic Projesi

Bu repo, MLflow'u Titanic verisiyle gerçek bir makine öğrenmesi projesi üzerinde öğretmek için hazırlanmıştır. Eğitim boyunca:
- Veri hazırlama ve feature engineering
- MLflow Tracking ile deney takibi
- Farklı modelleri karşılaştırma
- MLflow UI üzerinde analiz
- Model kaydetme ve yükleme

konularını uygulamalı olarak göreceksiniz.



## Proje Yapısı
- `configs/config_titanic.yaml`: Veri kaynağı, model seçimi ve hiperparametrelerin tanımlandığı ana config.
- `src/data/make_dataset.py`: Titanic verisini işler, train/test CSV'lerini `data/processed/` altına yazar.
- `src/models/train.py`: MLflow run'ı başlatır, parametre/metrik loglar, confusion matrix artifact'ı üretir, modeli kaydeder.
- `src/models/evaluate.py`: Son run'daki modeli yükler, test setinde değerlendirir, sonuçları yeni bir run'a loglar.
- `tests/test_train_smoke.py`: Pipeline'ın çalıştığını doğrulayan test.
- `KOMUT_REFERANSI.md`: Ders sırasında kullanılacak tüm komutların hızlı referansı.

```
├── configs/
│   └── config_titanic.yaml    ← Tüm ayarlar (model, veri, tracking)
├── data/
│   ├── raw/titanic.csv        ← Ham veri (Kaggle'dan)
│   └── processed/             ← İşlenmiş veri (otomatik oluşur)
├── src/
│   ├── data/make_dataset.py   ← Veri hazırlama + feature engineering
│   ├── models/train.py        ← Model eğitimi + MLflow tracking
│   ├── models/evaluate.py     ← Model değerlendirme + yükleme
│   └── utils/io.py            ← YAML okuma, override merge
├── mlruns/                    ← MLflow kayıtları (otomatik oluşur)
├── artifacts/                 ← Confusion matrix vs (otomatik oluşur)
└── tests/test_train_smoke.py  ← Pipeline testi
```

## Veri Seti Hakkında

### Titanic Dataset
Bu proje, Kaggle'ın ünlü Titanic veri setini kullanır. Veri seti, Titanic felaketinde hayatta kalma durumunu tahmin etmek için yolcu bilgilerini içerir.

**Kullanılan Kolonlar:**
- `survived`: Hayatta kalma durumu (0 = Hayır, 1 = Evet) - **TARGET**
- `pclass`: Yolcu sınıfı (1 = 1. sınıf, 2 = 2. sınıf, 3 = 3. sınıf)
- `sex`: Cinsiyet (male/female)
- `age`: Yaş
- `sibsp`: Gemideki kardeş/eş sayısı
- `parch`: Gemideki ebeveyn/çocuk sayısı
- `fare`: Bilet ücreti
- `embarked`: Biniş limanı (C = Cherbourg, Q = Queenstown, S = Southampton)

**Not:** Orijinal veri setinde `class`, `who`, `adult_male`, `deck`, `embark_town`, `alive`, `alone` gibi ek kolonlar da bulunabilir. Bu proje yukarıdaki 8 temel kolonu kullanır ve feature engineering ile yeni özellikler üretir (`family_size`, `is_alone`, `age_group`).

**Veri Seti İstatistikleri:**
- Toplam satır: 891
- Train set: ~712 satır (%80)
- Test set: ~179 satır (%20)

## Ön Gereksinimler
- Python 3.9–3.11
- Kaggle Titanic `train.csv` dosyasını `data/raw/titanic.csv` olarak yerleştirin ([buradan indirebilirsiniz](https://www.kaggle.com/c/titanic/data))
- Çalıştırmadan önce sanal ortam kurup bağımlılıkları yükleyin:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Titanic ile Uçtan Uca Komutlar
```bash
# 1) Veri hazırlama (feature engineering dahil)
python -m src.data.make_dataset --config configs/config_titanic.yaml

# 2) Eğitim — RandomForest run’ı (configte tanımlanan parametreler)
python -m src.models.train --config configs/config_titanic.yaml --run-name rf_titanic_default

# 3) Eğitim — LogisticRegression run’ı (override ile model/parametre değişimi)
python -m src.models.train --config configs/config_titanic.yaml --run-name lr_titanic \
    --override '{"model.name":"LogisticRegression","model.params.max_iter":2000}'

# 4) MLflow UI ile run karşılaştırma
python -m mlflow ui --backend-store-uri file:./mlruns --port 5000
# Tarayıcı: http://127.0.0.1:5000

# 5) Değerlendirme (son run’ı yükle, testte metriği logla)
python -m src.models.evaluate
```

## Ders Akışı 

### 1. MLflow'a Giriş 
- MLflow nedir, neden kullanılır?
- Temel bileşenler: Tracking, Models, Registry
- Alternatif araçlar ve MLflow'un avantajları

### 2. Proje ve Ortam Kurulumu 
- Proje yapısını inceleme
- Sanal ortam kurulumu ve bağımlılıklar
- Config dosyası mantığı (`config_titanic.yaml`)

### 3. Veri Hazırlama ve Feature Engineering 
- Titanic veri setini inceleme
- Eksik değer doldurma stratejileri
- Kategorik kodlama (sex, embarked)
- Yeni özellikler: family_size, is_alone, age_group
- Komut: `python -m src.data.make_dataset --config configs/config_titanic.yaml`

### 4. İlk MLflow Run - RandomForest 
- `src/models/train.py` kodunu inceleme
- MLflow tracking başlatma: `mlflow.set_experiment()`, `mlflow.start_run()`
- Parametre loglama: `mlflow.log_param()`
- Metrik loglama: `mlflow.log_metric()`
- Artifact kaydetme: confusion matrix, model
- Komut: `python -m src.models.train --config configs/config_titanic.yaml --run-name rf_titanic_default`

### 5. MLflow UI İncelemesi 
- UI başlatma: `python -m mlflow ui --backend-store-uri file:./mlruns --port 5000`
- Run'ları listeleme ve filtreleme
- Parametreleri karşılaştırma
- Metrikleri görselleştirme
- Artifact'ları indirme (confusion matrix, model)

### 6. İkinci Model - LogisticRegression 
- Config override kullanımı
- Farklı model ile yeni run
- Komut: `python -m src.models.train --config configs/config_titanic.yaml --run-name lr_titanic --override '{"model.name":"LogisticRegression","model.params.max_iter":2000}'`
- UI'da iki modeli karşılaştırma

### 7. Model Yükleme ve Değerlendirme 
- Kaydedilmiş modeli yükleme: `mlflow.sklearn.load_model()`
- Test setinde değerlendirme
- Tahmin örnekleri üretme
- Komut: `python -m src.models.evaluate`


## Sorun Giderme

**MLflow UI açılmıyor:**
```bash
# Port değiştirerek deneyin
python -m mlflow ui --backend-store-uri file:./mlruns --port 5001
```

**Import hatası:**
```bash
# Sanal ortamı aktive ettiniz mi?
source .venv/bin/activate
pip install -r requirements.txt
```

**Titanic CSV bulunamıyor:**
```bash
# Dosya yolunu kontrol edin
ls data/raw/titanic.csv
# Yoksa Kaggle'dan indirip yerleştirin
```

**Eski run'ları temizlemek:**
```bash
rm -rf mlruns artifacts data/processed
```

## Kaynaklar ve İleri Okuma
- MLflow Dokümantasyon: https://mlflow.org/docs/latest/index.html
- Titanic Dataset: https://www.kaggle.com/c/titanic
- MLflow GitHub: https://github.com/mlflow/mlflow
- Model Serving: https://mlflow.org/docs/latest/models.html#deploy-mlflow-models

---
