# MLflow Komut Referansı (Hızlı Kopya)


## Ortam Hazırlığı

```bash
# Sanal ortamı aktive et
source .venv/bin/activate

# Bağımlılıkları yükle (ilk kurulumda)
pip install -r requirements.txt
```

## Veri Hazırlama

```bash
python -m src.data.make_dataset --config configs/config_titanic.yaml
```

## Model Eğitimi

### RandomForest (Varsayılan)
```bash
python -m src.models.train --config configs/config_titanic.yaml --run-name rf_titanic_default
```

### LogisticRegression (Override ile)
```bash
python -m src.models.train --config configs/config_titanic.yaml --run-name lr_titanic \
    --override '{"model.name":"LogisticRegression","model.params.max_iter":2000}'
```

### KNN (Override ile)
```bash
python -m src.models.train --config configs/config_titanic.yaml --run-name knn_titanic \
    --override '{"model.name":"KNeighborsClassifier","model.params.n_neighbors":5}'
```

## MLflow UI

```bash
# UI'ı başlat
python -m mlflow ui --backend-store-uri file:./mlruns --port 5000

# Tarayıcıda aç: http://127.0.0.1:5000
```

## Model Değerlendirme

```bash
# Son modeli test setinde değerlendir
python -m src.models.evaluate
```

## Test

```bash
# Smoke test çalıştır
pytest tests/test_train_smoke.py -v
```

## Temizlik

```bash
# Tüm run'ları ve artifact'ları temizle
rm -rf mlruns artifacts data/processed

# Sadece işlenmiş veriyi temizle
rm -rf data/processed

# Pytest cache'i temizle
rm -rf .pytest_cache
```


## MLproject ile Çalıştırma (Conda)

```bash
# Veri hazırla
mlflow run . -e prepare_data

# RandomForest eğit
mlflow run . -e train

# LogisticRegression eğit
mlflow run . -e train -P run_name=lr_titanic \
    -P override='{"model.name":"LogisticRegression","model.params.max_iter":2000}'

# Değerlendir
mlflow run . -e evaluate

# Hepsini birden çalıştır
mlflow run . -e titanic_full

> Not: Model Registry komutlarını çalıştırmak için dosya tabanlı backend yerine SQL destekli bir MLflow tracking sunucusu kurmanız gerekir.
```

## Faydalı Git Komutları

```bash
# Değişiklikleri gör
git status

# Değişiklikleri kaydet
git add .
git commit -m "MLflow eğitimi tamamlandı"

# Uzak repo'ya gönder
git push
```

## Hata Ayıklama

```bash
# Python versiyonunu kontrol et
python --version

# MLflow versiyonunu kontrol et
python -m mlflow --version

# Kurulu paketleri listele
pip list

# Hangi Python kullanıldığını gör
which python

# Titanic CSV var mı kontrol et
ls -lh data/raw/titanic.csv

# İşlenmiş veri var mı kontrol et
ls -lh data/processed/

# MLflow run'ları var mı kontrol et
ls -lh mlruns/
```

## Pratik İpuçları

### Override Kullanımı
Override ile config dosyasını değiştirmeden farklı parametreler deneyebilirsiniz:

```bash
# Sadece n_estimators değiştir
python -m src.models.train --config configs/config_titanic.yaml --run-name rf_300 \
    --override '{"model.params.n_estimators":300}'

# Max_depth değiştir
python -m src.models.train --config configs/config_titanic.yaml --run-name rf_deep \
    --override '{"model.params.max_depth":20}'

# Birden fazla parametre değiştir
python -m src.models.train --config configs/config_titanic.yaml --run-name rf_tuned \
    --override '{"model.params.n_estimators":300,"model.params.max_depth":15,"model.params.min_samples_split":10}'
```

### Tag Kullanımı
Config'de tanımlı taglar otomatik eklenir, ancak kod içinde de ekleyebilirsiniz:

```python
mlflow.set_tag("developer", "Semih")
mlflow.set_tag("experiment_type", "hyperparameter_tuning")
```

### Run Karşılaştırma
MLflow UI'da:
1. İki veya daha fazla run seçin
2. "Compare" butonuna tıklayın
3. Parallel Coordinates Plot'ta parametrelerin metriklere etkisini görün
