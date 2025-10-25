# 🎓 MLflow Eğitimi — Süper Detaylı Ders Anlatım Kılavuzu

> **Bu dosya**: Her kod satırını, her terminal komutunu, MLflow UI'da neyin oluştuğunu, `mlruns/` klasöründe hangi dosyaların nerede olduğunu ADIM ADIM açıklıyor.

---

## 📋 Ön Hazırlık (Ders Öncesi - 5 dk)

### Terminal Kontrol:
```bash
cd /Users/semihbekdas/Documents/mlflowegitim
source .venv/bin/activate
which python  # Çıktı: /Users/semihbekdas/Documents/mlflowegitim/.venv/bin/python olmalı
ls data/raw/titanic.csv  # Dosya var mı kontrol
```

### Temiz Başlangıç (Opsiyonel - Eski run'ları sil):
```bash
rm -rf mlruns artifacts data/processed .pytest_cache
```

### Ekran Düzeni:
- **Sol**: VS Code (kodlar)
- **Sağ üst**: Terminal (komutlar)
- **Sağ alt**: Tarayıcı (MLflow UI)

---

## 🎯 BÖLÜM 1: Teorik Giriş + Proje Yapısı (10 dk)

### Sunum Slaytları (MLflow Nedir?)
- Problem: Jupyter'da kaybolmuş sonuçlar, hatırlanamayan parametreler
- MLflow tanımı: Açık kaynak MLOps platformu
- 4 bileşen: **Tracking** ✅, **Projects** ✅, **Models** ✅, ~~Registry~~ (SQL gerekir, bizde yok)
- Faydalar: Tekrarlanabilirlik, karşılaştırma, izlenebilirlik

### Proje Yapısını Göster:
```bash
tree -L 2
```

**Ekranda göster:**
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

**Mesaj:** "3 ana modül: veri hazırlama → eğitim → değerlendirme. Hepsini MLflow ile izleyeceğiz."

---

## 🎯 BÖLÜM 2: Config Dosyası Detaylı İnceleme (7 dk)

### Dosya: `configs/config_titanic.yaml`

**VS Code'da aç, satır satır göster:**

```yaml
experiment_name: mlflow_egitimi      # MLflow experiment adı
tracking_uri: file:./mlruns          # Kayıtlar lokal mlruns/ klasörüne
random_state: 42                     # Tekrarlanabilirlik seed'i
```
**ANLAT:** 
- "MLflow tracking'i `mlruns/` klasörüne kaydedecek (production'da SQL olabilir)"
- "Experiment adı: `mlflow_egitimi` - tüm run'lar bunun altında"

---

```yaml
data:
  source: titanic                    # Veri seti tipi
  csv_path: data/raw/titanic.csv     # Ham veri yolu
  test_size: 0.2                     # %80 train, %20 test
  stratify: true                     # Sınıf dengesi korunsun (survived: 0/1)
  processed_dir: data/processed      # İşlenmiş veri nereye?
```
**ANLAT:** 
- "`stratify=true`: Survived kolonundaki 0/1 oranı train/test'te aynı olsun"
- "`test_size=0.2`: 891 satır → 712 train, 179 test"

---

```yaml
model:
  name: RandomForestClassifier       # Varsayılan model (override ile değiştirilebilir)
  params:
    n_estimators: 200                # Ağaç sayısı
    max_depth: 10                    # Maksimum derinlik
    random_state: 42
```
**ANLAT:** 
- "Varsayılan: RandomForest, 200 ağaç, depth=10"
- "Bu parametreleri `--override` ile komut satırından değiştireceğiz"

---

```yaml
train:
  autolog: false                     # Manuel loglama (daha fazla kontrol)
  run_name: rf_titanic               # MLflow run adı (override edilebilir)
  tags:
    proje: mlops-egitim              # Custom tag'lar (filtreleme için)
    veri: titanic
```
**ANLAT:** 
- "`autolog=false`: Parametreleri biz manuel logluyoruz"
- "Tag'lar ile run'ları kategorize edebiliriz: 'proje=mlops-egitim olan run'ları göster'"

---

```yaml
evaluate:
  save_confusion_matrix: true        # Confusion matrix PNG'si kaydet
  top_n_samples_preview: 5           # İlk 5 tahmin örneğini artifact olarak kaydet
```
**ANLAT:** 
- "Confusion matrix: Hangi sınıfları karıştırdığını gösterir"
- "Preview: Örnek tahminleri görebiliriz (gerçek vs tahmin)"

---

**Mesaj:** "Tüm config burada. `--override` ile değiştirmeden farklı deneyler yapabiliriz."

---

## 🎯 BÖLÜM 3: Veri Hazırlama Kodu — Detaylı İnceleme (18 dk)

### Dosya: `src/data/make_dataset.py`

---

### 📌 3.1: CSV Yükleme ve Validasyon

```python
def _load_titanic_dataframe(csv_path: str) -> pd.DataFrame:
    """Kaggle Titanic verisini yükler ve kolon isimlerini normalize eder."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Titanic CSV bulunamadı: {csv_path}")
```
**ANLAT:** "CSV var mı kontrol ediyor, yoksa hata veriyor."

---

```python
    df = pd.read_csv(csv_path)
    cols = {c: c.strip().lower() for c in df.columns}  # "Survived " → "survived"
    df.rename(columns=cols, inplace=True)
```
**ANLAT:** 
- "Kolon isimlerini normalize ediyor: `Survived` → `survived`"
- "Boşlukları temizliyor: ` Age ` → `age`"
- "CSV'ler her zaman tutarlı değil, bu güvenlik önlemi"

---

```python
    required = {"survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")
```
**ANLAT:** 
- "8 zorunlu kolon olmalı"
- "Eksik varsa hata: Yanlış veri seti yüklendi kontrolü"

---

### 📌 3.2: Feature Engineering Fonksiyonu

```python
def _process_titanic_features(df: pd.DataFrame, stats: Dict | None = None):
    """Feature engineering + eksik değer doldurma"""
    processed = df.copy()  # Orijinali bozmuyoruz
```
**ANLAT:** "Kopya alıyoruz, orijinal veriyi değiştirmiyoruz."

---

```python
    if stats is None:  # İlk çağrı (train)
        stats = {}
        stats["age_median"] = float(processed["age"].median(skipna=True))
        stats["fare_median"] = float(processed["fare"].median(skipna=True))
        embarked_mode_series = processed["embarked"].dropna()
        stats["embarked_mode"] = embarked_mode_series.mode().iloc[0] if not embarked_mode_series.empty else "S"
        stats["sex_fallback"] = processed["sex"].dropna().mode().iloc[0] if not processed["sex"].dropna().empty else "male"
```
**ANLAT (ÇOK ÖNEMLİ!):** 
- "**stats sadece train'den hesaplanır!**"
- "`age_median`: Train'deki yaş ortancası (örn: 28.0)"
- "`fare_median`: Train'deki bilet fiyatı ortancası (örn: 14.45)"
- "`embarked_mode`: En sık liman (S, C, Q'dan biri)"
- "Bu değerler test setine de uygulanacak → **Data leakage önlenir!**"

---

```python
    # Eksik değerleri train'den gelen istatistiklerle doldur
    processed["age"] = processed["age"].fillna(stats["age_median"])
    processed["fare"] = processed["fare"].fillna(stats["fare_median"])
    processed["embarked"] = processed["embarked"].fillna(stats["embarked_mode"])
    processed["sex"] = processed["sex"].fillna(stats["sex_fallback"])
```
**ANLAT:** "Eksik yaşlar train medyanı ile dolduruluyor (test sızıntısı yok!)."

---

```python
    # Kategorik → Sayısal encoding
    processed["sex"] = processed["sex"].str.lower().str.strip()  # "Male " → "male"
    processed["embarked"] = processed["embarked"].str.upper().str.strip()  # "s" → "S"
    
    sex_map = {"male": 0, "female": 1}
    processed["sex_encoded"] = processed["sex"].map(sex_map).fillna(0).astype(int)
    
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    processed["embarked_encoded"] = processed["embarked"].map(embarked_map).fillna(0).astype(int)
```
**ANLAT:** 
- "Makine öğrenmesi sayısal veri ister"
- "male=0, female=1 olarak kodluyoruz"
- "S=0 (Southampton), C=1 (Cherbourg), Q=2 (Queenstown)"

---

```python
    # Yeni feature'lar
    processed["family_size"] = processed["sibsp"].astype(int) + processed["parch"].astype(int) + 1
    processed["is_alone"] = (processed["family_size"] == 1).astype(int)
```
**ANLAT:** 
- "`family_size`: Aile büyüklüğü (kardeş + ebeveyn + kendisi)"
  - Örn: sibsp=1, parch=0 → family_size=2
- "`is_alone`: Yalnız mı? 1=evet, 0=hayır"
- "Bu feature'lar survival tahmininde yardımcı olabilir!"

---

```python
    bins = [-np.inf, 16, 30, 50, np.inf]
    labels = [0, 1, 2, 3]
    processed["age_group_encoded"] = pd.cut(processed["age"], bins=bins, labels=labels).astype(int)
```
**ANLAT:** 
- "Yaşları gruplara ayırıyoruz:"
  - 0: 0-16 (çocuk)
  - 1: 16-30 (genç yetişkin)
  - 2: 30-50 (orta yaş)
  - 3: 50+ (yaşlı)
- "Modeller için yaş grupları bazen sürekli yaştan daha iyi çalışır"

---

```python
    feature_cols = [
        "pclass", "sex_encoded", "age", "sibsp", "parch", "fare",
        "embarked_encoded", "family_size", "is_alone", "age_group_encoded"
    ]
    return processed[feature_cols].copy(), stats
```
**ANLAT:** "10 feature seçtik. Bu kolonlar modele girecek."

---

### 📌 3.3: Data Leakage Önlemi (ÇOK ÖNEMLİ BÖLÜM!)

```python
def prepare_datasets(cfg: Dict):
    # ...veri yükleme...
    
    # 1️⃣ ÖNCE train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        feature_source,  # Ham feature'lar
        y,               # Target (survived)
        test_size=0.2,
        random_state=42,
        stratify=y       # Sınıf dengesi korunsun
    )
```
**ANLAT:** "İlk adım: Veriyi bölüyoruz. Henüz hiçbir işlem yapılmadı."

---

```python
    # 2️⃣ Stats SADECE train'den hesapla
    X_train, stats = _process_titanic_features(X_train_raw)  
    # stats = {"age_median": 28.0, "fare_median": 14.45, ...}
```
**ANLAT:** 
- "Train setinden istatistikler hesaplanıyor"
- "Test seti bu aşamada hiç görülmüyor!"

---

```python
    # 3️⃣ Test'e aynı stats'ı uygula
    X_test, _ = _process_titanic_features(X_test_raw, stats)  
    # Test'teki eksik yaşlar train'den gelen 28.0 ile dolduruluyor
```
**ANLAT (VURGULA!):** 
- "❌ **YANLIŞ YOL**: Tüm veriyi işle → sonra split et"
  - Bu durumda test bilgisi train'e sızar (data leakage!)
  - Model test setini zaten görmüş gibi olur
  
- "✅ **DOĞRU YOL**: Önce split et → train'den istatistik → test'e uygula"
  - Model test setini hiç görmeden eğitilir
  - Gerçek dünya senaryosuna uygun

---

```python
    # CSV'lere kaydet
    X_train.to_csv(processed_path / "X_train.csv", index=False)
    X_test.to_csv(processed_path / "X_test.csv", index=False)
    y_train.to_csv(processed_path / "y_train.csv", index=False)
    y_test.to_csv(processed_path / "y_test.csv", index=False)
    
    return X_train, X_test, y_train, y_test
```
**ANLAT:** "İşlenmiş veriyi kaydediyoruz. Her eğitimde tekrar işlemeye gerek yok."

---

### 🚀 Komutu Çalıştır:

```bash
python -m src.data.make_dataset --config configs/config_titanic.yaml
```

**Beklenen Çıktı:**
```
İşlenmiş veri kaydedildi: data/processed
```

**Dosyaları Kontrol Et:**
```bash
ls -lh data/processed/
```
**Çıktı:**
```
X_train.csv  (712 satır, 10 kolon)
X_test.csv   (179 satır, 10 kolon)
y_train.csv  (712 satır)
y_test.csv   (179 satır)
```

**CSV İçeriğini Göster:**
```bash
head -3 data/processed/X_train.csv
```
**Çıktı:**
```
pclass,sex_encoded,age,sibsp,parch,fare,embarked_encoded,family_size,is_alone,age_group_encoded
3,0,22.0,1,0,7.25,0,2,0,1
1,1,38.0,1,0,71.2833,1,2,0,2
```

**ANLAT:** "Feature'lar hazır: sex_encoded=0 (male), family_size=2, age_group=1 (16-30 yaş)"

---

## 🎯 BÖLÜM 4: Model Eğitimi Kodu — Satır Satır (25 dk)

### Dosya: `src/models/train.py`

---

### 📌 4.1: MLflow Setup

```python
def train(cfg: Dict, run_name: str | None = None):
    tracking_uri = cfg.get("tracking_uri", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
```
**ANLAT:** 
- "`mlflow.set_tracking_uri`: MLflow kayıtları nereye?"
- "Bizde: `file:./mlruns` → lokal klasör (otomatik oluşur)"
- "Production'da: `http://mlflow-server:5000` olabilir (SQL backend)"

---

```python
    mlflow.set_experiment(cfg.get("experiment_name", "mlflow_egitimi"))
```
**ANLAT:** 
- "**Experiment**: Birden fazla run'ı gruplayan kategori"
- "Örnek: 'titanic_project', 'diabetes_prediction' gibi"
- "Bizim experiment adımız: `mlflow_egitimi`"
- "Tüm run'lar bu experiment altında görünecek"

---

```python
    autolog = bool(cfg.get("train", {}).get("autolog", False))
    if autolog:
        mlflow.sklearn.autolog(log_models=False)
```
**ANLAT:** 
- "**Autolog**: sklearn parametrelerini otomatik loglar"
- "`log_models=False`: Modeli otomatik loglamıyor (çünkü signature ekleyeceğiz)"
- "Bizde `autolog=false` (config'te), tüm loglama manuel"

---

### 📌 4.2: Veri Yükleme

```python
def load_processed_or_raw(cfg: Dict):
    processed_dir = cfg.get("data", {}).get("processed_dir", "data/processed")
    p = Path(processed_dir)
    
    if (p / "X_train.csv").exists():
        # İşlenmiş veri varsa oku
        X_train = pd.read_csv(p / "X_train.csv")
        X_test = pd.read_csv(p / "X_test.csv")
        y_train = pd.read_csv(p / "y_train.csv").squeeze("columns")
        y_test = pd.read_csv(p / "y_test.csv").squeeze("columns")
        return X_train, X_test, y_train, y_test
    
    # Yoksa üret
    X_train, X_test, y_train, y_test = prepare_datasets(cfg)
    return X_train, X_test, y_train, y_test
```
**ANLAT:** 
- "İşlenmiş veri var mı kontrol ediyor"
- "Varsa okuyor (hızlı)"
- "Yoksa `make_dataset.prepare_datasets()` çağrılıp üretiliyor"

---

### 📌 4.3: Model Kurulumu

```python
def build_model(cfg: Dict) -> Tuple[Pipeline, Dict]:
    model_name = cfg.get("model", {}).get("name", "RandomForestClassifier")
    params = cfg.get("model", {}).get("params", {})
```
**ANLAT:** "Config'den model adı ve parametreleri alıyor."

---

```python
    if model_name == "RandomForestClassifier":
        allowed = {"n_estimators", "max_depth", "max_features", "min_samples_split", "min_samples_leaf", "bootstrap", "random_state"}
        rf_params = {k: v for k, v in params.items() if k in allowed and v is not None}
        model = RandomForestClassifier(**rf_params)
        steps = [("model", model)]  # Scaling YOK (RandomForest'e gerek yok)
        used_params = rf_params
```
**ANLAT:** 
- "**Parametre filtrelemesi**: Sadece allowed setindekiler geçer (güvenlik)"
- "RandomForest için StandardScaler gerekmez (ağaç tabanlı)"
- "Pipeline: Direkt model"
- "`used_params`: MLflow'a loglanacak"

---

```python
    elif model_name == "LogisticRegression":
        allowed = {"penalty", "C", "solver", "max_iter", "random_state", "multi_class"}
        lr_params = {k: v for k, v in params.items() if k in allowed and v is not None}
        lr_params.setdefault("max_iter", 2000)  # Varsayılan
        lr_params.setdefault("random_state", 42)
        model = LogisticRegression(**lr_params)
        steps = [("scaler", StandardScaler()), ("model", model)]  # Scaling VAR!
        used_params = lr_params
```
**ANLAT:** 
- "LogisticRegression için **StandardScaler gerekli**"
- "Pipeline: Önce scaling, sonra model"
- "`StandardScaler`: Feature'ları ortalama=0, std=1 yapıyor"

---

```python
    elif model_name == "KNeighborsClassifier":
        allowed = {"n_neighbors", "weights", "metric"}
        knn_params = {k: v for k, v in params.items() if k in allowed and v is not None}
        model = KNeighborsClassifier(**knn_params)
        steps = [("scaler", StandardScaler()), ("model", model)]
        used_params = knn_params
```
**ANLAT:** "KNN için de scaling gerekli (mesafe tabanlı)."

---

```python
    return Pipeline(steps), used_params
```
**ANLAT:** "Pipeline döndürüyor: `pipe.fit(X_train, y_train)` ile eğitiliyor."

---

### 📌 4.4: MLflow Run Başlatma

```python
    base_tags = {"run_scope": "train"}
    with mlflow.start_run(
        run_name=run_name or cfg.get("train", {}).get("run_name"),
        tags=base_tags
    ):
```
**ANLAT:** 
- "`mlflow.start_run`: Yeni bir deney kaydı başlatıyor"
- "`run_name`: UI'da görünecek isim (örn: `rf_titanic_default`)"
- "`tags`: run_scope='train' → evaluation run'larından ayırt etmek için"
- "**with bloğu**: Run otomatik kapanır (hata olsa bile)"

---

### 📌 4.5: Parametre Loglama

```python
        # Params
        mlflow.log_param("model_name", cfg.get("model", {}).get("name"))
        for k, v in used_params.items():
            mlflow.log_param(f"model.{k}", v)
```
**ANLAT:** 
- "`mlflow.log_param`: Parametreleri MLflow'a kaydediyor"
- "Örnek loglar:"
  ```
  model_name: RandomForestClassifier
  model.n_estimators: 200
  model.max_depth: 10
  model.random_state: 42
  ```
- "Bu parametreler UI'da görünecek ve karşılaştırılabilecek"

---

```python
        # Tags
        for tk, tv in (cfg.get("train", {}).get("tags", {}) or {}).items():
            mlflow.set_tag(tk, tv)
```
**ANLAT:** 
- "Config'deki custom tag'ları ekliyor:"
  ```
  proje: mlops-egitim
  veri: titanic
  ```
- "UI'da tag'lere göre filtreleme yapabilirsiniz"

---

### 📌 4.6: Model Eğitimi

```python
        # Train
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0
```
**ANLAT:** 
- "Modeli eğitiyor: RandomForest 200 ağaç üretiyor"
- "Eğitim süresini ölçüyor (benchmarking için)"

---

### 📌 4.7: Metrik Loglama

```python
        # Metrics
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1))
        mlflow.log_metric("train_time_sec", float(train_time))
```
**ANLAT:** 
- "Test setinde tahmin yapıyor (179 örnek)"
- "**Accuracy**: Doğru tahmin oranı (örn: 0.81 → %81)"
- "**F1 macro**: Sınıflar arası dengeli metrik (0-1 arası)"
- "`mlflow.log_metric`: Metrikleri MLflow'a kaydediyor"

---

### 📌 4.8: Confusion Matrix Artifact

```python
        # Confusion matrix artifact
        if cfg.get("evaluate", {}).get("save_confusion_matrix", True):
            import matplotlib.pyplot as plt
            
            cm = confusion_matrix(y_test, y_pred)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_pct = np.divide(cm, row_sums, where=row_sums != 0)  # Satır bazlı %
```
**ANLAT:** 
- "Confusion matrix oluşturuyor"
- "Satır bazlı yüzdeye çeviriyor: Her gerçek sınıfın tahmini nasıl dağılmış?"

---

```python
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm_pct, cmap="Blues")
            ax.set_title("Confusion Matrix — Normalize (Row %)")
            ax.set_xlabel("Tahmin")
            ax.set_ylabel("Gerçek")
            # ... sayıları yaz ...
            
            fig.savefig("artifacts/confusion_matrix.png", dpi=150)
            mlflow.log_artifact("artifacts/confusion_matrix.png")
            plt.close(fig)
```
**ANLAT:** 
- "Matplotlib ile görselleştiriyor"
- "PNG olarak kaydediyor: `artifacts/confusion_matrix.png`"
- "`mlflow.log_artifact`: Dosyayı MLflow'a yükl artifact olarak kayıt"
- "UI'dan indirilebilir olacak"

---

### 📌 4.9: Model Kaydı (en önemli kısım!)

```python
        # Log model
        signature = infer_signature(X_train, pipe.predict(X_train))
        input_example = X_train.head(min(len(X_train), 5))
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
```
**ANLAT:** 
- "`infer_signature`: Modelin input/output şemasını otomatik algılar"
  ```yaml
  inputs: [{"name": "pclass", "type": "long"}, {"name": "sex_encoded", "type": "long"}, ...]
  outputs: [{"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1]}}]
  ```
- "`input_example`: İlk 5 satır örnek olarak kaydediliyor (API testi için)"
- "`artifact_path='model'`: Model `mlruns/.../artifacts/model/` altına kaydedilir"
- "**Signature sayesinde production'da input validasyonu yapılır!**"

---

```python
        rid = mlflow.active_run().info.run_id
        print(f"Run finished: run_id={rid} | acc={acc:.4f} | f1_macro={f1:.4f}")
```
**ANLAT:** 
- "Run ID'yi yazdırıyor (örn: `6f3c01268e0d46c5ac73d2dc8ba9efbd`)"
- "Bu ID ile modeli yükleyebiliriz: `runs:/<run_id>/model`"

---

### 🚀 Komutu Çalıştır:

```bash
python -m src.models.train --config configs/config_titanic.yaml --run-name rf_titanic_default
```

**Beklenen Çıktı:**
```
Run finished: run_id=6f3c01268e0d46c5ac73d2dc8ba9efbd | acc=0.8101 | f1_macro=0.7912
```

**Dosya Yapısı (mlruns klasörü):**
```bash
ls -la mlruns/
```
**Çıktı:**
```
604750718302595916/    ← Experiment ID (mlflow_egitimi)
├── meta.yaml
└── 6f3c01268e0d46c5ac73d2dc8ba9efbd/    ← Run ID
    ├── meta.yaml
    ├── params/
    │   ├── model_name
    │   ├── model.n_estimators
    │   ├── model.max_depth
    │   └── model.random_state
    ├── metrics/
    │   ├── accuracy
    │   ├── f1_macro
    │   └── train_time_sec
    ├── tags/
    │   ├── run_scope
    │   ├── proje
    │   └── veri
    └── artifacts/
        ├── confusion_matrix.png
        └── model/
            ├── MLmodel          ← Model metadata (signature burada!)
            ├── model.pkl        ← Pickle dosyası
            ├── conda.yaml
            ├── python_env.yaml
            └── requirements.txt
```

**Bir dosyayı göster:**
```bash
cat mlruns/604750718302595916/6f3c01268e0d46c5ac73d2dc8ba9efbd/params/model_name
```
**Çıktı:** `RandomForestClassifier`

```bash
cat mlruns/604750718302595916/6f3c01268e0d46c5ac73d2dc8ba9efbd/metrics/accuracy
```
**Çıktı:** `1734567890123 0.8101 0` (timestamp, değer, step)

**ANLAT:** "Her parametre, metrik, artifact ayrı dosya olarak saklanıyor. UI bunları okuyor."

---

## 🎯 BÖLÜM 5: MLflow UI — Detaylı İnceleme (20 dk)

### 🚀 UI'ı Başlat:

**Yeni terminal sekmesi aç (Ctrl+Shift+`)**
```bash
python -m mlflow ui --backend-store-uri file:./mlruns --port 5000
```

**Tarayıcıda aç:**
```
http://127.0.0.1:5000
```

---

### 📌 5.1: Experiments Sayfası

**Ekranda göster:**
1. Sol menüde **"Experiments"** tıkla
2. **"mlflow_egitimi"** experiment'ini seç
3. Run listesi görünecek: `rf_titanic_default`

**ANLAT:** "Experiment: Run'ları gruplayan kategori. Tüm Titanic deneylerimiz burada."

---

### 📌 5.2: Run Detayları

**Run'a tıkla (rf_titanic_default)**

**Üst sekmeleri göster:**
- **Overview**: Genel bilgi
- **Metrics**: Grafikler
- **Parameters**: Parametre listesi
- **Tags**: Tag'lar
- **Artifacts**: Dosyalar

---

#### Metrics Sekmesi:
```
📊 Metrics:
   accuracy       0.8101
   f1_macro       0.7912
   train_time_sec 1.234
```
**ANLAT:** "Metrikler kaydedildi. Birden fazla run'ı seçince grafik oluşur."

---

#### Parameters Sekmesi:
```
📝 Parameters:
   model_name           RandomForestClassifier
   model.n_estimators   200
   model.max_depth      10
   model.random_state   42
```
**ANLAT:** "Tüm parametreler burada. Hangi ayarla bu sonuç elde edildi görebiliyoruz."

---

#### Tags Sekmesi:
```
🏷️ Tags:
   run_scope     train
   proje         mlops-egitim
   veri          titanic
   mlflow.user   semihbekdas
```
**ANLAT:** 
- "`run_scope=train`: Evaluation run'larından ayırt etmek için"
- "Custom tag'lar: Filtreleme ve organizasyon için"

---

#### Artifacts Sekmesi:

**Klasör yapısını göster:**
```
artifacts/
├── confusion_matrix.png    ← Görsel
└── model/
    ├── MLmodel            ← Model metadata
    ├── model.pkl          ← Pickle dosyası
    ├── conda.yaml
    ├── python_env.yaml
    ├── requirements.txt
    └── input_example.json ← Örnek input
```

**confusion_matrix.png'yi indir ve göster:**
- "Hangi sınıfları karıştırdığını gösteriyor"
- "Survived=0 tahminlerinde doğruluk %85, Survived=1'de %72 gibi"

---

**MLmodel dosyasını aç (tarayıcıda tıkla):**
```yaml
artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.11.9
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.4.2
mlflow_version: 2.14.1
model_size_bytes: 12345678
model_uuid: abcd-1234-efgh-5678
run_id: 6f3c01268e0d46c5ac73d2dc8ba9efbd
signature:
  inputs: '[{"name": "pclass", "type": "long"}, {"name": "sex_encoded", "type": "long"}, ...]'
  outputs: '[{"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1]}}]'
  params: null
utc_time_created: '2025-01-15 10:30:45.123456'
```

**ANLAT:** 
- "**signature**: Modelin input/output şeması"
- "**flavors**: Model nasıl yüklenecek? (sklearn, python_function)"
- "**run_id**: Hangi run'a ait?"
- "Production'da API bu signature'a göre input validate eder!"

---

### 📌 5.3: Run Karşılaştırma Hazırlığı

**Mesaj:** "Şimdi ikinci bir model eğitip karşılaştıralım!"

---

## 🎯 BÖLÜM 6: İkinci Model — LogisticRegression Override (15 dk)

### 🚀 Komutu Çalıştır (İLK terminal sekmesinde):

```bash
python -m src.models.train --config configs/config_titanic.yaml --run-name lr_titanic \
    --override '{"model.name":"LogisticRegression","model.params.max_iter":2000}'
```

**NE OLUYOR?**
1. Config'i okuyor
2. Override ile şunları değiştiriyor:
   ```json
   {
     "model.name": "LogisticRegression",
     "model.params.max_iter": 2000
   }
   ```
3. `build_model()` fonksiyonu LogisticRegression dalına giriyor
4. Pipeline: StandardScaler + LogisticRegression

**Beklenen Çıktı:**
```
Run finished: run_id=d96cb9318eb64119bad7b7b09c4340ac | acc=0.8212 | f1_macro=0.8103
```

**ANLAT:** 
- "Config dosyasını değiştirmedik!"
- "Sadece komut satırından override verdik"
- "LogisticRegression biraz daha iyi: %82.1 accuracy"

---

### 📌 mlruns Klasörü Değişimi:

```bash
ls mlruns/604750718302595916/
```
**Çıktı:**
```
6f3c01268e0d46c5ac73d2dc8ba9efbd/   ← RandomForest run
d96cb9318eb64119bad7b7b09c4340ac/   ← LogisticRegression run (YENİ!)
```

**ANLAT:** "Her run ayrı bir klasör. UI otomatik algılıyor."

---

### 📌 UI'da Karşılaştırma:

**Tarayıcıda (UI'ı yenile):**

1. **İki run'ı seç (checkbox):**
   - `rf_titanic_default`
   - `lr_titanic`

2. **"Compare" butonuna tıkla**

3. **Karşılaştırma Tablosu:**
```
| Run Name           | accuracy | f1_macro | model_name             | model.n_estimators | model.max_iter |
|--------------------|----------|----------|------------------------|-------------------|----------------|
| rf_titanic_default | 0.8101   | 0.7912   | RandomForestClassifier | 200               | -              |
| lr_titanic         | 0.8212   | 0.8103   | LogisticRegression     | -                 | 2000           |
```

**ANLAT:** 
- "LR biraz daha iyi (0.8212 vs 0.8101)"
- "Parametre farkları net görünüyor"

4. **Parallel Coordinates Plot:**
   - X ekseni: Parametreler (n_estimators, max_iter)
   - Y ekseni: Metrikler (accuracy)
   - "Parametre değişimlerinin metriğe etkisini görselleştiriyor"

**ANLAT:** "UI sayesinde hızlıca karar verebiliyoruz: Hangi model daha iyi?"

---

### 📌 Bonus: KNN Deneyelim (Zamanınız varsa)

```bash
python -m src.models.train --config configs/config_titanic.yaml --run-name knn_titanic \
    --override '{"model.name":"KNeighborsClassifier","model.params.n_neighbors":7}'
```

**Çıktı:** `acc=0.7989` (daha düşük)

**UI'da 3 modeli karşılaştırın:**
- RF: 0.8101
- LR: 0.8212 ← **KAZANAN**
- KNN: 0.7989

---

## 🎯 BÖLÜM 7: Model Değerlendirme — Satır Satır (15 dk)

### Dosya: `src/models/evaluate.py`

---

### 📌 7.1: MLflow Client ile Son Run'ı Bulma

```python
def main():
    cfg = read_yaml(DEFAULT_CONFIG)
    mlflow.set_tracking_uri(cfg.get("tracking_uri", "file:./mlruns"))
    experiment_name = cfg.get("experiment_name", "mlflow_egitimi")
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
```
**ANLAT:** "Experiment'i buluyoruz (mlflow_egitimi)."

---

```python
    client = MlflowClient()
    filter_string = "tags.run_scope = 'train'"
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
```
**ANLAT (ÇOK ÖNEMLİ!):** 
- "`filter_string`: Sadece tag'i `run_scope='train'` olanları getir"
- "**Neden?** Evaluation run'ları kendini yüklememesin!"
- "`order_by=['start_time DESC']`: En son yapılan en başta"
- "`max_results=1`: Sadece en yenisini al"
- "Sonuç: En son eğitilen model (lr_titanic)"

---

```python
    if not runs:
        raise RuntimeError("No runs found. Train a model first.")
    
    run = runs[0]
    run_id = run.info.run_id  # d96cb9318eb64119bad7b7b09c4340ac
```
**ANLAT:** "Run bulunamazsa hata veriyor (önce model eğit!)."

---

### 📌 7.2: Model Yükleme

```python
    model_uri = f"runs:/{run_id}/model"
    # Örnek: "runs:/d96cb9318eb64119bad7b7b09c4340ac/model"
    
    model = mlflow.sklearn.load_model(model_uri)
    # LogisticRegression pipeline'ı yüklendi (StandardScaler + LR)
```
**ANLAT:** 
- "`model_uri`: MLflow'un model referans formatı"
- "`mlflow.sklearn.load_model`: Modeli bellektenyükle"
- "Pipeline tüm preprocessing ile birlikte gelir (StandardScaler dahil!)"

---

### 📌 7.3: Test Verisi Yükleme

```python
def load_test(cfg):
    p = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    if (p / "X_test.csv").exists():
        X_test = pd.read_csv(p / "X_test.csv")
        y_test = pd.read_csv(p / "y_test.csv").squeeze("columns")
        return X_test, y_test
    
    # Yoksa üret
    _, X_test, _, y_test = prepare_datasets(cfg)
    return X_test, y_test
```
**ANLAT:** "Test setini yüklüyor (daha önce kaydettik)."

---

### 📌 7.4: Tahmin ve Metrik Hesaplama

```python
    X_test, y_test = load_test(cfg)
    y_pred = model.predict(X_test)  # 179 tahmin
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
```
**ANLAT:** "Test setinde tahmin yapıyor, accuracy ve F1 hesaplıyor."

---

### 📌 7.5: Evaluation Run Oluşturma

```python
    with mlflow.start_run(run_name="evaluate_last_run", tags={"run_scope": "evaluation"}):
        mlflow.log_param("evaluated_run_id", run_id)
        mlflow.log_metric("eval_accuracy", float(acc))
        mlflow.log_metric("eval_f1_macro", float(f1))
```
**ANLAT:** 
- "**Yeni bir run** açıyor (evaluation run)"
- "`run_scope='evaluation'`: Train run'larından ayırt etmek için"
- "`evaluated_run_id`: Hangi modeli değerlendirdi?"
- "Metrikler: `eval_accuracy`, `eval_f1_macro`"

---

```python
        # Save sample predictions
        preview_count = int(cfg.get("evaluate", {}).get("top_n_samples_preview", 5))
        preview_df = X_test.head(preview_count).copy()
        preview_df["actual"] = y_test.head(preview_count).values
        preview_df["prediction"] = y_pred[:preview_count]
        path = os.path.join("artifacts", "prediction_preview.csv")
        preview_df.to_csv(path, index=False)
        mlflow.log_artifact(path)
```
**ANLAT:** 
- "İlk 5 tahmini CSV'ye kaydediyor"
- "Gerçek vs Tahmin karşılaştırması"
- "Artifact olarak MLflow'a yüklüyor"

---

```python
    print(f"Evaluated run {run_id}: acc={acc:.4f}, f1={f1:.4f}")
```

---

### 🚀 Komutu Çalıştır:

```bash
python -m src.models.evaluate
```

**Beklenen Çıktı:**
```
Evaluated run d96cb9318eb64119bad7b7b09c4340ac: acc=0.8212, f1=0.8103
```

---

### 📌 mlruns Klasöründe Değişim:

```bash
ls mlruns/604750718302595916/
```
**Çıktı:**
```
6f3c01268e0d46c5ac73d2dc8ba9efbd/   ← RandomForest (train)
d96cb9318eb64119bad7b7b09c4340ac/   ← LogisticRegression (train)
3d33b39b443e4144a099ee7adb223799/   ← Evaluation run (YENİ!)
```

**Evaluation run içeriği:**
```
3d33b39b443e4144a099ee7adb223799/
├── params/
│   └── evaluated_run_id     ← "d96cb9318eb64119bad7b7b09c4340ac"
├── metrics/
│   ├── eval_accuracy        ← 0.8212
│   └── eval_f1_macro        ← 0.8103
├── tags/
│   └── run_scope            ← "evaluation"
└── artifacts/
    └── prediction_preview.csv
```

---

### 📌 UI'da Evaluation Run'ı Göster:

**Tarayıcıda (UI'ı yenile):**

1. Run listesinde `evaluate_last_run` görünecek
2. **Tags** sekmesi: `run_scope=evaluation`
3. **Parameters**: `evaluated_run_id=d96cb...`
4. **Artifacts**: `prediction_preview.csv` indir ve göster:

```csv
pclass,sex_encoded,age,sibsp,parch,fare,embarked_encoded,family_size,is_alone,age_group_encoded,actual,prediction
3,0,22.0,1,0,7.25,0,2,0,1,0,0
1,1,38.0,1,0,71.2833,1,2,0,2,1,1
3,1,26.0,0,0,7.925,0,1,1,1,1,1
1,1,35.0,1,0,53.1,0,2,0,2,1,1
3,0,35.0,0,0,8.05,0,1,1,2,0,0
```

**ANLAT:** 
- "actual=1, prediction=1: Doğru tahmin!"
- "actual=0, prediction=0: Doğru tahmin!"
- "Örnek tahminleri görebiliyoruz"

---

## 🎯 BÖLÜM 8: Özet ve İleri Adımlar (10 dk)

### 🚀 Terminal Geçmişini Göster:

```bash
history | grep python
```
**Çıktı:**
```bash
python -m src.data.make_dataset --config configs/config_titanic.yaml
python -m src.models.train --config configs/config_titanic.yaml --run-name rf_titanic_default
python -m src.models.train --config configs/config_titanic.yaml --run-name lr_titanic --override '{"model.name":"LogisticRegression","model.params.max_iter":2000}'
python -m src.models.evaluate
python -m mlflow ui --backend-store-uri file:./mlruns --port 5000
```

**ANLAT:** "Bu 5 komutla uçtan uca MLflow workflow'u tamamladık!"

---

### ✅ Bugün Öğrendiklerimiz:

```
✅ Config yönetimi (YAML + override)
✅ Data leakage önlemi (split → stats → apply)
✅ Feature engineering (family_size, age_group, encoding)
✅ MLflow Tracking (params, metrics, tags)
✅ Model signature (input/output validation)
✅ Artifact'lar (confusion matrix, prediction preview)
✅ MLflow UI (karşılaştırma, filtreleme)
✅ Model yükleme (mlflow.sklearn.load_model)
✅ Run scope tagging (train vs evaluation)
✅ Override ile parametre deneme
```

---

### 📌 İleri Adımlar:

1. **Hyperparameter Tuning:**
   ```bash
   # GridSearch + MLflow
   for n_est in 50 100 200 300; do
       python -m src.models.train --config configs/config_titanic.yaml \
           --run-name rf_$n_est --override "{\"model.params.n_estimators\":$n_est}"
   done
   # UI'da en iyi n_estimators'ı bulun
   ```

2. **SQL Backend + Model Registry:**
   - PostgreSQL kurulumu
   - `mlflow server --backend-store-uri postgresql://...`
   - Model versiyonlama: Staging → Production

3. **MLflow Projects:**
   - `MLproject` dosyası ile taşınabilir pipeline
   - `mlflow run . -e train`

4. **Model Serving:**
   ```bash
   mlflow models serve -m runs:/<run_id>/model -p 5001
   curl -X POST http://localhost:5001/invocations -H 'Content-Type: application/json' \
       -d '{"inputs": [[3, 0, 22.0, 1, 0, 7.25, 0, 2, 0, 1]]}'
   ```

5. **Docker Deployment:**
   ```bash
   mlflow models build-docker -m runs:/<run_id>/model -n titanic-model
   docker run -p 5001:8080 titanic-model
   ```

---

### 🎯 Soru-Cevap:

**S: MLflow UI kapanınca run'lar kaybolur mu?**
C: Hayır! `mlruns/` klasöründe saklanır, UI tekrar açtığınızda görünür.

**S: Farklı makinede nasıl paylaşırım?**
C: 1) `mlruns/` klasörünü Git'e ekle, 2) SQL backend + S3 kullan.

**S: Production'da nasıl kullanılır?**
C: SQL backend + Model Registry + REST API (mlflow models serve) veya Docker.

**S: Autolog vs Manuel log?**
C: Autolog hızlıdır ama signature ekleyemezsiniz. Production için manuel önerilir.

**S: Run'ları nasıl silerim?**
C: `rm -rf mlruns/<experiment_id>/<run_id>/` veya UI'da "Delete" butonu.

---


### Ders Sırası Komutlar:
```bash
# 1. Veri hazırlama
python -m src.data.make_dataset --config configs/config_titanic.yaml

# 2. RandomForest
python -m src.models.train --config configs/config_titanic.yaml --run-name rf_titanic_default

# 3. MLflow UI (yeni terminal)
python -m mlflow ui --backend-store-uri file:./mlruns --port 5000

# 4. LogisticRegression (ilk terminal)
python -m src.models.train --config configs/config_titanic.yaml --run-name lr_titanic \
    --override '{"model.name":"LogisticRegression","model.params.max_iter":2000}'

# 5. Evaluate
python -m src.models.evaluate
```

