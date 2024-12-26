
# Duygu Analizi: Temel Proje

İkili durum sınıflandırmasını kullanarak, istenen sitede bulunan ilgili ürünün yorumlarını (değerlendirmelerini) olumlu/olumsuz biçiminde sınıflandıracak bir yazılım amaçlanmaktadır.

[Kullanılan veriseti hazır alınmıştır.](https://www.kaggle.com/datasets/burhanbilenn/turkish-customer-reviews-for-binary-classification)
## Sınıflar

#### Veriseti Manipülasyonu

```
  main.Dataset
```

| İsim | Tip     | Açıklama                |
| :-------- | :------- | :------------------------- |
| `datasetPath*` | `string` |Verisetinizin konumu |
| `trainDataPercentage*` | `int` |Eğitim alanının yüzdeliği |
| `get_reviews_and_labels` | `function` |Verisetini test ve eğitim, yorum ve durum olarak dörde ayırır |


  
## Bilgisayarınızda Çalıştırın

Projeyi klonlayın

```bash
  git clone https://github.com/hdenizkaraman/sentiment-analyze-example
```

Proje dizinine gidin

```bash
  cd sentiment-analyze-example
```


Gerekli paketleri yükleyin

```bash
  pip3 install -r requirements.txt
```

Sunucuyu çalıştırın

```bash
  python3 run.py
```

  