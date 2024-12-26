
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

#### Model

```
  main.SentimentAnalyzer
```

| İsim | Tip     | Açıklama                |
| :-------- | :------- | :------------------------- |
| `dataset*` | `tuple` |Dört elemana parçanlanmış veriseti  |
| `tokenize*` | `function` |TextVectorization kullanılır |
| `most_common_words` | `function` |En çok kullanılan 10 kelime yazdırılır, 1000 kelime döndürülür |
| `build_layers` | `function` |Modeli oluşturur ya da yükler |
| `performance_info` | `function` |Loss, Accuarcy değerlerini gösterir |
| `predict_test_review` | `function` |Test verilerini sınar |
| `predict_review` | `function` |Gelen değerin negatifliğini yorumlar |
| `save` | `function` |Modeli kaydeder |

  
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

  
## Ekran Görüntüleri

![Uygulama Ekran Görüntüsü](https://i.hizliresim.com/smzp8wt.jpg)

  
## Kullanılan Teknolojiler

**Interface:** Streamlit

**Background:** Python, Tensorflow, Numpy, Pandas

  