# Pnömoni Tespiti Uygulaması (Chest X-Ray)

## Proje Özeti

Bu proje, göğüs röntgeni (**Chest X-Ray**) görüntülerinden **pnömoni (zatürre)** tespiti yapmak amacıyla geliştirilmiş, **derin öğrenme tabanlı** bir görüntü sınıflandırma uygulamasıdır. Sistem, **DenseNet121 mimarisi** kullanılarak eğitilmiş bir model ve bu modeli son kullanıcıya sunan **Streamlit tabanlı bir web arayüzü** içermektedir.

Uygulama, kullanıcı tarafından yüklenen bir Chest X-Ray görüntüsünü analiz ederek görüntünün **Normal** mi yoksa **Pnömoni** mi olduğunu tahmin eder ve ilgili olasılık değerini kullanıcıya sunar.

---

## Projenin Amacı

* Medikal görüntüler üzerinden pnömoni tespiti yapan bir derin öğrenme modeli geliştirmek
* Transfer learning yaklaşımı kullanarak sınırlı veri ile yüksek performans elde etmek
* Kullanıcı dostu bir web arayüzü ile model çıktılarının görselleştirilmesini sağlamak
* Sağlık alanında yapay zekâ uygulamalarına yönelik bir karar destek sistemi örneği sunmak

---

## Veri Seti

* **Veri Seti Adı:** Chest X-Ray (Pneumonia)
* **Sınıflar:** Normal, Pnömoni
* **Veri Türü:** Gri tonlamalı medikal görüntüler
* **Kaynak:** Herkese açık (public) medikal görüntü veri seti

---

## Yöntem

### 1. Veri Ön İşleme

* Görüntülerin yeniden boyutlandırılması
* Normalizasyon işlemleri
* Etiketlerin sayısal formata dönüştürülmesi
* Eğitim ve doğrulama veri kümelerinin oluşturulması

### 2. Model Mimarisi

* **Temel Model:** ImageNet üzerinde önceden eğitilmiş **DenseNet121**
* Transfer learning uygulanarak temel katmanların dondurulması
* İkili sınıflandırma için özel fully connected katmanlar

### 3. Eğitim ve Değerlendirme

* **Kayıp Fonksiyonu:** Binary Crossentropy
* **Optimizasyon Algoritması:** Adam
* **Değerlendirme Metrikleri:** Accuracy, Confusion Matrix

---

## Uygulama Arayüzü

Eğitilen model, **Streamlit** kullanılarak geliştirilen bir web uygulamasına entegre edilmiştir. Uygulama:

* Kullanıcının X-Ray görüntüsü yüklemesine olanak tanır
* Yüklenen görüntüyü arayüzde gösterir
* Pnömoni olasılığını hesaplar
* Sonucu **Normal / Pnömoni** şeklinde sunar

---

## Kullanılan Teknolojiler

* **Programlama Dili:** Python
* **Derin Öğrenme:** TensorFlow, Keras
* **Model Mimarisi:** DenseNet121
* **Görüntü İşleme:** OpenCV, NumPy
* **Web Arayüzü:** Streamlit
* **Görselleştirme:** Matplotlib

---

## Uyarı

> ⚠️ **Önemli Not**
> Bu uygulama **klinik tanı aracı değildir**.
> Tıbbi teşhis amacıyla kullanılamaz ve doktor muayenesi ile klinik değerlendirme yerine geçmez.
> Yalnızca **eğitsel ve karar destek amaçlı** olarak geliştirilmiştir.

---

## Kısıtlar

* Model performansı kullanılan veri setinin kalitesi ve dağılımına bağlıdır
* Yanlış pozitif ve yanlış negatif tahminler oluşabilir
* Gerçek klinik ortamlar için doğrulanmamıştır

---

## Gelecek Çalışmalar

* ROC-AUC, Precision ve Recall gibi ek metriklerin değerlendirilmesi
* Farklı CNN mimarileri ile performans karşılaştırması
* Veri artırma (data augmentation) teknikleri ile genelleme yeteneğinin artırılması
* Uygulamanın bulut ortamına taşınması

---

## Not

Bu proje, **bilgisayarla görme ve derin öğrenme** alanındaki yetkinlikleri göstermek amacıyla hazırlanmış bir portföy çalışmasıdır.
