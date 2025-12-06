# 🫁 Pnömoni Tespiti Uygulaması (Chest X-Ray)

Bu proje, göğüs röntgeni (Chest X-Ray) görüntülerinden **pnömoni (zatürre)** tespiti yapmak için
**DenseNet121 tabanlı bir derin öğrenme modeli** ve **Streamlit** arayüzü içermektedir.  
Uygulama, yüklenen X-Ray görüntüsü için pnömoni olasılığını hesaplayıp sonucu **“Normal / Pnömoni”**
şeklinde göstermektedir.

> ⚠️ Bu sistem **klinik tanı aracı değildir**.  
> Doktor muayenesi ve klinik değerlendirme yerine geçmez, yalnızca **karar destek** amaçlıdır.

---

## 📁 Proje Yapısı

```text
pneumonia-app-2/
│  app.py                 # Streamlit arayüzü
│  README.md
│  requirements.txt
│  .gitignore
│
├─pneumonia_streamlit_model/   # TensorFlow SavedModel (DenseNet121 tabanlı model)
│   assets/
│   variables/
│   saved_model.pb
│
├─notebooks/                   # (İsteğe bağlı, Colab dosyaları için)
│   chest_xray_eda.ipynb
│   pneumonia_densenet_model.ipynb
│
└─images/                      # (İsteğe bağlı, grafik ve ekran görüntüleri için)
    app_screenshot.png
    confusion_matrix.png
    roc_curve.png
