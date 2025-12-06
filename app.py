import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go  # Gauge chart için
import cv2

# -------------------------------------------------
# SAYFA AYARLARI
# -------------------------------------------------
st.set_page_config(
    page_title="Pnömoni Tespiti Uygulaması",
    layout="wide",
    page_icon="🫁"
)

# -------------------------------------------------
# MODEL YÜKLEME (SavedModel klasörü)
# -------------------------------------------------
MODEL_DIR = "pneumonia_streamlit_model"  # senin klasör adın

@st.cache_resource(show_spinner="Model yükleniyor...")
def load_saved_model():
    model = tf.saved_model.load(MODEL_DIR)
    return model

saved_model = load_saved_model()
infer = saved_model.signatures["serving_default"]  # tahmin fonksiyonu

# -------------------------------------------------
# SESSION STATE (İSTATİSTİK & GEÇMİŞ)
# -------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []  # {prob_percent, label}

# -------------------------------------------------
# ÜST HEADER + MİNİ KARTLAR
# -------------------------------------------------
top_left, top_right = st.columns([3, 2])

with top_left:
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #0052A2, #0090FF);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;">
      <h1 style="color: white; margin: 0; font-size: 30px;">
        🫁 Pnömoni Tespiti Uygulaması
      </h1>
      <p style="color: #f0f4ff; margin-top: 8px; font-size: 16px;">
        Göğüs röntgeni (X-Ray) görüntülerinden pnömoni olasılığını tahmin eden,
        derin öğrenme tabanlı bir karar destek aracı.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Bu uygulama, **DenseNet tabanlı bir derin öğrenme modeli** kullanarak göğüs röntgenlerinden
    **pnömoni olasılığını** hesaplar.  
    Sonuçlar, **doktor muayenesi ve klinik değerlendirme yerine geçmez**, sadece **destek amaçlıdır**.
    """)

with top_right:
    total_preds = len(st.session_state.prediction_history)
    if total_preds > 0:
        last_pred = st.session_state.prediction_history[-1]
        last_prob = last_pred["prob_percent"]
        last_label = last_pred["label"]
    else:
        last_prob = None
        last_label = "Henüz tahmin yapılmadı"

    st.markdown("""
    <div style="display: flex; gap: 10px; flex-direction: column;">
    """, unsafe_allow_html=True)

    # Kart 1: Toplam Tahmin Sayısı
    st.markdown(f"""
    <div style="
        background: #ffffff;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e6e9f0;">
        <div style="font-size: 12px; color: #6b7280;">Toplam Tahmin Sayısı</div>
        <div style="font-size: 22px; font-weight: 600; color: #111827;">{total_preds}</div>
    </div>
    """, unsafe_allow_html=True)

    # Kart 2: Son Tahmin Skoru
    prob_text = f"{last_prob:.1f}%" if last_prob is not None else "-"
    st.markdown(f"""
    <div style="
        background: #ffffff;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e6e9f0;">
        <div style="font-size: 12px; color: #6b7280;">Son Tahmin Model Skoru</div>
        <div style="font-size: 22px; font-weight: 600; color: #111827;">{prob_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Kart 3: Son Tahmin Sınıfı
    st.markdown(f"""
    <div style="
        background: #ffffff;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e6e9f0;">
        <div style="font-size: 12px; color: #6b7280;">Son Tahmin Sınıfı</div>
        <div style="font-size: 14px; font-weight: 500; color: #111827;">{last_label}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("Hakkında")
st.sidebar.info("""
Bu uygulama:

- Derin öğrenme tabanlı bir sınıflandırma modelidir  
- Göğüs X-Ray görüntülerinden pnömoni olasılığı hesaplar  
- Klinik tanı aracı değildir, karar destek amaçlıdır  
- Staj projesi kapsamında geliştirilmiştir  

Geliştiren: **Ayça Cansu Donduran**
""")

with st.sidebar.expander("🩺 Pnömoni nedir?"):
    st.write("""
Pnömoni (zatürre), akciğer dokusunun enfeksiyona bağlı iltihaplanmasıdır.  
Genellikle bakteri, virüs veya daha nadir olarak mantarlar tarafından oluşturulur.  

Belirtiler arasında:
- Öksürük  
- Ateş, titreme  
- Nefes darlığı  
- Göğüs ağrısı  

bulunabilir. Kesin tanı için klinik muayene, görüntüleme ve laboratuvar bulguları birlikte değerlendirilmelidir.
""")

st.sidebar.header("Nasıl Kullanılır?")
st.sidebar.write("""
1. Göğüs X-Ray görüntüsünü yükleyin  
2. Model görüntüyü analiz edip sınıf tahmini üretir  
3. Sonuç kartında tahmin edilen sınıfı inceleyin  
""")

st.sidebar.header("Teknik Bilgi")
st.sidebar.write("""
- Model türü: **DenseNet tabanlı CNN**  
- Çıktı: `0–1` aralığında pnömoni olasılığı  
- Girdi boyutu: **224×224, 3 kanal (RGB)**  
""")

# -------------------------------------------------
# GÖRÜNTÜ YÜKLEME
# -------------------------------------------------
st.subheader("🎞️ Görüntü Yükleme")
uploaded_file = st.file_uploader(
    "Göğüs X-Ray görüntüsü yükleyin (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# ÖN İŞLEME
# -------------------------------------------------
def preprocess_image(pil_image: Image.Image):
    img_resized = pil_image.resize((224, 224)).convert("RGB")
    img_arr = np.array(img_resized) / 255.0
    img_arr = img_arr.astype("float32")
    img_arr = np.expand_dims(img_arr, axis=0)  # (1, 224, 224, 3)
    return img_arr

def render_gauge(prob_percent: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#0052A2"},
            'steps': [
                {'range': [0, 30], 'color': "#d4f8e8"},
                {'range': [30, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#f8d7da"},
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=260
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# GRADIENT TABANLI ISI HARİTASI (Saliency) 
# -------------------------------------------------
def compute_saliency_heatmap(img_arr: np.ndarray):
    """
    SavedModel'in serving_default fonksiyonu üzerinden,
    giriş pikseline göre gradyan alarak saliency haritası üretir.
    Daha açıklayıcı olması için:
      - 0–1 aralığına normalize edilir
      - Gaussian blur ile yumuşatılır
      - Sadece en yüksek %3'lük bölge bırakılır
    """
    img_tensor = tf.convert_to_tensor(img_arr)  # (1, 224, 224, 3)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = infer(img_tensor)  # dict
        output_tensor = list(preds.values())[0]
        prob = output_tensor[:, 0]  # pnömoni olasılığı

    grads = tape.gradient(prob, img_tensor)  # (1, 224, 224, 3)
    grads = tf.reduce_mean(tf.abs(grads), axis=-1)[0].numpy()  # (224, 224)

    # 0–1 aralığına ölçekle
    grads = grads - grads.min()
    if grads.max() > 0:
        grads = grads / grads.max()

        # Biraz yumuşatma (gürültüyü azaltmak için)
        grads = cv2.GaussianBlur(grads.astype(np.float32), (11, 11), 0)

        # Sadece en güçlü bölgeleri bırak (üst %3)
        thresh = np.percentile(grads, 99)
        mask = grads >= thresh
        filtered = np.zeros_like(grads)
        filtered[mask] = grads[mask]
        grads = filtered

    return grads  # (224, 224), 0–1 aralığı


def overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.25):
    """
    Saliency heatmap'i orijinal görüntünün üzerine bindirir.
    """
    img = np.array(pil_image.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img_uint8 = np.uint8(img)
    superimposed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(superimposed)
# -------------------------------------------------
# ANALİZ KISMI
# -------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_left, col_right = st.columns([1.1, 1.2])

    # SOL: Görüntü
    with col_left:
        st.markdown("""
        <div style="
            background: #ffffff;
            border-radius: 12px;
            padding: 12px;
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.08);
            border: 1px solid #e5e7eb;
            margin-bottom: 10px;">
          <h3 style="margin-top: 0; font-size: 18px; color: #111827;">📷 Yüklenen Göğüs X-Ray Görüntüsü</h3>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # SAĞ: Tahmin + Gauge + Isı Haritası
    with col_right:
        st.markdown("""
        <div style="
            background: #ffffff;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.08);
            border: 1px solid #e5e7eb;
            margin-bottom: 10px;">
          <h3 style="margin-top: 0; font-size: 18px; color: #111827;">🧠 Model Tahmini ve Skor Göstergesi</h3>
        """, unsafe_allow_html=True)

        with st.spinner("Görüntü analiz ediliyor, lütfen bekleyin..."):
            img_arr = preprocess_image(image)
            preds = infer(tf.constant(img_arr))
            output_tensor = list(preds.values())[0].numpy()[0]  # (1,1) -> [0]
            prob = float(output_tensor[0])
            prob_percent = prob * 100.0

        threshold = 0.5
        if prob >= threshold:
            class_label = "Pnömoni"
            color = "#7a0000"
            icon = "🔴"
        else:
            class_label = "Normal"
            color = "#004d26"
            icon = "🟢"

        st.markdown(f"""
        <div style="
            padding: 14px;
            background: {color};
            color: white;
            border-radius: 10px;
            font-size: 17px;
            margin-bottom: 12px;">
            <b>{icon} Tahmin: {class_label}</b>
        </div>
        """, unsafe_allow_html=True)

        render_gauge(prob_percent)

        st.markdown("#### Sonuç Özeti")
        if class_label == "Normal":
            st.write("""
- Model bu görüntüyü **Normal** sınıfında değerlendirmiştir.  
- Bu, modelin bu X-Ray'de pnömoni bulgusunu belirgin şekilde görmediği anlamına gelir.  
- Yine de klinik karar için **doktor muayenesi** ve diğer tetkikler gereklidir.
""")
        else:
            st.write("""
- Model bu görüntüyü **Pnömoni** sınıfına atamıştır.  
- Bu, modelin bu X-Ray'de pnömoni ile uyumlu bulgular tespit ettiğini düşündüğü anlamına gelir.  
- Sonucun mutlaka **klinik değerlendirme ve doktor muayenesi** ile birlikte ele alınması gerekir.
""")

        st.info("""
**Önemli Not:**  
Bu sonuç, yalnızca yapay zekâ modelinin tahminidir.  
Kesin tanı için **doktor muayenesi, klinik bulgular ve ek tetkikler** gereklidir.
""")


        st.markdown("</div>", unsafe_allow_html=True)

    # Geçmişe kaydet
    st.session_state.prediction_history.append({
        "prob_percent": prob_percent,
        "label": class_label
    })

    with st.expander("📊 Son 5 Tahmin Geçmişi"):
        history = st.session_state.prediction_history[-5:][::-1]
        if len(history) == 0:
            st.write("Henüz tahmin yapılmadı.")
        else:
            for i, h in enumerate(history, start=1):
                st.markdown(f"""
                **{i}. Tahmin**  
                - Model skoru: **{h['prob_percent']:.1f}%**  
                - Sınıf: **{h['label']}**
                """)
                st.markdown("---")

else:
    st.info("Analize başlamak için yukarıdan bir göğüs X-Ray görüntüsü yükleyin.")

