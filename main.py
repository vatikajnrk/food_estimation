import os
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

import dotenv
from google import genai

dotenv.load_dotenv()

# Path to your model file
MODEL_PATH = './model/mobilenetv2_fruit_veggie.keras'
# Path to class names file (one label per line)
LABELS_PATH = './model/class_names.txt'


@st.cache_resource
def get_model():
    """Load and cache the Keras model."""
    return load_model(MODEL_PATH)


def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Resize, convert to RGB and scale image to a model-ready numpy array.

    Returns a batch of 1 image: shape (1, H, W, C)
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.asarray(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_labels(path: str) -> list:
    """Load class labels from a text file (one label per line).

    Returns a list of labels. Raises IOError if file can't be read.
    """
    labels = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            v = line.strip()
            if v:
                labels.append(v)
    return labels


def predict(img: Image.Image, model, labels=None):
    x = preprocess_image(img)
    preds = model.predict(x)
    # preds expected shape (1, num_classes) or (1,)
    if preds.ndim == 2:
        probs = preds[0]
    else:
        probs = np.array([preds[0]])

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx]) * 100.0
    if labels:
        if 0 <= top_idx < len(labels):
            label = labels[top_idx]
        else:
            label = f"class_{top_idx}"
    else:
        label = f"class_{top_idx}"

    # prepare top-3 for display
    top_k = min(3, probs.shape[0])
    top_indices = probs.argsort()[-top_k:][::-1]
    top_results = [(int(i), float(probs[i]) * 100.0, (labels[i] if labels and i < len(labels) else f'class_{i}')) for i in top_indices]

    return {
        'label': label,
        'index': top_idx,
        'confidence': confidence,
        'probs': probs,
        'top_results': top_results,
    }
    
def main():
    st.set_page_config(page_title='Food / Fruit & Veg Classifier', layout='centered')
    st.title('Food / Fruit & Veg Estimation Classifier')

    # Load labels from `class_names.txt` if available (no sidebar settings)
    try:
        labels = load_labels(LABELS_PATH)
        st.info(f'Loaded {len(labels)} labels from `{LABELS_PATH}`')
    except Exception as e:
        labels = None
        st.warning(f'Could not load labels from {LABELS_PATH}: {e}')

    # Load model
    try:
        model = get_model()
        st.success('Model loaded')
    except Exception as e:
        st.error(f'Error loading model: {e}')
        st.stop()
    uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'bmp'])

    if uploaded_file is None:
        st.info('Please upload an image to preview and classify.')
    else:
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
        except Exception as e:
            st.error(f'Cannot open image: {e}')
            return

        st.subheader('Preview')
        # use_container_width replaces deprecated use_column_width
        st.image(image, use_container_width=True)

        if st.button('Classify image'):
            with st.spinner('Running prediction...'):
                try:
                    result = predict(image, model, labels=labels)
                except Exception as e:
                    st.error(f'Prediction failed: {e}')
                    return

            conf = result['confidence']
            label = result['label']

            # If confidence is low, show warning color
            if conf < 99.0:
                st.warning(f"Low confidence: {label} ({conf:.2f}%) — kemungkinan bukan kelas yang cocok")
            else:
                st.success(f"Prediksi: {label} ({conf:.2f}%)")
            
                try:
                    with st.spinner('Mengambil informasi penyimpanan dari Gemini...'):
                        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", contents=f"""Berdasarkan hasil klasifikasi gambar {label}. 
                            berikan estimasi waktu penyimpanan yang ideal untuk menjaga kesegarannya,
                            serta tips penyimpanan yang tepat agar tetap segar lebih lama (semisal berikan tips penyimpanan di tempat yang sejuk atau kering dan 
                            berikan estimasi pada masing-masing penyimpanan). **BERIKAN RESPON DALAM BENTUK LIST**""",
                        )
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Gagal mendapatkan informasi dari Gemini: {e}")
                    st.info("Silakan periksa API key Gemini Anda atau coba lagi nanti.")

            st.subheader('Top predictions')
            for idx, conf_i, lab in result['top_results']:
                st.write(f"{lab} (index {idx}): {conf_i:.2f}%")

            # Optionally show raw probabilities as a small bar chart if labels provided
            try:
                if labels and len(labels) == result['probs'].shape[0]:
                    import pandas as pd
                    df = pd.DataFrame({ 'prob': result['probs'] * 100.0 }, index=labels)
                    st.bar_chart(df['prob'])
            except Exception:
                # don't fail if charting fails
                pass
        
        


if __name__ == '__main__':
    main()