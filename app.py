"""
🔍 DETEKSI HOAX BERITA GOSIP ARTIS
Machine Learning Web Application
FIXED: Example buttons now work properly!

Kelompok 2 - TRPL Semester 5
Politeknik Negeri Padang
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
from pathlib import Path

# Sastrawi NLP
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_OK = True
except ImportError:
    SASTRAWI_OK = False

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Deteksi Hoax Berita",
    page_icon="🔍",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main {padding: 1rem;}
    
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .result-fakta {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    
    .result-hoax {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    
    .confidence {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    try:
        path = Path("models")
        return {
            'svm': joblib.load(path / "best_svm_model.pkl"),
            'nb': joblib.load(path / "naive_bayes_model.pkl"),
            'tfidf': joblib.load(path / "tfidf_vectorizer.pkl"),
            'encoder': joblib.load(path / "label_encoder.pkl")
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

@st.cache_resource
def init_sastrawi():
    if not SASTRAWI_OK:
        return None
    try:
        return {
            'stopword': StopWordRemoverFactory().create_stop_word_remover(),
            'stemmer': StemmerFactory().create_stemmer()
        }
    except:
        return None

# ═══════════════════════════════════════════════════════════════
# PREPROCESSING - SAMA DENGAN NOTEBOOK
# ═══════════════════════════════════════════════════════════════

def preprocess(text, sastrawi):
    """
    8 Steps preprocessing (sama dengan notebook):
    1. Lowercase
    2. Remove URL
    3. Remove mentions/hashtags
    4. Remove numbers
    5. Remove punctuation
    6. Remove whitespace
    7. Stopword removal (Sastrawi)
    8. Stemming (Sastrawi)
    """
    if not text:
        return ''
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    
    if sastrawi:
        text = sastrawi['stopword'].remove(text)
        text = sastrawi['stemmer'].stem(text)
    
    return text

# ═══════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════

def predict(text, models, sastrawi, model_choice='svm'):
    processed = preprocess(text, sastrawi)
    
    if len(processed.strip()) < 5:
        return None
    
    tfidf = models['tfidf'].transform([processed])
    model = models['svm'] if model_choice == 'svm' else models['nb']
    
    pred = model.predict(tfidf)[0]
    proba = model.predict_proba(tfidf)[0]
    label = models['encoder'].inverse_transform([pred])[0]
    confidence = max(proba) * 100
    
    # Comparison
    pred_svm = models['svm'].predict(tfidf)[0]
    pred_nb = models['nb'].predict(tfidf)[0]
    label_svm = models['encoder'].inverse_transform([pred_svm])[0]
    label_nb = models['encoder'].inverse_transform([pred_nb])[0]
    conf_svm = max(models['svm'].predict_proba(tfidf)[0]) * 100
    conf_nb = max(models['nb'].predict_proba(tfidf)[0]) * 100
    
    return {
        'label': label,
        'confidence': confidence,
        'processed': processed,
        'comparison': {
            'svm': {'label': label_svm, 'conf': conf_svm},
            'nb': {'label': label_nb, 'conf': conf_nb}
        }
    }

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    # Load
    models = load_models()
    if not models:
        st.stop()
    
    sastrawi = init_sastrawi()
    
    # Header
    st.markdown("""
    <div class="hero">
        <h1>🔍 Deteksi Hoax Berita Gosip Artis</h1>
        <p>Machine Learning: SVM & Naive Bayes • Preprocessing: Sastrawi</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">Kelompok 2 • TRPL Semester 5 • Politeknik Negeri Padang</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")
        
        model_choice = st.radio(
            "Pilih Model:",
            ["SVM Optimized (57%)", "Naive Bayes (54%)"]
        )
        model_key = 'svm' if 'SVM' in model_choice else 'nb'
        
        st.markdown("---")
        st.markdown("### 📊 Info Model")
        
        if model_key == 'svm':
            st.success("**SVM Optimized**")
            st.metric("Accuracy", "57%")
            st.markdown("Best params:\n- C=100\n- kernel=linear")
        else:
            st.info("**Naive Bayes**")
            st.metric("Accuracy", "54%")
        
        st.markdown("---")
        st.markdown("### 📚 Dataset")
        st.markdown("""
        - Total: 499 berita
        - Fakta: 250 (50%)
        - Hoax: 249 (50%)
        - Features: 2247
        """)
        
        st.markdown("---")
        st.markdown("### 🛠️ Preprocessing")
        st.markdown("""
        ✅ Sastrawi Stopword  
        ✅ Sastrawi Stemming  
        ✅ TF-IDF (bigram)
        """)
    
    # Main content
    st.markdown("## 🔮 Analisis Berita")
    
    # Example buttons - FIXED VERSION!
    st.markdown("**Contoh cepat:** *(Klik untuk mengisi otomatis)*")
    col1, col2, col3 = st.columns(3)
    
    examples = [
        "Artis terkenal ditemukan telah meninggal dunia di rumahnya secara mendadak. Pihak kepolisian masih menyelidiki penyebab kematiannya.",
        "Selebriti terkenal mengaku memiliki kekuatan supranatural dan bisa berbicara dengan makhluk dari dimensi lain sejak kecil.",
        "Aktris Korea Song Hye Kyo akan menghadiri acara fashion week di Paris minggu depan sebagai brand ambassador Louis Vuitton."
    ]
    
    # Button handlers - THIS IS THE FIX!
    if col1.button("📰 Berita 1", use_container_width=True):
        st.session_state.text_input = examples[0]
    if col2.button("📰 Berita 2", use_container_width=True):
        st.session_state.text_input = examples[1]
    if col3.button("📰 Berita 3", use_container_width=True):
        st.session_state.text_input = examples[2]
    
    # Input - CONNECTED TO SESSION STATE!
    text = st.text_area(
        "Masukkan teks berita:",
        value=st.session_state.text_input,
        height=150,
        placeholder="Contoh: Artis terkenal ditemukan meninggal dunia di rumahnya...",
        key="text_area"
    )
    
    # Update session state when user types
    if text != st.session_state.text_input:
        st.session_state.text_input = text
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔍 Analisis Berita"):
        if not text or len(text.strip()) < 10:
            st.error("❌ Masukkan berita (minimal 10 karakter)")
        else:
            with st.spinner("🔄 Menganalisis..."):
                result = predict(text, models, sastrawi, model_key)
            
            if not result:
                st.error("❌ Teks terlalu pendek setelah preprocessing")
            else:
                # Result
                st.markdown("## 📋 Hasil Analisis")
                
                if result['label'] == 'Fakta':
                    st.markdown(f"""
                    <div class="result-fakta">
                        <h2>✅ FAKTA</h2>
                        <div class="confidence">{result['confidence']:.1f}%</div>
                        <p>Berita ini diprediksi sebagai <b>FAKTA</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-hoax">
                        <h2>⚠️ HOAX</h2>
                        <div class="confidence">{result['confidence']:.1f}%</div>
                        <p>Berita ini diprediksi sebagai <b>HOAX</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence warning
                if result['confidence'] < 60:
                    st.warning("⚠️ **Confidence rendah!** Model tidak yakin dengan prediksi ini. Disarankan verifikasi dari sumber terpercaya.")
                
                # Model disagreement warning
                if result['comparison']['svm']['label'] != result['comparison']['nb']['label']:
                    st.error("🔀 **Model Disagreement!** SVM dan Naive Bayes memberikan prediksi berbeda. Ini menunjukkan berita memiliki karakteristik ambigu. Sangat disarankan untuk verifikasi manual!")
                
                # Details
                with st.expander("🔍 Detail Analisis"):
                    col1, col2 = st.columns(2)
                    
                    with col1: 
                        st.markdown("**Info:**")
                        st.markdown(f"""
                        - Original: {len(text)} karakter
                        - Processed: {len(result['processed'])} karakter
                        - Words: {len(result['processed'].split())} kata
                        - Model: {model_choice}
                        """)
                    
                    with col2:
                        st.markdown("**Preprocessing:**")
                        st.code(result['processed'][:100] + "..." if len(result['processed']) > 100 else result['processed'], language="text")
                    
                    st.markdown("**Perbandingan Model:**")
                    comp_df = pd.DataFrame({
                        'Model': ['SVM Optimized (57%)', 'Naive Bayes (54%)'],
                        'Prediksi': [
                            result['comparison']['svm']['label'],
                            result['comparison']['nb']['label']
                        ],
                        'Confidence': [
                            f"{result['comparison']['svm']['conf']:.1f}%",
                            f"{result['comparison']['nb']['conf']:.1f}%"
                        ]
                    })
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                # Warning
                st.info("💡 **Disclaimer:** Model memiliki akurasi moderate (54-57%). Selalu verifikasi informasi dari sumber terpercaya sebelum menyebarkan berita!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 1rem;">
        <p><b>Kelompok 2 - TRPL Politeknik Negeri Padang</b></p>
        <p>Machine Learning Project • 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()