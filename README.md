# 🔍 Deteksi Hoax Berita Gosip Artis

Aplikasi Machine Learning untuk mendeteksi berita hoax pada gosip artis menggunakan **Support Vector Machine (SVM)** dan **Naive Bayes** dengan preprocessing **Sastrawi** untuk bahasa Indonesia.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎯 Tentang Project

**Project ini merupakan tugas akhir mata kuliah Machine Learning**

- **Program Studi:** Teknik Rekayasa Perangkat Lunak (TRPL)
- **Institusi:** Politeknik Negeri Padang
- **Semester:** 5
- **Kelompok:** 2

## 📊 Dataset

- **Total:** 499 berita gosip artis
- **Fakta:** 250 berita (50%)
- **Hoax:** 249 berita (50%)
- **Sumber:** Berita online Indonesia

## 🤖 Model Machine Learning

### Model yang Digunakan:

1. **Support Vector Machine (SVM) Optimized**
   - Accuracy: **57%**
   - Best Parameters: C=100, kernel=linear
   - Model utama untuk prediksi

2. **Naive Bayes**
   - Accuracy: **54%**
   - Baseline model
   - Comparison model

### Preprocessing:

- ✅ **Sastrawi Stopword Removal** (Bahasa Indonesia)
- ✅ **Sastrawi Stemming** (Bahasa Indonesia)
- ✅ **TF-IDF Vectorization** (5000 features, bigram)
- ✅ **8-step cleaning pipeline**

### Hyperparameter Tuning:

- Grid Search CV dengan 5-fold cross-validation
- 32 kombinasi parameter tested
- Best model selected berdasarkan accuracy

## 🚀 Fitur Aplikasi

- ✨ **Real-time Prediction:** Analisis berita secara langsung
- 🎨 **Modern UI:** Interface yang clean dan user-friendly
- 📊 **Model Comparison:** Lihat prediksi dari kedua model
- ⚠️ **Confidence Score:** Tingkat keyakinan prediksi
- 🔍 **Detail Analysis:** Lihat preprocessing dan statistik
- 📱 **Responsive Design:** Mobile-friendly

## 💻 Teknologi yang Digunakan

- **Framework:** Streamlit
- **ML Libraries:** scikit-learn 1.7.2
- **NLP:** Sastrawi (Indonesian NLP)
- **Language:** Python 3.10

## 📦 Installation (Local)

```bash
# Clone repository
git clone https://github.com/your-username/hoax-detector.git
cd hoax-detector

# Install dependencies
pip install -r requirements.txt

# Run aplikasi
streamlit run app.py
```

## ⚠️ Disclaimer

**PENTING:** Aplikasi ini adalah project edukasi dengan akurasi moderate (54-57%). 

- ⚠️ **Jangan 100% bergantung** pada prediksi model
- ✅ **Selalu verifikasi** dari sumber terpercaya
- 📚 **Gunakan untuk pembelajaran**, bukan keputusan critical

Model memiliki limitasi:
- Dataset relatif kecil (499 samples)
- Overlap tinggi antara karakteristik Fakta dan Hoax
- Calibration issues (Naive Bayes overconfident)

## 📈 Performance Metrics

| Metric | SVM | Naive Bayes |
|--------|-----|-------------|
| Test Accuracy | 57% | 54% |
| CV Accuracy | 52.39% | 51.63% |
| Precision | 0.57 | 0.54 |
| Recall | 0.57 | 0.54 |
| F1-Score | 0.57 | 0.54 |

## 🎓 Tim Pengembang

**Kelompok 2 - TRPL Semester 5**

Politeknik Negeri Padang

## 📝 License

Project ini dibuat untuk keperluan edukasi.

## 🙏 Acknowledgments

- Dosen Mata Kuliah Machine Learning
- Politeknik Negeri Padang
- Sastrawi Library untuk Indonesian NLP
- Streamlit untuk framework aplikasi

## 📧 Contact

Untuk pertanyaan atau feedback, silakan buka issue di repository ini.

---

**⭐ Jangan lupa star repository ini jika bermanfaat!**

*Dibuat dengan ❤️ untuk pembelajaran Machine Learning*
