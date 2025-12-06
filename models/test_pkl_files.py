"""
Test apakah file .pkl valid dan bisa digunakan
Run ini di folder models/ kamu
"""

import joblib
import os

print("=" * 60)
print("TESTING FILE .PKL")
print("=" * 60)

# List file yang harus ada
required_files = [
    'best_svm_model.pkl',
    'naive_bayes_model.pkl', 
    'tfidf_vectorizer.pkl',
    'label_encoder.pkl'
]

# Cek keberadaan file
print("\n1. CHECKING FILES:")
for filename in required_files:
    if os.path.exists(filename):
        size = os.path.getsize(filename) / 1024  # KB
        print(f"   ✓ {filename}: {size:.1f} KB")
    else:
        print(f"   ✗ {filename}: NOT FOUND!")

# Test load each file
print("\n2. TESTING LOAD:")
models = {}
errors = []

for filename in required_files:
    try:
        model = joblib.load(filename)
        models[filename] = model
        print(f"   ✓ {filename}: Load OK")
    except Exception as e:
        print(f"   ✗ {filename}: ERROR - {e}")
        errors.append((filename, str(e)))

# Test TF-IDF specifically
print("\n3. TESTING TF-IDF VECTORIZER:")
if 'tfidf_vectorizer.pkl' in models:
    tfidf = models['tfidf_vectorizer.pkl']
    
    # Check if fitted
    try:
        # Ini akan error jika belum fitted
        test_text = ["test berita"]
        result = tfidf.transform(test_text)
        print(f"   ✓ TF-IDF is FITTED!")
        print(f"   ✓ Vocabulary size: {len(tfidf.vocabulary_)}")
        print(f"   ✓ Feature shape: {result.shape}")
    except Exception as e:
        print(f"   ✗ TF-IDF NOT FITTED!")
        print(f"   ✗ Error: {e}")
        print(f"\n   >>> MASALAHNYA DI SINI! <<<")

# Test SVM
print("\n4. TESTING SVM MODEL:")
if 'best_svm_model.pkl' in models:
    svm = models['best_svm_model.pkl']
    print(f"   ✓ SVM loaded")
    print(f"   ✓ Kernel: {svm.kernel}")
    print(f"   ✓ C: {svm.C}")

# Summary
print("\n" + "=" * 60)
if errors:
    print("❌ ADA MASALAH!")
    print("\nFile yang bermasalah:")
    for filename, error in errors:
        print(f"   - {filename}: {error}")
else:
    print("✅ SEMUA FILE OK!")
print("=" * 60)
