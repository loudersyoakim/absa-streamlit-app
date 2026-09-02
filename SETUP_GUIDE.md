# 📋 SETUP GUIDE - ABSA Integrated Single Page App

## Daftar Isi
1. [Overview](#overview)
2. [Struktur Folder](#struktur-folder)
3. [Setup Awal](#setup-awal)
4. [Integrasi Inference Logic](#integrasi-inference-logic)
5. [Struktur Data Result](#struktur-data-result)
6. [Kalimat Loading (Untuk Disepakati)](#kalimat-loading-untuk-disepakati)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Aplikasi ini adalah **single-page Streamlit app** yang mengintegrasikan:
- ✅ Model selection (IndoBERT / mBERT)
- ✅ 3 input modes (Teks, Upload File, URL Tokopedia)
- ✅ History management (dengan edit/delete)
- ✅ Complete result visualization sesuai mockup Anda
- ✅ Aspect-based sentiment analysis

**Current Status:** Template dengan sample data (ready untuk integrasi inference)

---

## Struktur Folder

```
ABSA_Project/
├── app.py                          # Main application (ganti nama dari app_absa_integrated.py)
├── inference.py                    # [DARI ABSA_PERTAMA] - Inference logic Anda
├── utils.py                        # Utility functions
├── requirements.txt                # Dependencies
├── model/                          # Model directory
│   ├── indobert_acd/
│   ├── indobert_asc/
│   ├── mbert_acd/
│   ├── mbert_asc/
│   └── label_maps.json
├── data/
│   ├── history.json               # Auto-generated history
│   └── .gitkeep
└── .streamlit/
    └── config.toml                # Streamlit config
```

---

## Setup Awal

### 1. Persiapan Environment

```bash
# Clone atau copy project ke folder baru
mkdir ABSA_Project
cd ABSA_Project

# Copy file app_absa_integrated.py (rename jadi app.py)
cp app_absa_integrated.py app.py

# Copy files dari ABSA_pertama
cp -r /path/to/ABSA_pertama/model ./
cp -r /path/to/ABSA_pertama/requirements.txt ./
cp /path/to/ABSA_pertama/inference.py ./
cp /path/to/ABSA_pertama/utils.py ./
```

### 2. Install Dependencies

```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
pip install streamlit plotly pandas

# Pastikan streamlit >= 1.28.0
streamlit --version
```

### 3. Test Run

```bash
streamlit run app.py
```

---

## Integrasi Inference Logic

### Letak Integration Point

File: `app.py`, Section **8. PEMROSESAN PESAN & SIMULASI HASIL** (Around line 530)

### Current Code (Replace dengan logic Anda):

```python
# ====== SAMPLE RESULT DATA (Replace with actual inference) ======
sample_results = { ... }

st.session_state.analysis_results = sample_results
```

### Replace dengan:

```python
# ====== CALL YOUR INFERENCE LOGIC ======
try:
    # 1. Process input berdasarkan mode
    if st.session_state.input_mode == "Teks":
        input_data = pesan_user
    elif st.session_state.input_mode == "Upload File":
        input_data = file_terupload_list  # List of UploadedFile objects
    elif st.session_state.input_mode == "URL Tokopedia":
        input_data = url_terkirim
    
    # 2. Call inference function
    results = run_inference(
        input_data=input_data,
        model_type=pilihan_model,  # "IndoBERT" or "mBERT"
        mode=st.session_state.input_mode
    )
    
    # 3. Store results
    st.session_state.analysis_results = results
    
except Exception as e:
    st.error(f"❌ Error during analysis: {str(e)}")
    st.session_state.analysis_results = None
```

---

## Struktur Data Result

Hasil inference HARUS memiliki struktur berikut:

```python
{
    "top_positive": {
        "aspect": str,              # Nama aspek paling positif
        "percentage": float,        # Persentase positif (0-100)
        "positive": int,            # Jumlah ulasan positif
        "neutral": int,             # Jumlah ulasan netral
        "negative": int,            # Jumlah ulasan negatif
        "total": int                # Total ulasan untuk aspek ini
    },
    
    "top_negative": {
        "aspect": str,
        "percentage": float,        # Persentase negatif (0-100)
        "positive": int,
        "neutral": int,
        "negative": int,
        "total": int
    },
    
    "overall_sentiment_distribution": {
        "Positif": int,             # Total positif
        "Netral": int,              # Total netral
        "Negatif": int              # Total negatif
    },
    
    "aspect_distribution": {
        "Aspek1": int,              # Jumlah ulasan untuk Aspek1
        "Aspek2": int,
        ...
    },
    
    "aspects_detailed": [
        {
            "aspect": str,                          # Nama aspek (e.g., "Display", "Harga")
            "positive_count": int,                  # Total positif untuk aspek ini
            "neutral_count": int,                   # Total netral
            "negative_count": int,                  # Total negatif
            "positive_reviews": List[str],          # Max 3 ulasan positif
            "neutral_reviews": List[str],           # Max 3 ulasan netral
            "negative_reviews": List[str]           # Max 3 ulasan negatif
        },
        ...
    ]
}
```

### Contoh Implementation:

```python
def run_inference(input_data, model_type: str, mode: str) -> dict:
    """
    Run ABSA inference
    
    Args:
        input_data: Teks, list of files, atau URL
        model_type: "IndoBERT" atau "mBERT"
        mode: "Teks", "Upload File", atau "URL Tokopedia"
    
    Returns:
        dict: Structured result matching format above
    """
    
    # 1. Data preprocessing berdasarkan mode
    if mode == "Teks":
        reviews = [input_data]
    elif mode == "Upload File":
        reviews = parse_files(input_data)  # Dari ABSA_pertama
    elif mode == "URL Tokopedia":
        reviews = scrape_tokopedia(input_data)  # Dari ABSA_pertama
    
    # 2. Load model
    if model_type == "IndoBERT":
        model = load_indobert_model()
        acd_model = load_indobert_acd()
        asc_model = load_indobert_asc()
    else:
        model = load_mbert_model()
        acd_model = load_mbert_acd()
        asc_model = load_mbert_asc()
    
    # 3. Run ACD (Aspect Category Detection)
    aspect_results = {}
    for review in reviews:
        aspects = acd_model.predict(review)
        for aspect in aspects:
            if aspect not in aspect_results:
                aspect_results[aspect] = {
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0,
                    'reviews': {'positive': [], 'neutral': [], 'negative': []}
                }
    
    # 4. Run ASC (Aspect-based Sentiment Classification)
    for review in reviews:
        for aspect in aspect_results:
            sentiment = asc_model.predict(review, aspect)
            aspect_results[aspect][sentiment.lower()] += 1
            if len(aspect_results[aspect]['reviews'][sentiment.lower()]) < 3:
                aspect_results[aspect]['reviews'][sentiment.lower()].append(review[:100])
    
    # 5. Structure result
    result = {
        "top_positive": calculate_top_positive(aspect_results),
        "top_negative": calculate_top_negative(aspect_results),
        "overall_sentiment_distribution": calculate_overall_sentiment(aspect_results),
        "aspect_distribution": calculate_aspect_distribution(aspect_results),
        "aspects_detailed": structure_detailed_results(aspect_results)
    }
    
    return result
```

---

## Kalimat Loading (Untuk Disepakati)

### Current Default (Bisa diubah):

**Saat File Diupload / Teks Diinput:**
```
"⏳ Sedang memproses ulasan Anda..."
```

### Proposal Kalimat Tambahan (Pilih yang cocok):

**Saat Preprocessing:**
- "🔄 Menyiapkan data..."
- "📖 Membaca ulasan..."
- "✨ Membersihkan teks..."

**Saat Model Running:**
- "🤖 Model menganalisis aspek dalam ulasan..."
- "💭 Mendeteksi sentimen untuk setiap aspek..."
- "⚙️ Memproses klasifikasi sentimen..."

**Saat Finalizing:**
- "📊 Menyusun hasil analisis..."
- "🎯 Menghitung statistik..."
- "✅ Hampir selesai, tunggu sebentar..."

**Saat Selesai:**
- "✨ Analisis selesai! Berikut hasilnya:"
- "🎉 Siap! Hasil analisis telah diperbarui."

---

## Konfigurasi Streamlit

File: `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#10B981"
backgroundColor = "#131314"
secondaryBackgroundColor = "#1E1F22"
textColor = "#E3E3E3"
font = "sans serif"

[client]
showErrorDetails = true

[server]
headless = true
port = 8501
```

---

## Features Implementation Checklist

- [x] Single page layout
- [x] Sidebar dengan model selection
- [x] 3 input modes (Teks, File, URL)
- [x] History management UI
- [x] Result visualization (Baris 1, 2, 3, 4)
- [ ] **TODO:** Integrate actual inference
- [ ] **TODO:** History save/load to file
- [ ] **TODO:** Delete/Edit history items
- [ ] **TODO:** URL scraping implementation
- [ ] **TODO:** Batch file processing

---

## Troubleshooting

### Q: Plotly chart tidak muncul
**A:** Pastikan Plotly version >= 5.0
```bash
pip install --upgrade plotly
```

### Q: CSS styling tidak bekerja
**A:** Streamlit mungkin di-cache. Jalankan dengan:
```bash
streamlit run app.py --logger.level=debug --client.clearCacheOnRerun=true
```

### Q: Model load error
**A:** Pastikan path model di inference.py benar:
```python
# Pastikan PYTHONPATH include parent directory
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

### Q: History tidak tersimpan
**A:** Akan diimplementasi di next phase dengan:
- `history_store.py` (dari ABSA_pertama)
- Session state persistence
- JSON file storage

---

## Next Steps

1. **✅ Confirm loading messages** - Pilih dari proposal di atas
2. **✅ Test template** - Jalankan app.py dan check UI
3. **📝 Implement inference** - Copy logic dari ABSA_pertama ke section 8
4. **🧪 Test with real data** - Upload files CSV/XLSX dan URL Tokopedia
5. **💾 Implement history persistence** - Save to JSON
6. **🎨 Fine-tune styling** - Adjust colors/spacing sesuai kebutuhan

---

## Questions?

Hubungi tim development dengan:
- Error logs (jika ada)
- Data sample yang problematic
- Screenshot UI issues
