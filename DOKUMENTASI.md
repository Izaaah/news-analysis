# Dokumentasi Program News Analyzer dengan BERT

## 📋 Daftar Isi
1. [Overview Program](#overview-program)
2. [Arsitektur BERT](#arsitektur-bert)
3. [Fitur-Fitur Program](#fitur-fitur-program)
4. [Alur Kerja Program](#alur-kerja-program)
5. [Penjelasan Detail Setiap Fitur](#penjelasan-detail-setiap-fitur)
6. [Struktur File](#struktur-file)

---

## 🎯 Overview Program

Program ini adalah **News Analyzer** yang menggunakan teknologi **BERT (Bidirectional Encoder Representations from Transformers)** untuk menganalisis dan mengklasifikasikan berita berbahasa Indonesia. Program ini dapat:

- ✅ Mengklasifikasikan berita ke dalam kategori (Ekonomi, Teknologi, Gaya Hidup)
- ✅ Mencari berita serupa dari database menggunakan semantic similarity
- ✅ Menampilkan tingkat keyakinan (confidence) dari prediksi

---

## 🤖 Arsitektur BERT

### Apa itu BERT?

**BERT (Bidirectional Encoder Representations from Transformers)** adalah model bahasa berbasis Transformer yang dikembangkan oleh Google. BERT memiliki kemampuan untuk memahami konteks kata secara dua arah (bidirectional), sehingga lebih baik dalam memahami makna teks dibandingkan model sebelumnya.

### Model BERT yang Digunakan

Program ini menggunakan **BERT yang sudah di-fine-tune** untuk klasifikasi berita Indonesia. Model disimpan di folder `bert_news_model/`.

### Dua Model BERT dalam Program:

1. **`BertForSequenceClassification`** (`clf_model`)
   - Digunakan untuk **klasifikasi kategori berita**
   - Input: Teks berita
   - Output: Probabilitas untuk setiap kategori (Ekonomi, Teknologi, Gaya Hidup)

2. **`BertModel`** (`embed_model`)
   - Digunakan untuk **membuat embedding/vektor** dari teks
   - Input: Teks berita
   - Output: Vektor 768 dimensi yang merepresentasikan makna semantik teks

---

## 🚀 Fitur-Fitur Program

### 1. **Klasifikasi Kategori Berita** 🏷️
Mengklasifikasikan berita ke dalam 3 kategori:
- **Ekonomi**: Berita tentang ekonomi, bisnis, keuangan
- **Teknologi**: Berita tentang teknologi, gadget, inovasi
- **Gaya Hidup**: Berita tentang lifestyle, kesehatan, hiburan

### 2. **Pencarian Berita Serupa** 🔍
Mencari berita yang mirip secara semantik dari database menggunakan cosine similarity pada embedding BERT.

### 3. **Tingkat Keyakinan (Confidence)** 📊
Menampilkan tingkat keyakinan model dalam melakukan prediksi kategori (0-100%).

---

## 🔄 Alur Kerja Program

```
┌─────────────────┐
│  User Input     │  ← User memasukkan teks berita
│  (Teks Berita)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  1. Tokenisasi dengan BERT          │
│     - Teks → Token IDs               │
│     - Max length: 256 tokens         │
└────────┬────────────────────────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ 2. Klasifikasi  │ │ 3. Embedding    │ │ 4. Similarity   │
│    Kategori      │ │    Vektor       │ │    Search       │
│                  │ │                 │ │                 │
│ clf_model        │ │ embed_model     │ │ Cosine Similarity│
│ → Kategori       │ │ → Vektor 768D   │ │ → Top 5 Berita  │
│ → Confidence    │ │                 │ │    Serupa       │
└────────┬─────────┘ └────────┬────────┘ └────────┬────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  5. Hasil       │
                    │  - Kategori     │
                    │  - Confidence  │
                    │  - Berita Serupa│
                    └─────────────────┘
```

---

## 📖 Penjelasan Detail Setiap Fitur

### 1. Klasifikasi Kategori Berita

**File**: `model.py` - fungsi `predict_label()`

**Cara Kerja**:
```python
def predict_label(text):
    # 1. Tokenisasi teks menjadi token IDs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # 2. Forward pass melalui model BERT classifier
    with torch.no_grad():
        outputs = clf_model(**inputs)
        
    # 3. Konversi logits menjadi probabilitas menggunakan softmax
    probs = torch.softmax(outputs.logits, dim=1)
    
    # 4. Ambil kategori dengan probabilitas tertinggi
    label_id = torch.argmax(probs).item()
    confidence = probs[0][label_id].item()
    
    return label_id, confidence
```

**Penjelasan**:
- **Tokenisasi**: Teks diubah menjadi token-token yang dimengerti BERT
- **Truncation**: Teks dipotong jika lebih dari 256 token
- **Padding**: Teks yang pendek diisi dengan padding token
- **Softmax**: Mengubah skor mentah (logits) menjadi probabilitas (0-1)
- **Argmax**: Memilih kategori dengan probabilitas tertinggi

**Output**:
- `label_id`: ID kategori (0=Ekonomi, 1=Lifestyle, 2=Teknologi)
- `confidence`: Tingkat keyakinan (0.0 - 1.0)

---

### 2. Pembuatan Embedding (Vektor Semantik)

**File**: `model.py` - fungsi `get_embedding()`

**Cara Kerja**:
```python
def get_embedding(text):
    # 1. Tokenisasi teks
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # 2. Forward pass melalui BERT encoder
    with torch.no_grad():
        outputs = embed_model(**inputs)
        
    # 3. Ambil embedding dari token [CLS] (token pertama)
    #    Token [CLS] merepresentasikan seluruh kalimat
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape (1, 768)
    
    return cls_embedding.squeeze().numpy()
```

**Penjelasan**:
- **Token [CLS]**: Token khusus di awal yang merepresentasikan seluruh kalimat
- **Embedding 768D**: Setiap teks diubah menjadi vektor 768 dimensi
- **Semantic Meaning**: Vektor ini menangkap makna semantik teks
- **Similarity**: Teks dengan makna serupa akan memiliki vektor yang mirip

**Output**:
- Vektor numpy dengan shape (768,) yang merepresentasikan makna teks

---

### 3. Pencarian Berita Serupa (Similarity Search)

**File**: `app.py` - fungsi `cosine_similarity()` dan logika di route `/predict`

**Cara Kerja**:
```python
# 1. Buat embedding untuk teks input
query_emb = get_embedding(text)

# 2. Hitung cosine similarity dengan semua embedding di database
scores = [cosine_similarity(query_emb, e) for e in embeddings]

# 3. Urutkan dan ambil 5 teratas
top_idx = np.argsort(scores)[::-1][:5]
```

**Rumus Cosine Similarity**:
```
similarity = (A · B) / (||A|| × ||B||)
```
- `A · B`: Dot product antara dua vektor
- `||A||`: Norm (panjang) vektor A
- Hasil: -1 (sangat berbeda) sampai 1 (sangat mirip)

**Penjelasan**:
- **Pre-computed Embeddings**: Semua berita di database sudah di-embedding sebelumnya (disimpan di `embeddings/embeddings.npy`)
- **Cosine Similarity**: Mengukur kemiripan arah vektor (bukan panjang)
- **Top 5**: Mengambil 5 berita dengan similarity score tertinggi

**Output**:
- List 5 berita dengan similarity score tertinggi

---

### 4. Build Embeddings (Pre-processing)

**File**: `make_embeddings.py`

**Cara Kerja**:
```python
def build_and_cache_embeddings(csv_path):
    # 1. Baca dataset berita
    df = pd.read_csv(csv_path)
    
    # 2. Untuk setiap berita, buat embedding
    for i, row in df.iterrows():
        text = row["text"]
        emb = get_embedding(text)  # Panggil BERT
        all_embeddings.append(emb)
        info.append({"text": text, "label": label})
    
    # 3. Simpan embeddings dan metadata
    np.save("embeddings/embeddings.npy", np.array(all_embeddings))
    json.dump(info, "embeddings/emb_index.json")
```

**Penjelasan**:
- **Pre-compute**: Embedding dibuat sekali untuk semua berita di database
- **Caching**: Disimpan dalam file `.npy` untuk akses cepat
- **Metadata**: Informasi teks dan label disimpan di JSON
- **Efisiensi**: Tidak perlu menghitung embedding setiap kali ada query

**Kapan dijalankan**:
- Saat pertama kali setup program
- Ketika dataset berita diperbarui

---

## 📁 Struktur File

```
newsSearching/
│
├── app.py                      # Flask web application (main)
├── model.py                    # BERT model functions (klasifikasi & embedding)
├── make_embeddings.py          # Script untuk membuat embeddings database
├── model copy.py               # Model lengkap dengan fitur tambahan (T5, keywords)
│
├── bert_news_model/            # Model BERT yang sudah di-fine-tune
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── label_classes.json      # ["ekonomi", "lifestyle", "teknologi"]
│
├── embeddings/                 # Pre-computed embeddings
│   ├── embeddings.npy          # Array numpy semua embeddings (N x 768)
│   └── emb_index.json          # Metadata: text dan label untuk setiap embedding
│
├── dataset/
│   └── dataset_berita.csv      # Dataset training/testing
│
├── templates/
│   └── index.html             # Frontend HTML
│
└── static/
    ├── css/
    │   └── style.css          # Styling
    └── js/
        └── main.js            # JavaScript frontend
```

---

## 🔧 Cara Kerja Detail

### 1. Saat Program Dimulai (`app.py`)

```python
# Load embeddings yang sudah di-precompute
embeddings = np.load("embeddings/embeddings.npy")  # Shape: (N, 768)
emb_data = json.load("embeddings/emb_index.json")  # Metadata

# Load model BERT
tokenizer = BertTokenizer.from_pretrained("bert_news_model")
clf_model = BertForSequenceClassification.from_pretrained("bert_news_model")
embed_model = BertModel.from_pretrained("bert_news_model")
```

### 2. Saat User Mengirim Teks Berita

**Step 1: Tokenisasi**
```
Input: "Samsung meluncurkan smartphone baru"
       ↓
Token IDs: [101, 1234, 5678, 9012, ...]  (BERT vocabulary)
```

**Step 2: Klasifikasi**
```
Token IDs → BERT Classifier → Logits [2.1, 0.5, 3.8]
                              ↓
                         Softmax → [0.15, 0.05, 0.80]
                              ↓
                    Argmax → Kategori: Teknologi (confidence: 80%)
```

**Step 3: Embedding**
```
Token IDs → BERT Encoder → Hidden States (256 tokens × 768 dim)
                        ↓
                   [CLS] token → Vektor 768D
```

**Step 4: Similarity Search**
```
Query Embedding (768D) × Database Embeddings (N × 768D)
                    ↓
            Cosine Similarity Scores
                    ↓
            Sort & Take Top 5
```

---

## 📊 Contoh Output

### Input:
```
"Samsung meluncurkan smartphone baru dengan chipset AI dan kamera canggih."
```

### Output:
```json
{
  "input_text": "Samsung meluncurkan smartphone baru...",
  "predicted_label": 2,
  "predicted_category": "teknologi",
  "confidence": 0.95,
  "similar_news": [
    {
      "text": "Apple mengumumkan iPhone terbaru dengan fitur AI...",
      "label": "teknologi",
      "score": 0.87
    },
    {
      "text": "Xiaomi rilis ponsel flagship dengan kamera 108MP...",
      "label": "teknologi",
      "score": 0.82
    }
  ]
}
```

---

## 🎓 Konsep Penting

### 1. **Fine-tuning BERT**
- BERT pre-trained pada data umum
- Di-fine-tune pada dataset berita Indonesia
- Belajar pola khusus untuk klasifikasi berita

### 2. **Embedding vs Classification**
- **Embedding**: Menangkap makna semantik (untuk similarity)
- **Classification**: Menentukan kategori spesifik (untuk labeling)

### 3. **Cosine Similarity**
- Mengukur kemiripan arah vektor
- Tidak terpengaruh panjang teks
- Range: -1 (berlawanan) sampai 1 (sama)

### 4. **Token [CLS]**
- Token khusus di awal setiap input BERT
- Merekam informasi seluruh kalimat
- Digunakan sebagai representasi kalimat

---

## 🚀 Penggunaan Program

### 1. Setup Awal
```bash
# Buat embeddings untuk semua berita di database
python make_embeddings.py
```

### 2. Jalankan Web App
```bash
python app.py
```

### 3. Akses di Browser
```
http://localhost:5000
```

### 4. Masukkan Teks Berita
- Paste atau ketik teks berita
- Klik "Analisis Berita"
- Lihat hasil klasifikasi dan berita serupa

---

## 📝 Catatan Teknis

- **Max Length**: 256 tokens (sekitar 200-300 kata)
- **Embedding Dimension**: 768 (ukuran standar BERT-base)
- **Top K Similar**: 5 berita teratas
- **Device**: CPU atau GPU (otomatis terdeteksi)

---

## 🔮 Fitur Tambahan (di `model copy.py`)

File `model copy.py` berisi implementasi lengkap dengan fitur tambahan:

1. **T5 Summarization**: Ringkasan berita menggunakan T5
2. **Keyword Extraction**: Ekstraksi kata kunci menggunakan attention weights
3. **Mean Pooling**: Alternatif untuk membuat embedding (bukan hanya [CLS])

Fitur-fitur ini bisa diintegrasikan ke `app.py` jika diperlukan.

---

## 📚 Referensi

- **BERT Paper**: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- **HuggingFace Transformers**: https://huggingface.co/transformers/
- **Cosine Similarity**: https://en.wikipedia.org/wiki/Cosine_similarity

---

**Dibuat dengan ❤️ menggunakan BERT dan Flask**

