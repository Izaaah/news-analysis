# 📰 Penjelasan Singkat Program News Analyzer

## 🎯 Apa yang Dilakukan Program Ini?

Program ini menggunakan **BERT (AI dari Google)** untuk menganalisis berita berbahasa Indonesia. Program bisa:
1. **Mengklasifikasikan** berita ke kategori (Ekonomi/Teknologi/Gaya Hidup)
2. **Mencari berita serupa** dari database
3. **Menampilkan tingkat keyakinan** prediksi

---

## 🤖 Apa itu BERT?

**BERT** = Model AI yang bisa memahami makna teks dalam bahasa Indonesia.

**Cara Kerja BERT:**
```
Teks Berita → BERT → Vektor Angka (768 angka)
```

Vektor ini merepresentasikan **makna** teks. Teks dengan makna mirip akan punya vektor yang mirip juga.

---

## 🔧 3 Fungsi Utama Program

### 1️⃣ **Klasifikasi Kategori** (`predict_label()`)

**Input:** Teks berita  
**Output:** Kategori + Tingkat Keyakinan

**Proses:**
```
Teks → Tokenisasi → BERT Classifier → Softmax → Kategori
```

**Contoh:**
- Input: "Samsung rilis smartphone baru"
- Output: **Teknologi** (95% keyakinan)

---

### 2️⃣ **Pembuatan Embedding** (`get_embedding()`)

**Input:** Teks berita  
**Output:** Vektor 768 angka

**Proses:**
```
Teks → Tokenisasi → BERT Encoder → [CLS] Token → Vektor 768D
```

**Guna:** Vektor ini dipakai untuk mencari berita serupa.

---

### 3️⃣ **Pencarian Berita Serupa** (Similarity Search)

**Input:** Teks berita baru  
**Output:** 5 berita paling mirip dari database

**Proses:**
```
1. Buat embedding untuk teks input
2. Bandingkan dengan semua embedding di database
3. Hitung kemiripan (cosine similarity)
4. Ambil 5 teratas
```

**Rumus Cosine Similarity:**
- Semakin mirip = Skor mendekati 1.0
- Semakin berbeda = Skor mendekati 0.0

---

## 📊 Alur Lengkap Program

```
┌─────────────────────────────────────────┐
│  USER MASUKKAN TEKS BERITA             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  1. TOKENISASI                          │
│     "Samsung rilis..."                  │
│     → [101, 2345, 6789, ...]            │
└──────────────┬──────────────────────────┘
               │
               ├──────────────┬──────────────┐
               │              │              │
               ▼              ▼              ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
│ 2. KLASIFIKASI  │ │ 3. EMBEDDING │ │ 4. SIMILARITY│
│                 │ │              │ │              │
│ BERT Classifier │ │ BERT Encoder │ │ Cosine Match │
│                 │ │              │ │              │
│ → Teknologi    │ │ → Vektor 768D │ │ → Top 5 News │
│ → 95% yakin    │ │              │ │              │
└────────┬────────┘ └──────┬───────┘ └──────┬───────┘
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  5. HASIL DITAMPILKAN │
              │  - Kategori: Teknologi │
              │  - Keyakinan: 95%     │
              │  - 5 Berita Serupa    │
              └────────────────────────┘
```

---

## 📁 File-File Penting

| File | Fungsi |
|------|--------|
| `app.py` | Web server Flask, handle request user |
| `model.py` | Fungsi BERT (klasifikasi & embedding) |
| `make_embeddings.py` | Buat embeddings untuk semua berita di database |
| `embeddings/embeddings.npy` | Semua embedding berita (disimpan untuk cepat) |
| `bert_news_model/` | Model BERT yang sudah dilatih |

---

## 🔑 Konsep Penting

### **Tokenisasi**
Teks diubah jadi angka yang dimengerti BERT.
```
"Samsung" → 1234
"rilis" → 5678
```

### **Embedding**
Teks diubah jadi vektor angka yang merepresentasikan makna.
```
"Smartphone baru" → [0.2, -0.5, 0.8, ..., 0.1] (768 angka)
```

### **Cosine Similarity**
Mengukur kemiripan dua vektor.
```
Berita A vs Berita B → Skor 0.87 (sangat mirip!)
```

### **Softmax**
Mengubah skor mentah jadi probabilitas.
```
Logits: [2.1, 0.5, 3.8]
         ↓
Probabilitas: [15%, 5%, 80%] ← Pilih yang tertinggi
```

---

## 💡 Contoh Praktis

### Input:
```
"Apple mengumumkan iPhone 15 dengan chip A17 Pro"
```

### Proses:
1. **Tokenisasi**: Teks → Token IDs
2. **Klasifikasi**: BERT → "Teknologi" (98% yakin)
3. **Embedding**: BERT → Vektor [0.1, -0.3, 0.7, ...]
4. **Similarity**: Cari di database → 5 berita tentang iPhone/Apple

### Output:
```json
{
  "kategori": "Teknologi",
  "keyakinan": 0.98,
  "berita_serupa": [
    "Samsung rilis Galaxy S24...",
    "Xiaomi luncurkan flagship baru...",
    ...
  ]
}
```

---

## 🚀 Cara Pakai

1. **Jalankan program:**
   ```bash
   python app.py
   ```

2. **Buka browser:**
   ```
   http://localhost:5000
   ```

3. **Masukkan teks berita** dan klik "Analisis Berita"

4. **Lihat hasil:**
   - Kategori berita
   - Tingkat keyakinan
   - Berita serupa

---

## 🎓 Mengapa BERT Bagus?

✅ **Bidirectional**: Baca teks dari kiri-kanan DAN kanan-kiri  
✅ **Context-aware**: Paham konteks kata dalam kalimat  
✅ **Pre-trained**: Sudah belajar dari jutaan teks  
✅ **Fine-tuned**: Disesuaikan untuk berita Indonesia  

---

**Selamat menggunakan! 🎉**



