import torch
import re
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

MODEL_DIR = "bert_news_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
clf_model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
embed_model = BertModel.from_pretrained(MODEL_DIR)

clf_model.eval()
embed_model.eval()


# ---------------------
# PREDIKSI KATEGORI
# ---------------------
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = clf_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs).item()
    return label_id, probs[0][label_id].item()


# ---------------------
# EMBEDDING (VECTOR)
# ---------------------
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape (1, 768)
    return cls_embedding.squeeze().numpy()


# ---------------------
# EKSTRAKSI KATA KUNCI (menggunakan attention BERT)
# ---------------------
def extract_keywords(text, top_k=5):
    """
    Ekstrak kata kunci menggunakan attention weights dari BERT
    """
    try:
        # Tokenisasi
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        
        # Forward pass dengan output_attentions
        embed_model.eval()
        with torch.no_grad():
            # Pastikan output_attentions=True di model call
            outputs = embed_model(**inputs, output_attentions=True)
            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                # Fallback jika model tidak support attention
                raise ValueError("Model tidak support output_attentions")
            attentions = outputs.attentions  # Tuple dari semua layer attention
        
        # Rata-rata attention dari semua layer dan head
        attn_sum = None
        for layer_attn in attentions:
            # layer_attn: (batch=1, num_heads, seq_len, seq_len)
            # Rata-rata semua head
            head_mean = layer_attn.mean(dim=1)  # (1, seq_len, seq_len)
            # Jumlah attention yang diterima setiap token
            token_received = head_mean.sum(dim=1)  # (1, seq_len)
            if attn_sum is None:
                attn_sum = token_received
            else:
                attn_sum += token_received
        
        # Rata-rata dari semua layer
        attn_avg = attn_sum / len(attentions)  # (1, seq_len)
        attn_avg = attn_avg.squeeze().cpu().numpy()  # (seq_len,)
        
        # Konversi token IDs ke tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        mask = inputs['attention_mask'][0].cpu().numpy()
        
        # Kumpulkan skor untuk setiap token
        scores = []
        for i, tok in enumerate(tokens):
            if mask[i] == 0:  # Skip padding
                continue
            if tok in tokenizer.all_special_tokens:  # Skip special tokens
                continue
            scores.append((i, tok, float(attn_avg[i])))
        
        # Gabungkan wordpieces (token yang dimulai dengan ##)
        merged = []
        current_word = ""
        current_score = 0.0
        for idx, tok, sc in scores:
            if tok.startswith("##"):
                current_word += tok[2:]
                current_score += sc
            else:
                if current_word != "":
                    merged.append((current_word, current_score))
                current_word = tok
                current_score = sc
        if current_word != "":
            merged.append((current_word, current_score))
        
        # Filter token yang hanya punctuation atau terlalu pendek
        filtered = [(w, s) for w, s in merged if len(w) > 2 and any(c.isalnum() for c in w)]
        
        # Urutkan berdasarkan skor dan ambil top_k
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
        keywords = [w for w, s in filtered[:top_k]]
        
        # Bersihkan karakter aneh
        keywords = [k.replace("▁", "").strip() for k in keywords]
        keywords = [k for k in keywords if k and len(k) > 1]
        
        return keywords[:top_k] if keywords else ["tidak", "ditemukan"]
    
    except Exception as e:
        print(f"Error extracting keywords with attention: {e}")
        print("Falling back to simple keyword extraction...")
        # Fallback: ambil kata-kata penting sederhana
        try:
            words = re.findall(r'\b\w{4,}\b', text.lower())
            # Hapus stopwords sederhana
            stopwords = {'yang', 'dari', 'dengan', 'untuk', 'pada', 'adalah', 'ini', 'itu', 'dan', 'atau', 'tidak', 'akan', 'sudah', 'juga', 'dapat', 'lebih', 'serta', 'bahwa', 'oleh', 'kepada', 'sebagai', 'dalam', 'seperti', 'karena', 'jika', 'ketika', 'setelah', 'sebelum', 'namun', 'tetapi', 'meskipun', 'walaupun', 'maka', 'jadi', 'maka', 'sehingga', 'agar', 'supaya', 'karena', 'sebab', 'mengapa', 'bagaimana', 'apa', 'siapa', 'dimana', 'kapan', 'kenapa', 'dengan', 'adalah', 'yang', 'dari', 'untuk', 'pada', 'dalam', 'sebagai', 'juga', 'atau', 'akan', 'sudah', 'tidak', 'dapat', 'lebih', 'serta', 'bahwa', 'oleh', 'kepada', 'seperti', 'karena', 'jika', 'ketika', 'setelah', 'sebelum', 'namun', 'tetapi', 'meskipun', 'walaupun', 'maka', 'jadi', 'sehingga', 'agar', 'supaya', 'karena', 'sebab', 'mengapa', 'bagaimana', 'apa', 'siapa', 'dimana', 'kapan', 'kenapa'}
            keywords = [w for w in words if w not in stopwords]
            # Ambil yang paling sering muncul
            from collections import Counter
            word_freq = Counter(keywords)
            result = [w for w, _ in word_freq.most_common(top_k)]
            if result:
                return result
            else:
                # Jika masih kosong, ambil kata apapun yang panjang
                all_words = re.findall(r'\b\w{3,}\b', text.lower())
                return list(set(all_words))[:top_k] if all_words else []
        except Exception as e2:
            print(f"Error in fallback keyword extraction: {e2}")
            return []


# ---------------------
# RINGKASAN TEKS (extractive summarization sederhana)
# ---------------------
def summarize_text(text, max_sentences=3):
    """
    Buat ringkasan menggunakan extractive summarization sederhana
    Berdasarkan panjang kalimat dan posisi dalam teks
    """
    try:
        # Split menjadi kalimat
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Pilih kalimat penting berdasarkan:
        # 1. Kalimat di awal (biasanya lebih penting)
        # 2. Panjang kalimat (tidak terlalu pendek atau panjang)
        scored_sentences = []
        for i, sent in enumerate(sentences):
            score = 0
            # Bonus untuk kalimat di awal
            score += (len(sentences) - i) / len(sentences) * 0.3
            # Bonus untuk panjang yang wajar (30-100 karakter)
            if 30 <= len(sent) <= 100:
                score += 0.4
            elif 20 <= len(sent) <= 150:
                score += 0.2
            # Bonus untuk kalimat yang mengandung kata penting
            important_words = ['adalah', 'merupakan', 'menyatakan', 'mengatakan', 'menjelaskan', 
                             'menurut', 'dalam', 'untuk', 'dengan', 'akan', 'sudah', 'baru']
            if any(word in sent.lower() for word in important_words):
                score += 0.1
            
            scored_sentences.append((sent, score, i))
        
        # Urutkan berdasarkan skor
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil top sentences, tapi urutkan kembali berdasarkan posisi asli
        top_sentences = sorted(scored_sentences[:max_sentences], key=lambda x: x[2])
        summary = '. '.join([s[0] for s in top_sentences])
        
        # Tambahkan titik di akhir jika belum ada
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        return summary if summary else text[:200] + "..."
    
    except Exception as e:
        print(f"Error summarizing text: {e}")
        # Fallback: ambil 200 karakter pertama
        return text[:200] + "..." if len(text) > 200 else text
