from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import re
from collections import Counter
import requests
from bs4 import BeautifulSoup

# Import model functions dengan error handling
try:
    from model import predict_label, get_embedding, summarize_text, extract_keywords
    MODEL_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Error importing model functions: {e}")
    MODEL_FUNCTIONS_AVAILABLE = False
    # Fallback functions
    def summarize_text(text):
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 3:
            return text
        return '. '.join(sentences[:3]) + '.'
    
    def extract_keywords(text, top_k=5):
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stopwords = {'yang', 'dari', 'dengan', 'untuk', 'pada', 'adalah', 'ini', 'itu', 'dan', 'atau', 'tidak', 'akan', 'sudah', 'juga', 'dapat', 'lebih', 'serta', 'bahwa', 'oleh', 'kepada', 'sebagai', 'dalam', 'seperti', 'karena', 'jika', 'ketika', 'setelah', 'sebelum', 'namun', 'tetapi', 'meskipun', 'walaupun', 'maka', 'jadi', 'sehingga', 'agar', 'supaya', 'karena', 'sebab', 'mengapa', 'bagaimana', 'apa', 'siapa', 'dimana', 'kapan', 'kenapa'}
        keywords = [w for w in words if w not in stopwords]
        word_freq = Counter(keywords)
        return [w for w, _ in word_freq.most_common(top_k)]
    
    # Import yang wajib
    from model import predict_label, get_embedding

app = Flask(__name__)

# ---------------- LOAD EMBEDDINGS ----------------
try:
    embeddings = np.load("embeddings/embeddings.npy")
    with open("embeddings/emb_index.json", "r", encoding="utf-8") as f:
        emb_data = json.load(f)
except Exception as e:
    print("ERROR loading embeddings:", e)
    embeddings = None
    emb_data = []

# ---------------- LOAD LABEL CLASSES ----------------
try:
    with open("bert_news_model-2/label_classes.json", "r", encoding="utf-8") as f:
        label_classes = json.load(f)
except Exception as e:
    print("ERROR loading label classes:", e)
    label_classes = []


def cosine_similarity(a, b):
    """Safe cosine similarity"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------- ROUTE HTML ----------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------- TEST ENDPOINT ----------------
@app.route("/test", methods=["GET"])
def test():
    """Test endpoint to verify summary and keywords can be sent"""
    test_result = {
        "input_text": "Test berita teknologi",
        "predicted_label": 2,
        "predicted_category": "teknologi",
        "confidence": 0.95,
        "similar_news": [],
        "summary": "Ini adalah test summary",
        "keywords": ["test", "teknologi", "berita"]
    }
    print(f"[TEST] Sending test response with keys: {list(test_result.keys())}")
    return jsonify(test_result)


# ---------------- ROUTE PREDIKSI ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = (data.get("text") or "").strip()

        if not text:
            return jsonify({"error": "Text tidak boleh kosong"}), 400

        # Initialize default values FIRST - before any processing
        summary = text[:200] + "..." if len(text) > 200 else text
        keywords = []
        
        print(f"[DEBUG] Initialized defaults - summary: {summary[:50]}..., keywords: {keywords}")

        # ---- 1. Prediksi kategori ----
        label_id, confidence = predict_label(text)
        if 0 <= label_id < len(label_classes):
            category_name = label_classes[label_id]
        else:
            category_name = "Unknown"

        # ---- 2. Embedding input ----
        query_emb = get_embedding(text)

        # ---- 3. Similar news (hanya yang kategori sama) ----
        similar = []
        if embeddings is not None and len(embeddings) > 0:
            scores = [cosine_similarity(query_emb, e) for e in embeddings]
            
            # Filter hanya yang kategori sama, lalu ambil top 5
            filtered_indices = []
            for i, score in enumerate(scores):
                if emb_data[i].get("label") == category_name:
                    filtered_indices.append((i, score))
            
            # Sort by score dan ambil top 5
            filtered_indices.sort(key=lambda x: x[1], reverse=True)
            top_idx = [idx for idx, _ in filtered_indices[:5]]

            for i in top_idx:
                original_text = emb_data[i]["text"]
                # Cari URL dari berbagai kemungkinan field
                news_url = emb_data[i].get("url") or emb_data[i].get("link") or None

                # Remove image tags if exist
                if "<img" in original_text:
                    original_text = original_text.split("<img")[0].strip()

                similar.append({
                    "text": original_text[:200] + "..." if len(original_text) > 200 else original_text,
                    "label": emb_data[i]["label"],
                    "score": float(scores[i]),
                    "url": news_url,
                    "index": int(i)  # Untuk akses data lengkap nanti
                })

        # ---- 4. Summary ----
        try:
            result_summary = summarize_text(text)
            if result_summary and isinstance(result_summary, str) and result_summary.strip():
                summary = result_summary.strip()
            print(f"[DEBUG] Summary generated: {summary[:100]}...")
        except Exception as e:
            print(f"[ERROR] Error generating summary: {e}")
            import traceback
            traceback.print_exc()
            # summary tetap menggunakan default yang sudah di-set di awal

        # ---- 5. Keywords ----
        try:
            result_keywords = extract_keywords(text)
            if result_keywords and isinstance(result_keywords, list) and len(result_keywords) > 0:
                keywords = result_keywords
            print(f"[DEBUG] Keywords extracted: {keywords}")
        except Exception as e:
            print(f"[ERROR] Error extracting keywords: {e}")
            import traceback
            traceback.print_exc()
            # keywords tetap menggunakan default empty list yang sudah di-set di awal

        # ---- Final Output ----
        # CRITICAL: Ensure summary and keywords variables exist
        print(f"[DEBUG] Before final output - summary type: {type(summary)}, value: {str(summary)[:50] if summary else 'None'}")
        print(f"[DEBUG] Before final output - keywords type: {type(keywords)}, value: {keywords}")
        
        # Force convert to ensure they exist and are valid
        if not summary or not isinstance(summary, str):
            summary = text[:200] + "..." if len(text) > 200 else text
            print(f"[DEBUG] Summary was invalid, using fallback: {summary[:50]}")
        
        if not isinstance(keywords, list):
            keywords = []
            print(f"[DEBUG] Keywords was invalid, using empty list")
        
        # Build result dictionary - summary and keywords MUST be included
        result = {
            "input_text": str(text),
            "predicted_label": int(label_id),
            "predicted_category": str(category_name),
            "confidence": float(confidence),
            "similar_news": list(similar),
            "summary": str(summary),  # FORCE include - must be string
            "keywords": list(keywords)  # FORCE include - must be list
        }
        
        # CRITICAL: Double-check fields exist
        if "summary" not in result:
            print("[CRITICAL ERROR] summary missing from result dict! Adding...")
            result["summary"] = text[:200] + "..." if len(text) > 200 else text
        
        if "keywords" not in result:
            print("[CRITICAL ERROR] keywords missing from result dict! Adding...")
            result["keywords"] = []
        
        # Final verification with detailed logging
        print(f"[DEBUG] ===== FINAL RESULT VERIFICATION =====")
        print(f"[DEBUG] Result dictionary keys: {list(result.keys())}")
        print(f"[DEBUG] 'summary' in result: {'summary' in result}")
        print(f"[DEBUG] 'keywords' in result: {'keywords' in result}")
        print(f"[DEBUG] summary value type: {type(result.get('summary'))}")
        print(f"[DEBUG] keywords value type: {type(result.get('keywords'))}")
        print(f"[DEBUG] summary value preview: {str(result.get('summary', 'MISSING'))[:100]}")
        print(f"[DEBUG] keywords value: {result.get('keywords', 'MISSING')}")
        print(f"[DEBUG] ======================================")
        
        # Create JSON response
        try:
            response_data = jsonify(result)
            print(f"[DEBUG] jsonify() successful, status code: {response_data.status_code}")
            
            # Verify JSON serialization
            import json as json_module
            json_str = json_module.dumps(result, ensure_ascii=False, default=str)
            print(f"[DEBUG] JSON serialization successful, length: {len(json_str)}")
            has_summary = '"summary"' in json_str
            has_keywords = '"keywords"' in json_str
            print(f"[DEBUG] JSON contains 'summary': {has_summary}")
            print(f"[DEBUG] JSON contains 'keywords': {has_keywords}")
            
            # Final check before returning
            print(f"[DEBUG] About to return response with {len(result)} keys")
            print(f"[DEBUG] Response will contain summary: {'summary' in result}")
            print(f"[DEBUG] Response will contain keywords: {'keywords' in result}")
            
            return response_data
        except Exception as json_error:
            print(f"[CRITICAL ERROR] jsonify() failed: {json_error}")
            import traceback
            traceback.print_exc()
            # Return minimal response but still include summary and keywords
            error_response = {
                "error": "JSON serialization failed",
                "input_text": str(text) if 'text' in locals() else "",
                "predicted_label": int(label_id) if 'label_id' in locals() else 0,
                "predicted_category": str(category_name) if 'category_name' in locals() else "Unknown",
                "confidence": float(confidence) if 'confidence' in locals() else 0.0,
                "similar_news": [],
                "summary": str(summary) if 'summary' in locals() and summary else (text[:200] + "..." if 'text' in locals() and len(text) > 200 else (text if 'text' in locals() else "")),
                "keywords": list(keywords) if 'keywords' in locals() and isinstance(keywords, list) else []
            }
            print(f"[DEBUG] Error response keys: {list(error_response.keys())}")
            return jsonify(error_response), 500
    
    except Exception as e:
        print(f"[FATAL ERROR] in predict(): {e}")
        import traceback
        traceback.print_exc()
        # Return error response but still include summary and keywords
        error_text = text if 'text' in locals() else ""
        error_summary = error_text[:200] + "..." if error_text and len(error_text) > 200 else (error_text if error_text else "Error processing text")
        error_response = {
            "error": str(e),
            "input_text": error_text,
            "predicted_label": 0,
            "predicted_category": "Unknown",
            "confidence": 0.0,
            "similar_news": [],
            "summary": error_summary,  # MUST include
            "keywords": []  # MUST include
        }
        print(f"[DEBUG] Fatal error response keys: {list(error_response.keys())}")
        print(f"[DEBUG] Fatal error response has summary: {'summary' in error_response}")
        print(f"[DEBUG] Fatal error response has keywords: {'keywords' in error_response}")
        return jsonify(error_response), 500


# ---------------- ROUTE FETCH FULL TEXT FROM URL ----------------
@app.route("/fetch-article", methods=["POST"])
def fetch_article():
    """Fetch full text dari URL artikel"""
    try:
        data = request.json
        url = data.get("url")
        
        if not url:
            return jsonify({"error": "URL tidak boleh kosong"}), 400
        
        # Scrape artikel dari URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Hapus script dan style
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Cari konten artikel (coba beberapa selector umum)
            article_content = None
            selectors = [
                'article',
                '.article-content',
                '.content',
                '.post-content',
                '#content',
                '.detail-text'
            ]
            
            for selector in selectors:
                article = soup.select_one(selector)
                if article:
                    article_content = article.get_text(separator=' ', strip=True)
                    break
            
            # Jika tidak ketemu, ambil dari body
            if not article_content:
                body = soup.find('body')
                if body:
                    article_content = body.get_text(separator=' ', strip=True)
            
            if not article_content:
                return jsonify({"error": "Tidak dapat mengambil konten artikel"}), 400
            
            # Bersihkan teks
            article_content = ' '.join(article_content.split())
            
            return jsonify({
                "success": True,
                "url": url,
                "full_text": article_content
            })
            
        except requests.RequestException as e:
            return jsonify({"error": f"Error fetching URL: {str(e)}"}), 500
            
    except Exception as e:
        print(f"[ERROR] Error in fetch_article: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------- ROUTE ANALYZE FULL ARTICLE ----------------
@app.route("/analyze-article", methods=["POST"])
def analyze_article():
    """Analisis artikel lengkap dengan semua fitur"""
    try:
        data = request.json
        full_text = data.get("full_text") or data.get("text", "")
        url = data.get("url", "")
        
        if not full_text:
            return jsonify({"error": "Full text tidak boleh kosong"}), 400
        
        # Initialize defaults
        summary = full_text[:200] + "..." if len(full_text) > 200 else full_text
        keywords = []
        
        # 1. Prediksi kategori
        label_id, confidence = predict_label(full_text)
        if 0 <= label_id < len(label_classes):
            category_name = label_classes[label_id]
        else:
            category_name = "Unknown"
        
        # 2. Embedding untuk similar news
        query_emb = get_embedding(full_text)
        
        # 3. Similar news (kategori sama)
        similar = []
        if embeddings is not None and len(embeddings) > 0:
            scores = [cosine_similarity(query_emb, e) for e in embeddings]
            
            filtered_indices = []
            for i, score in enumerate(scores):
                if emb_data[i].get("label") == category_name:
                    filtered_indices.append((i, score))
            
            filtered_indices.sort(key=lambda x: x[1], reverse=True)
            top_idx = [idx for idx, _ in filtered_indices[:5]]
            
            for i in top_idx:
                original_text = emb_data[i]["text"]
                news_url = emb_data[i].get("link") or emb_data[i].get("url") or None
                
                if "<img" in original_text:
                    original_text = original_text.split("<img")[0].strip()
                
                similar.append({
                    "text": original_text[:200] + "..." if len(original_text) > 200 else original_text,
                    "label": emb_data[i]["label"],
                    "score": float(scores[i]),
                    "url": news_url,
                    "index": int(i)
                })
        
        # 4. Summary
        try:
            result_summary = summarize_text(full_text)
            if result_summary and isinstance(result_summary, str) and result_summary.strip():
                summary = result_summary.strip()
        except Exception as e:
            print(f"[ERROR] Error generating summary: {e}")
        
        # 5. Keywords
        try:
            result_keywords = extract_keywords(full_text)
            if result_keywords and isinstance(result_keywords, list) and len(result_keywords) > 0:
                keywords = result_keywords
        except Exception as e:
            print(f"[ERROR] Error extracting keywords: {e}")
        
        return jsonify({
            "input_text": full_text,
            "predicted_label": int(label_id),
            "predicted_category": category_name,
            "confidence": float(confidence),
            "similar_news": similar,
            "summary": str(summary),
            "keywords": list(keywords),
            "url": url
        })
        
    except Exception as e:
        print(f"[ERROR] Error in analyze_article: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    # Penting: disable reloader untuk Windows (hindari WinError 10038)
    app.run(debug=True, use_reloader=False)
