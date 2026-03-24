import os
import math
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix

BERT_MODEL_NAME = "indobenchmark/indobert-base-p1" 
T5_MODEL_NAME = "google/mt5-small"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFIER_SAVE_DIR = "bert_news_model"
EMBEDDING_CACHE = "embeddings.npy"
EMBEDDING_INDEX = "emb_index.json"
T5_SAVE_DIR = "t5_summary_model"

MAX_LEN = 256
BATCH_SIZE_EMB = 8

def read_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert 'text' in df.columns and 'label' in df.columns, "CSV harus punya kolom 'text' dan 'label'"
    df = df.dropna(subset=['text','label']).reset_index(drop=True)
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str)
    return df

# 1) TRAIN / FINE-TUNE BERT CLASSIFIER (Trainer API)
def train_bert_classifier(csv_path: str, out_dir: str = CLASSIFIER_SAVE_DIR, epochs: int = 3):
    df = read_dataset(csv_path)
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    print(f"Labels: {list(le.classes_)}")

    from datasets import Dataset
    ds = Dataset.from_pandas(df[['text','label_id']])
    ds = ds.rename_column("label_id", "labels")

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LEN)
    ds = ds.map(tokenize_fn, batched=True)
    ds = ds.remove_columns(['text'])
    ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    # split
    split = ds.train_test_split(test_size=0.2)
    train_ds = split['train']
    test_ds = split['test']

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(le.classes_))
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir="bert_train_output",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # Evaluasi
    preds_output = trainer.predict(test_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = preds_output.label_ids

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=list(le.classes_)))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

    # Save model + tokenizer + label encoder
    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    # save label encoder classes
    with open(os.path.join(out_dir, "label_classes.json"), "w", encoding="utf-8") as f:
        json.dump(list(le.classes_), f, ensure_ascii=False)

    print(f"\nSaved classifier to {out_dir}")

# 2) Functions untuk load classifier untuk inference
def load_classifier(model_dir: str = CLASSIFIER_SAVE_DIR):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    # load label classes
    with open(os.path.join(model_dir, "label_classes.json"), "r", encoding="utf-8") as f:
        classes = json.load(f)
    return tokenizer, model, classes

# 3) BERT embeddings builder (mean pooling)
def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (batch, seq_len, 1)
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = (summed / counts).cpu().numpy()
    return mean_pooled

def build_and_cache_embeddings(csv_path: str,
                               bert_encoder: BertModel = None,
                               tokenizer: BertTokenizer = None,
                               cache_file: str = EMBEDDING_CACHE,
                               index_file: str = EMBEDDING_INDEX):
    df = read_dataset(csv_path)
    if bert_encoder is None or tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_encoder = BertModel.from_pretrained(BERT_MODEL_NAME)
        bert_encoder.to(DEVICE)
    bert_encoder.eval()

    texts = df['text'].tolist()
    all_embs = []
    batch = []
    indices = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE_EMB), desc="Embedding batches"):
        batch_texts = texts[i:i+BATCH_SIZE_EMB]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        input_ids = enc['input_ids'].to(DEVICE)
        attn_mask = enc['attention_mask'].to(DEVICE)
        with torch.no_grad():
            out = bert_encoder(input_ids=input_ids, attention_mask=attn_mask)
            last_hidden = out.last_hidden_state  # (batch, seq_len, hidden)
            pooled = mean_pooling(last_hidden, attn_mask)  # (batch, hidden) numpy
            for row in pooled:
                all_embs.append(row)
    all_embs = np.vstack(all_embs) 

    np.save(cache_file, all_embs)
    index = []
    for idx, row in df.iterrows():
        index.append({"idx": idx, "text": row['text'], "label": row['label']})
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Saved embeddings to {cache_file} and index to {index_file}")

# 4) Similarity search (uses cached embeddings)
def load_embeddings(cache_file: str = EMBEDDING_CACHE, index_file: str = EMBEDDING_INDEX):
    embs = np.load(cache_file)
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)
    return embs, index

def get_embedding_for_text(text: str, tokenizer: BertTokenizer, bert_encoder: BertModel) -> np.ndarray:
    bert_encoder.eval()
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors="pt")
    input_ids = enc['input_ids'].to(DEVICE)
    attn_mask = enc['attention_mask'].to(DEVICE)
    with torch.no_grad():
        out = bert_encoder(input_ids=input_ids, attention_mask=attn_mask)
        pooled = mean_pooling(out.last_hidden_state, attn_mask)  # shape (1, hidden)
    return pooled.reshape(-1)

def search_similar(query: str, top_k: int = 3,
                   tokenizer: BertTokenizer = None, bert_encoder: BertModel = None,
                   embeddings: np.ndarray = None, index: List[Dict] = None) -> List[Dict]:
    if embeddings is None or index is None:
        embeddings, index = load_embeddings()
    if tokenizer is None or bert_encoder is None:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_encoder = BertModel.from_pretrained(BERT_MODEL_NAME)
        bert_encoder.to(DEVICE)
    q_emb = get_embedding_for_text(query, tokenizer, bert_encoder).reshape(1, -1)
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        meta = index[int(idx)]
        results.append({
            "idx": int(idx),
            "text": meta.get("text",""),
            "label": meta.get("label",""),
            "score": float(sims[idx])
        })
    return results

# 5) Keyword extraction using attention (simple approach)
def extract_keywords_attention(text: str, tokenizer: BertTokenizer, bert_model: BertModel,
                               top_k: int = 5) -> List[str]:
    # tokenize
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LEN)
    input_ids = enc['input_ids'].to(DEVICE)
    attn_mask = enc['attention_mask'].to(DEVICE)

    # forward dengan output_attentions=True
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
        attentions = outputs.attentions

    attn_sum = None
    for layer_attn in attentions:
        head_mean = layer_attn.mean(dim=1)
        token_received = head_mean.sum(dim=1)
        if attn_sum is None:
            attn_sum = token_received
        else:
            attn_sum += token_received
    attn_avg = attn_sum / len(attentions)
    attn_avg = attn_avg.squeeze().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    mask = enc['attention_mask'][0].cpu().numpy()
    scores = []
    for i, tok in enumerate(tokens):
        if mask[i] == 0:
            continue
        if tok in tokenizer.all_special_tokens:
            continue
        scores.append((i, tok, float(attn_avg[i])))

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

    filtered = [(w, s) for w, s in merged if any(c.isalnum() for c in w)]
    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
    keywords = [w for w, s in filtered[:top_k]]
    keywords = [k.replace("▁","").strip() for k in keywords]
    return keywords

# 6) Summarization menggunakan T5
def load_t5_model(t5_dir: str = None):
    if t5_dir and os.path.exists(t5_dir):
        tokenizer = T5Tokenizer.from_pretrained(t5_dir)
        model = T5ForConditionalGeneration.from_pretrained(t5_dir)
    else:
        tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)
    model.to(DEVICE)
    return tokenizer, model

def summarize_text(text: str, t5_tokenizer: T5Tokenizer, t5_model: T5ForConditionalGeneration,
                   max_length: int = 60, min_length: int = 15) -> str:
    t5_model.eval()
    
    input_text = "summarize: " + text.strip()
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        summary_ids = t5_model.generate(inputs,
                                       num_beams=4,
                                       length_penalty=2.0,
                                       max_length=max_length,
                                       min_length=min_length,
                                       no_repeat_ngram_size=2,
                                       early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary

# 7) Prediction pipeline (classifier + extras)
def predict_all_features(text: str,
                         classifier_tokenizer: BertTokenizer,
                         classifier_model: BertForSequenceClassification,
                         classes: List[str],
                         bert_encoder: BertModel,
                         t5_tokenizer: T5Tokenizer,
                         t5_model: T5ForConditionalGeneration,
                         embeddings: np.ndarray = None,
                         index: List[Dict] = None,
                         top_k_sim: int = 3,
                         top_k_keywords: int = 5) -> Dict:

    # classification
    classifier_model.eval()
    enc = classifier_tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LEN)
    input_ids = enc['input_ids'].to(DEVICE)
    attn_mask = enc['attention_mask'].to(DEVICE)
    with torch.no_grad():
        outputs = classifier_model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
    category = classes[pred_id]

    # summarization
    summary = summarize_text(text, t5_tokenizer, t5_model)

    # keywords (attention using bert_encoder)
    keywords = extract_keywords_attention(text, classifier_tokenizer, bert_encoder, top_k=top_k_keywords)

    # similarity
    similar = search_similar(text, top_k=top_k_sim,
                             tokenizer=classifier_tokenizer, bert_encoder=bert_encoder,
                             embeddings=embeddings, index=index)

    return {
        "category": category,
        "confidence": confidence,
        "summary": summary,
        "keywords": keywords,
        "similar": similar
    }

# 8) Contoh main usage
def main_demo(csv_path="dataset_berita.csv"):
    # 1) load classifier untuk inference
    if not os.path.exists(CLASSIFIER_SAVE_DIR):
        print(f"Classifier model not found in {CLASSIFIER_SAVE_DIR}. Run train_bert_classifier(...) first.")
        return
    clf_tokenizer, clf_model, classes = load_classifier(CLASSIFIER_SAVE_DIR)
    # 2) load bert encoder untuk embeddings dan attention
    bert_encoder = BertModel.from_pretrained(BERT_MODEL_NAME)
    bert_encoder.to(DEVICE)
    # 3) load t5 untuk summarization
    t5_tokenizer, t5_model = load_t5_model()

    # 4) pastikan embeddings cached
    if not os.path.exists(EMBEDDING_CACHE) or not os.path.exists(EMBEDDING_INDEX):
        print("Embeddings cache not found. Building embeddings...")
        build_and_cache_embeddings(csv_path, bert_encoder=bert_encoder, tokenizer=clf_tokenizer)
    embeddings, index = load_embeddings()

    # 5) contoh input
    demo_texts = [
        "Samsung meluncurkan smartphone baru dengan chipset AI dan kamera canggih.",
        "Pemerintah mengumumkan paket stimulus ekonomi untuk UMKM.",
        "Timnas Indonesia berhasil memenangkan pertandingan persahabatan."
    ]
    for txt in demo_texts:
        print("\n--- INPUT ---")
        print(txt)
        out = predict_all_features(txt,
                                   classifier_tokenizer=clf_tokenizer,
                                   classifier_model=clf_model,
                                   classes=classes,
                                   bert_encoder=bert_encoder,
                                   t5_tokenizer=t5_tokenizer,
                                   t5_model=t5_model,
                                   embeddings=embeddings,
                                   index=index,
                                   top_k_sim=3,
                                   top_k_keywords=5)
        print("\nPREDIKSI KATEGORI:", out['category'], f"(confidence: {out['confidence']:.3f})")
        print("RINGKASAN:", out['summary'])
        print("KATA PENTING:", out['keywords'])
        print("BERITA MIRIP:")
        for r in out['similar']:
            print(f"  - ({r['score']:.3f}) {r['label']}: {r['text'][:120]}...")
    
# 9) CLI utility
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full Transformer IR backend")
    parser.add_argument("--mode", type=str, choices=["train","embed","demo"], default="demo",
                        help="train: fine-tune classifier, embed: build embeddings, demo: run demo pipeline")
    parser.add_argument("--data", type=str, default="dataset_berita.csv", help="path to dataset csv")
    parser.add_argument("--epochs", type=int, default=3, help="epochs for train")
    args = parser.parse_args()

    if args.mode == "train":
        print("Training BERT classifier...")
        train_bert_classifier(args.data, out_dir=CLASSIFIER_SAVE_DIR, epochs=args.epochs)
    elif args.mode == "embed":
        print("Building & caching embeddings...")
        tok = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        enc = BertModel.from_pretrained(BERT_MODEL_NAME)
        enc.to(DEVICE)
        build_and_cache_embeddings(args.data, bert_encoder=enc, tokenizer=tok)
    else:
        main_demo(args.data)
