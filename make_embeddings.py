import os
import json
import numpy as np
import pandas as pd
from model import get_embedding

def build_and_cache_embeddings(csv_path):
    df = pd.read_csv(csv_path)

    os.makedirs("embeddings", exist_ok=True)

    all_embeddings = []
    info = []

    for i, row in df.iterrows():
        text = row["text"]
        label = row["label"]
        # Coba ambil URL dari kolom 'url' atau 'link'
        url = row.get("url") or row.get("link") or None

        emb = get_embedding(text)
        all_embeddings.append(emb)

        info.append({
            "text": text,
            "label": label,
            "url": url,
            "link": url  # Backward compatibility
        })

        print(f"Embedding {i+1}/{len(df)} selesai")

    np.save("embeddings/embeddings.npy", np.array(all_embeddings))

    with open("embeddings/emb_index.json", "w") as f:
        json.dump(info, f, indent=2)

    print("Selesai! Embeddings + index disimpan di folder embeddings/")

if __name__ == "__main__":
    build_and_cache_embeddings("dataset_berita.csv")