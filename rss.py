import feedparser

rss_sources = {
    "teknologi": "https://www.cnbcindonesia.com/tech/rss",
    "ekonomi": "https://www.cnbcindonesia.com/news/rss",
    "lifestyle": "https://www.cnbcindonesia.com/lifestyle/rss"
}

data = []

for label, url in rss_sources.items():
    feed = feedparser.parse(url, agent="Mozilla/5.0")

    print(label, len(feed.entries))  # <-- ini harus keluar

    for i, entry in enumerate(feed.entries[:100]):
        title = entry.title if 'title' in entry else ""
        summary = entry.summary if 'summary' in entry else ""

        data.append({
            "text": title + " " + summary,
            "label": label,
            "url": entry.link
        })

import pandas as pd
df = pd.DataFrame(data)
df.to_csv("dataset_berita.csv", index=False)

print("Total berita:", len(df))
