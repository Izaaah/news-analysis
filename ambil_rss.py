import feedparser
import pandas as pd

# RSS stabil
rss_sources = {
    "teknologi": "https://www.cnbcindonesia.com/tech/rss",
    "ekonomi": "https://www.cnbcindonesia.com/news/rss",
    "lifestyle": "https://www.cnbcindonesia.com/lifestyle/rss"
}

data = []

headers = {'User-Agent': 'Mozilla/5.0'}

for label, url in rss_sources.items():
    feed = feedparser.parse(url, request_headers=headers)
    print(label, len(feed.entries))

    count = 0
    for entry in feed.entries:
        if count >= 50:
            break

        title = entry.title if 'title' in entry else ""
        summary = entry.summary if 'summary' in entry else ""
        text = title + " " + summary

        data.append({
            "text": text,
            "label": label
        })

        count += 1

df = pd.DataFrame(data)
df.to_csv("dataset_berita.csv", index=False)
print("Total berita:", len(df))
