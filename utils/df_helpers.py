import os
import json
import pandas as pd

def get_AFND_dataframe(path):
    # Paths
    dataset_path =  os.path.join(path, 'Dataset')
    sources_file = os.path.join(path, 'sources.json')

    # Load sources credibility info
    with open(sources_file, 'r') as f:
        sources_info = json.load(f)

    # List to collect all articles
    articles_data = []

    # Traverse each source folder in the Dataset directory
    for source_folder in os.listdir(dataset_path):
        source_path = os.path.join(dataset_path, source_folder)

        if not os.path.isdir(source_path):
            continue

        # Get credibility from sources.json
        credibility = sources_info.get(source_folder, 'unknown')

        # Path to the scrapped_articles.json
        articles_file = os.path.join(source_path, 'scraped_articles.json')

        if os.path.exists(articles_file):
            with open(articles_file, 'r') as f:
                try:
                    data = json.load(f)
                    articles = data.get('articles', [])
                except json.JSONDecodeError:
                    print(f"Could not decode JSON for {articles_file}")
                    continue

                for article in articles:
                    articles_data.append({
                        'source': source_folder,
                        'credibility': credibility,
                        'title': article.get('title', ''),
                        'text': article.get('text', ''),
                        'published_date': article.get('published date', '')
                    })

    # Convert to DataFrame
    df = pd.DataFrame(articles_data)

    return df