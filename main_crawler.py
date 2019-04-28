import os

from crawl import pipeline

if __name__ == "__main__":
    base_url = 'https://content.guardianapis.com/search?'
    api_key = os.getenv('THEAPIGUARDIAN')
    categories = {'film', 'technology', 'travel', 'food', 'business', 'fashion', 'education', 'artanddesign',
                  'football', 'games'}
    incremental_label = 'page'
    key_label = 'api-key'
    query_label = 'q'

    articles_per_cat = 200

    output_path = './data/articles_'
    pipeline(base_url, api_key, categories, incremental_label, key_label, query_label, articles_per_cat, output_path)
