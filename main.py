"""
IMPORTANT:

To make DeGuardianBoi run it is necessary to obtain the API key from The Guardian and set it in the
THEAPIGUARDIAN environment variable.
"""
import json
import math
import os
from tqdm import tqdm

from deguardianboi import DeGuardianBoi

base_url = 'https://content.guardianapis.com/search?'
api_key = os.getenv('THEAPIGUARDIAN')
categories = ['film', 'technology', 'travel', 'food', 'business', 'fashion', 'education', 'artanddesign', 'football',
              'games']
incremental_label = 'page'
key_label = 'api-key'
query_label = 'q'

articles_per_cat = 200

output_path = './data/articles_'


def concat_params(params):
    return '&'.join([(str(key) + '=' + str(val)).lower() for key, val in params.items()])


def get_n_pages(total_articles, articles_per_page=10):
    return max(math.ceil(total_articles / articles_per_page), 1)


dgb = DeGuardianBoi()
data = {}
with tqdm(total=articles_per_cat * len(categories), desc=categories[0]) as pbar:
    for c in categories:
        data[c] = {}
        pbar.desc = c
        for i in range(1, 1 + get_n_pages(articles_per_cat)):
            url = base_url + concat_params({
                key_label: api_key,
                query_label: c,
                incremental_label: i
            })
            res = dgb.fetch_json(url)
            for r in res['response']['results']:
                url = r['webUrl']
                # print(f'Fetching: {url}')
                corpus = dgb.fetch_article(url)
                if corpus is not False:
                    r['corpus'] = corpus
                    data[c][r['id']] = r
                else:
                    print(f'The corpus could not be loaded. The article {url} has been skipped.')
                pbar.update(1)
        comp_path = output_path + c + '.json'
        with open(comp_path, 'w') as fout:
            json.dump(data[c], fout)
            print(f"\nDumped {comp_path} successfully.")
