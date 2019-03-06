import json
import random

users_path = './data/users.json'
categories = {'film', 'technology', 'travel', 'food', 'business', 'fashion', 'education', 'artanddesign', 'football',
              'games'}
n_users = 500
min_cat, max_cat = 2, 5
users = {}

for i in range(n_users):
    n_interests = random.randint(min_cat, max_cat)
    interests = set()
    for _ in range(n_interests):
        interests.add(random.choice(list(categories - interests)))
    users[str(i)] = list(interests)
with open(users_path, 'w') as fout:
    json.dump(users, fout)
    print(f"Users dumped successfully.")
