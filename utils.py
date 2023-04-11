import random
import string
import concurrent.futures
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from sklearn.model_selection import train_test_split
import re

nltk.download('stopwords')

def is_english(query):
    try:
        lang = detect(query)
        return lang == 'en'
    except:
        return False

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    processed_data = [line.strip().split('<SEP>')[-1] for line in data]

    english_data = []
    batch_size = 1000

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for idx in range(0, len(processed_data), batch_size):
            batch_queries = processed_data[idx: idx + batch_size]

            # Parallelize language detection
            results = list(executor.map(is_english, batch_queries))

            for query, is_en in zip(batch_queries, results):
                if is_en:
                    english_data.append(query)

            # Print progress
            print(f"Processed {idx + len(batch_queries)} lines out of {len(processed_data)}")

    return english_data

clean_chars = re.compile(r'[^A-Za-zöäüÖÄÜß,.!?’\'$%€0-9\(\)\- ]', re.MULTILINE)
def clean_data(data):
    cleaned_data = []
    for query in data:
        #query = ''.join(c for c in query if c not in string.punctuation)
        #query = query.lower()
        query = clean_chars.sub('', query)
        cleaned_data.append(query.strip())
    return cleaned_data

def remove_stopwords(data):
    stop_words = set(stopwords.words('english'))
    filtered_data = []
    for query in data:
        words = word_tokenize(query)
        query = ' '.join([word for word in words if word not in stop_words])
        filtered_data.append(query)
    return filtered_data

def replace_with_adjacent_char(word):
    adjacent_keys = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcxs', 'e': 'rdsw', 'f': 'rtgvcd',
        'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujko', 'j': 'uikmnh', 'k': 'iolmj', 'l': 'kop',
        'm': 'njkl', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol', 'q': 'wsa', 'r': 'edft', 's': 'wazxde',
        't': 'rfgy', 'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zasdc', 'y': 'tghu', 'z': 'asx'
    }
    if len(word) < 1:
        return word
    idx = random.randint(0, len(word)-1)
    char = word[idx]
    if char in adjacent_keys:
        word = word[:idx] + random.choice(adjacent_keys[char]) + word[idx+1:]
    return word

def introduce_errors(query, error_rate=0.1):
    def swap_adjacent_chars(word):
        if len(word) < 2:
            return word
        idx = random.randint(0, len(word)-2)
        return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]

    def remove_char(word):
        if len(word) < 2:
            return word
        idx = random.randint(0, len(word)-1)
        return word[:idx] + word[idx+1:]

    def insert_char(word):
        idx = random.randint(0, len(word))
        char = random.choice(string.ascii_lowercase)
        return word[:idx] + char + word[idx:]

    def combine_words(words):
        idx = random.randint(0, len(words)-2)
        words[idx] = words[idx] + words[idx+1]
        del words[idx+1]
        return words

    words = query.split()
    if len(words) == 0:
        return query
        
    if random.random() < error_rate:
        idx = random.randint(0, len(words)-1)
        error_type = random.choice(['swap', 'remove', 'insert', 'adjacent'])
        if error_type == 'swap':
            words[idx] = swap_adjacent_chars(words[idx])
        elif error_type == 'remove':
            words[idx] = remove_char(words[idx])
        elif error_type == 'insert':
            words[idx] = insert_char(words[idx])
        elif error_type == 'adjacent':
            words[idx] = replace_with_adjacent_char(words[idx])
        if len(words) > 1 and random.random() < error_rate:
            words = combine_words(words)
    return ' '.join(words)

def generate_dataset(data, error_rate=0.1):
    pairs = []
    for query in data:
        if not query.strip():
            continue
        erroneous_query = introduce_errors(query, error_rate)
        pairs.append((erroneous_query, query))
    return pairs

def split_dataset(pairs, train_size=0.985, val_size=0.01, test_size=0.05):
    train_data, test_val_data = train_test_split(pairs, train_size=train_size)
    val_data, test_data = train_test_split(test_val_data, test_size=test_size/(val_size+test_size))
    return train_data, val_data, test_data

def truncate_dataset(data_en, max_len):
    data = []
    for rec in data_en:
        reclen = len(rec.split())
        if reclen <= max_len:
            data.append(rec)
        else:
            data.append(' '.join(rec.split()[:max_len]))
    return data

def save_to_file(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def load_english_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data    

def save_data_to_file(data, file_name):
    with open(file_name, 'w') as f:
        json.dump([list(pair) for pair in data], f)

def load_data_from_file(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return [tuple(pair) for pair in data]



def freeze_params(model):
    for params in model.parameters():
        params.requires_grad = False

