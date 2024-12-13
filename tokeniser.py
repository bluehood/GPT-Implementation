from collections import defaultdict
import re
import json

def clean_dataset(dataset):
    dataset = dataset.lower()
    dataset = re.sub(r'\n', ' ', dataset)
    dataset = re.sub(r'[^a-z0-9.,!?;\'\" ]', ' ', dataset)
    return dataset

def basic_tokenize(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z0-9.,!?;\'\" ]', ' ', text)
    tokens = text.split()
    return tokens

def create_vocabulary(tokens, min_frequency=2):
    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3,
    }
    
    token_idx = len(vocab)
    for token, count in token_counts.items():
        if count >= min_frequency:
            vocab[token] = token_idx
            token_idx += 1
    
    return vocab

def encode_text(text, vocab):
    tokens = basic_tokenize(text)
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return encoded

def decode_text(encoded, vocab):
    tokens = [vocab.get(idx, '<UNK>') for idx in encoded]
    text = ' '.join(tokens)
    return text

def save_tokeniser(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_tokeniser(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    with open("./datasets/alice_in_wonderland.txt", "r") as f:
        dataset = f.read()
    
    dataset = clean_dataset(dataset)
    tokens = basic_tokenize(dataset)
    vocab = create_vocabulary(tokens)
    save_tokeniser(vocab, "./models/alice_in_wonderland_tokeniser.json")

def debug():
    import debugpy
    debugpy.listen(('127.0.0.1', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    return

if __name__ == "__main__":
    debug()
    main()