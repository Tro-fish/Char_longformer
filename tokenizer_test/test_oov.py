from tqdm import tqdm


def read_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return {line.strip() for line in file}


def tokenize_corpus(file_path, vocab):
    syllable_usage = {}
    total_used = set()
    oov_char = set()
    total_char = set()
    UnicodeDecodeError_count = 0

    oov_count = 0
    total_count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in tqdm(file):
            try:
                processed_line = line.strip()
                for char in processed_line:
                    total_char.add(char)
                    if char in vocab:
                        total_used.add(char)
                        if char in syllable_usage:
                            syllable_usage[char] += 1
                        else:
                            syllable_usage[char] = 1
                    else:
                        oov_char.add(char)
            except:
                UnicodeDecodeError_count+=1
                continue  # Skip lines that cause a UnicodeDecodeError
    print("UnicodeDecodeError_count:", UnicodeDecodeError_count)
    return total_used, syllable_usage, len(oov_char) / len(total_char) * 100, oov_char

# Paths to the vocab and corpus files
vocab_file_path = 'kakao_vocab.txt'
corpus_file_path = 'total_corpus.txt'

# Read the vocab and tokenize the corpus
vocab = read_vocab(vocab_file_path)
total_used, syllable_usage, oov_rate, oov_char = tokenize_corpus(corpus_file_path, vocab)

# Output the results
print("Number of syllables in vocab actually used:", len(total_used))
print("OOV rate:", oov_rate)
print("Number of OOV syllables:", (oov_char))
#print("Frequency of each used syllable:", syllable_usage[:100])