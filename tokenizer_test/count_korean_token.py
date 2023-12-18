import pandas as pd
from transformers import MT5Tokenizer
from collections import Counter
from tqdm import tqdm

# 한국어 문자를 포함하는지 확인하는 함수
def is_korean(token):
    return any('\uAC00' <= char <= '\uD7A3' for char in token)

# 파일의 총 라인 수를 계산하는 함수
def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return sum(1 for _ in file)

# 토큰화 및 토큰 사용 빈도 추적 (파일에서 한 줄씩 읽어오는 버전)
def analyze_korean_token_usage(file_path, tokenizer, korean_vocab, total_lines):
    token_counter = Counter()
    total_korean_tokens = 0

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in tqdm(file, total=total_lines):
            tokens = tokenizer.tokenize(line)
            for token in tokens:
                if token in korean_vocab:
                    token_counter[token] += 1
                    total_korean_tokens += 1

    return token_counter, total_korean_tokens

# 한국어 토큰 사용률 계산 및 CSV 파일로 저장
def calculate_and_save_usage(token_counter, total_korean_tokens, output_file):
    data = []
    for token, count in token_counter.items():
        usage_percentage = (count / total_korean_tokens) * 100
        data.append([token, usage_percentage])

    df = pd.DataFrame(data, columns=['used_korean_token', 'used_percentage'])
    df.to_csv(output_file, index=False)

# 메인 함수
def main(corpus_file, korean_vocab_file, output_file):
    # MT5 tokenizer 로드
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

    # 한국어 토큰 목록 불러오기
    with open(korean_vocab_file, 'r', encoding='utf-8') as file:
        korean_vocab = set(line.strip() for line in file if is_korean(line))

    # 파일의 총 라인 수 계산
    total_lines = count_lines(corpus_file)

    # 데이터셋을 토큰화하고 한국어 토큰 사용 추적
    token_counter, total_korean_tokens = analyze_korean_token_usage(corpus_file, tokenizer, korean_vocab, total_lines)

    # 사용률 계산 및 CSV 파일로 저장
    calculate_and_save_usage(token_counter, total_korean_tokens, output_file)

# 실제 파일 경로
corpus_file_path = 'total_corpus.txt' # 실제 텍스트 파일 경로로 변경 필요
korean_vocab_file_path = 'korean_vocab.txt' # 실제 순수 한국어 토큰 파일 경로로 변경 필요
output_csv_file = 'korean_token_usage.csv'

# 메인 함수 실행
main(corpus_file_path, korean_vocab_file_path, output_csv_file)