import pandas as pd
from transformers import MT5Tokenizer
from collections import Counter
from tqdm import tqdm

# 파일의 총 라인 수를 계산하는 함수
def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return sum(1 for _ in file)

# 토큰화 및 토큰 사용 빈도 추적 (파일에서 한 줄씩 읽어오는 버전)
def analyze_token_usage(file_path, tokenizer, total_lines):
    token_counter = Counter()
    total_tokens = 0

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in tqdm(file, total=total_lines):
            tokens = tokenizer.tokenize(line)
            for token in tokens:
                token_counter[token] += 1
                total_tokens += 1

    return token_counter, total_tokens

# 메인 함수
def main(corpus_file, output_file):
    # MT5 tokenizer 로드 및 모든 토큰 추출
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

    # 파일의 총 라인 수 계산
    total_lines = count_lines(corpus_file)

    # 데이터셋을 토큰화하고 모든 토큰 사용 추적
    token_counter, total_tokens = analyze_token_usage(corpus_file, tokenizer, total_lines)

    # 사용률 계산 및 CSV 파일로 저장
    data = [(token, (count / total_tokens) * 100) for token, count in token_counter.items()]
    df = pd.DataFrame(data, columns=['used_token', 'used_percentage'])
    df.to_csv(output_file, index=False)

# 파일 경로
corpus_file_path = 'total_corpus.txt' # 실제 텍스트 파일 경로로 변경 필요
output_csv_file = 'token_usage.csv'

# 메인 함수 실행
main(corpus_file_path, output_csv_file)
