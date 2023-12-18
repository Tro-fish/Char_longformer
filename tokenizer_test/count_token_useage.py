
from collections import Counter
import pandas as pd
from tqdm import tqdm

# 파일의 총 라인 수를 계산하는 함수
def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return sum(1 for _ in file)
    
def process_text_corpus(corpus_path, vocab_path):
    """
    텍스트 코퍼스 파일을 읽고, 주어진 어휘집(vocab)을 사용하여 토큰 사용 횟수를 계산하고
    OOV 비율을 계산하는 함수.
    """
    # 파일의 총 라인 수 계산
    total_lines = count_lines(corpus_path)

    # 어휘집 파일 읽기
    with open(vocab_path, 'r', encoding='utf-8', errors='ignore') as file:
        vocab = set(file.read().splitlines())

    # 토큰 및 OOV 카운트를 위한 준비
    token_counts = Counter()
    oov_count = 0
    total_count = 0

    # 코퍼스 파일 읽기 및 처리
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in tqdm(file, total=total_lines):
            # 음절 단위로 토큰화
            tokens = list(line.strip())
            total_count += len(tokens)
            for token in tokens:
                if token in vocab:
                    token_counts[token] += 1
                else:
                    oov_count += 1

    # 결과 저장을 위한 DataFrame 생성
    token_df = pd.DataFrame(token_counts.items(), columns=['token', 'usage'])
    csv_file_path = 'token_usage.csv'
    token_df.to_csv(csv_file_path, index=False)

    # OOV 비율 계산
    oov_rate = oov_count / total_count if total_count > 0 else 0

    return csv_file_path, oov_rate

# 이 코드는 'text_corpus.txt' 파일과 'kakao_vocab.txt' 파일이 필요합니다.
# 코드 사용 예시:
csv_file, oov_rate = process_text_corpus('/Users/waniboyy/VSCODE/Char_longformer/total_corpus.txt', 'selected_tokens.txt')
print(oov_rate)
# 이 코드는 'text_corpus.txt' 파일이 없으므로 실행할 수 없습니다.
# 파일이 제공되면, 이 코드를 사용하여 원하는 결과를 얻을 수 있습니다.
