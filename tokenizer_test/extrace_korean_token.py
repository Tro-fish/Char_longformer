# 먼저 제공된 파일을 읽고 한국어 토큰만 필터링하여 새 파일로 저장하겠습니다.

input_vocab_file = '/mnt/data/mt5_vocab.txt'
output_korean_vocab_file = '/mnt/data/korean_vocab.txt'

# 한국어 문자를 포함하는지 확인하는 함수
def is_korean(token):
    return any('\uAC00' <= char <= '\uD7A3' for char in token)

# 입력 파일에서 어휘 목록을 읽고 한국어 토큰만 필터링합니다.
with open(input_vocab_file, 'r', encoding='utf-8') as file:
    korean_vocab = [line.strip() for line in file if is_korean(line)]

# 한국어 토큰의 수
korean_vocab_count = len(korean_vocab)

# 새 파일에 한국어 어휘 목록을 저장합니다.
with open(output_korean_vocab_file, 'w', encoding='utf-8') as file:
    for vocab in korean_vocab:
        file.write(vocab + '\n')

korean_vocab_count, output_korean_vocab_file