from transformers import MT5Tokenizer

# MT5 tokenizer 로드
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

# 어휘 사전 가져오기
vocab = tokenizer.get_vocab()

# 어휘 개수 출력
vocab_size = len(vocab)
print("MT5 tokenizer의 전체 어휘 개수:", vocab_size)