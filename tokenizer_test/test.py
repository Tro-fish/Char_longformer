import re
import numpy as np

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        data = file.read()

    # 로짓 값 추출
    logits = re.findall(r'logits=tensor\(.*?\)', data)
    print(len(logits))
    # 최대값의 인덱스 추출 (무작위 선택)
    max_indices = []
    for logit in logits:
        logit_values = np.fromstring(logit, sep=', ')
        max_value = np.max(logit_values)
        # 최대값과 같은 모든 인덱스를 찾음
        max_indices_candidates = np.where(logit_values == max_value)[0]
        # 무작위로 하나 선택
        max_index = np.random.choice(max_indices_candidates)
        max_indices.append(max_index)

    # 결과를 새 파일에 저장
    with open(output_file_path, 'w') as file:
        for index in max_indices:
            file.write(f'{index}\n')

# 사용 예시
input_file_path = 'predictions.txt'  # 입력 파일 경로
output_file_path = 'max_indices_output.txt'  # 출력 파일 경로

process_file(input_file_path, output_file_path)
