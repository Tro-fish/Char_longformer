# 코드 실행 상태가 초기화되었으므로 필요한 라이브러리를 다시 임포트하고 파일을 다시 읽어야 합니다.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 정렬된 CSV 파일을 다시 읽어옵니다.
df = pd.read_csv('sorted_token_usage.csv')

# Box-Cox 변환을 위해 0보다 큰 값이 필요합니다.
# 'frequency' 값 중 0이 있는지 확인하고, 0보다 큰 최소값을 찾습니다.
min_positive_freq = df[df['frequency'] > 0]['frequency'].min()

# 0인 값들을 0보다 조금 더 큰 값으로 대체합니다.
df['frequency'] = df['frequency'].replace(0, min_positive_freq / 2)

# Box-Cox 변환 수행
df['frequency_transformed'], _ = stats.boxcox(df['frequency'])

# 전처리된 데이터의 분포를 확인하기 위해 정규분포와 비교하는 그래프를 그립니다.
plt.figure(figsize=(10, 6))
plt.hist(df['frequency_transformed'], bins=50, density=True, alpha=0.6, color='g')

# 분포의 평균과 표준편차를 계산합니다.
mean_transformed = df['frequency_transformed'].mean()
std_dev_transformed = df['frequency_transformed'].std()

# 정규분포 그래프 그리기 (비교용)
x_transformed = np.linspace(min(df['frequency_transformed']), max(df['frequency_transformed']), 100)
y_transformed = stats.norm.pdf(x_transformed, mean_transformed, std_dev_transformed)
plt.plot(x_transformed, y_transformed, 'k', linewidth=2)

title = "Transformed Frequency Distribution"
plt.title(title)
plt.xlabel('Transformed Frequency')
plt.ylabel('Density')
plt.show()


def select_tokens_and_save(df, num_std_dev, file_path):
    """
    주어진 표준편차 범위 내의 토큰을 선택하고 .txt 파일로 저장하는 함수.
    """
    lower_bound = mean_transformed - num_std_dev * std_dev_transformed
    upper_bound = mean_transformed + num_std_dev * std_dev_transformed
    selected_tokens = df[(df['frequency_transformed'] >= lower_bound)]

    # 선택된 토큰들의 'token' 값을 .txt 파일로 저장
    with open(file_path, 'w', encoding='utf-8') as file:
        for token in selected_tokens['token']:
            file.write(token + '\n')

    return file_path

# 사용 예시 (2 표준편차 범위 내의 토큰 선택)
file_path = 'selected_tokens.txt'
selected_tokens_file = select_tokens_and_save(df, 0.5, file_path)