# 정렬된 DataFrame에서 'usage' 열의 값을 사용하여 등장 빈도를 계산합니다.
import pandas as pd

sorted_token_df = pd.read_csv('sorted_token_usage.csv')  # CSV 파일을 읽어서 DataFrame을 생성합니다.

# 전체 'usage'의 합계를 계산합니다.
total_usage = sorted_token_df['usage'].sum()

# 각 토큰의 등장 빈도를 계산합니다.
sorted_token_df['frequency'] = sorted_token_df['usage'] / total_usage

sorted_token_df.to_csv('sorted_token_usage.csv', index=False)  # 결과를 CSV 파일로 저장합니다.
# 계산된 등장 빈도가 포함된 DataFrame을 확인합니다.
print(sorted_token_df.head())  # 처음 몇 개의 행을 출력하여 결과 확인