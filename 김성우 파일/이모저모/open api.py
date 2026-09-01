import requests
from urllib.parse import unquote
import pandas as pd

url = "https://api.odcloud.kr/api/15037510/v1/uddi:f7adc849-309d-4d5a-a3da-be315760553d"

# 여기에 본인의 인증키 입력
service_key_encoded = "LTWObc6kT9FuzxMXSBVWNsD%2F2Xh3xeaDSxJuXCTbocfDhsxzDzCssUWufGzNS%2FN7s3WqI6THbVUWwcXgjIv35w%3D%3D"

# %2F, %3D 등으로 인코딩된 인증키를 디코딩
service_key = unquote(service_key_encoded)

params = {
    "page": 1,
    "perPage": 1000,
    "serviceKey": service_key
}

response = requests.get(url, params=params)

print(response.status_code)
print(response.text[:1000])


# JSON 응답을 파이썬 데이터로 변환
result = response.json()

# 실제 데이터 부분만 추출
df = pd.DataFrame(result["data"])

# 확인
print(df.head())
print(df.shape)

# CSV 저장
df.to_csv(
    "해운대구_방문객수.csv",
    index=False,
    encoding="utf-8-sig"
)

print("CSV 저장 완료!")