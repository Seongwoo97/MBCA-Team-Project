import requests
import pandas as pd
from urllib.parse import unquote
import time

# ============================
# 기본 설정
# ============================

url = "https://apis.data.go.kr/B551011/AreaTarDemDsService/areaTarSjrnDsList"

encoded_key = "GANimFHCZX2R5NDfBLBhJtZK9D8qKyq%2BLbBE8QM9ju7V%2FONCovF6rE6gNj1xfqnEY7fV7I1C3TazYUGRLJiGIw%3D%3D"
service_key = unquote(encoded_key)


# 부산 16개 구군
busan_gu = {
    "중구": "26110",
    "서구": "26140",
    "동구": "26170",
    "영도구": "26200",
    "부산진구": "26230",
    "동래구": "26260",
    "남구": "26290",
    "북구": "26320",
    "해운대구": "26350",
    "사하구": "26380",
    "금정구": "26410",
    "강서구": "26440",
    "연제구": "26470",
    "수영구": "26500",
    "사상구": "26530",
    "기장군": "26710"
}


# 체류강도 전체 지표
indicators = {
    "2101": "타권역 방문자 비중",
    "2102": "숙박 비중",
    "2103": "1박 방문자수",
    "2104": "2박 방문자수",
    "2105": "3박 방문자수"
}


# ============================
# 조회할 월 설정
# ============================

months = pd.period_range(
    start="2025-01",
    end="2026-08",
    freq="M"
).strftime("%Y%m")


# ============================
# 데이터 수집
# ============================

all_data = []

for month in months:

    for gu_name, gu_code in busan_gu.items():

        for indicator_code, indicator_name in indicators.items():

            params = {
                "serviceKey": service_key,
                "numOfRows": 100,
                "pageNo": 1,
                "MobileOS": "ETC",
                "MobileApp": "BusanTourAnalysis",

                "baseYm": month,

                "areaCd": "26",
                "signguCd": gu_code,

                "tarSjrnDsIxCd": indicator_code,

                "_type": "json"
            }


            try:

                response = requests.get(
                    url,
                    params=params,
                    timeout=10
                )

                data = response.json()


                # 비정상 응답
                if "response" not in data:

                    print(
                        "API 오류:",
                        month,
                        gu_name,
                        indicator_name,
                        data
                    )

                    continue


                header = data["response"]["header"]

                if header["resultCode"] != "0000":

                    print(
                        "API 오류:",
                        month,
                        gu_name,
                        indicator_name,
                        header
                    )

                    continue


                body = data["response"]["body"]

                total_count = body.get(
                    "totalCount",
                    0
                )


                if total_count == 0:

                    print(
                        "데이터 없음:",
                        month,
                        gu_name,
                        indicator_name
                    )

                    continue


                items = body.get(
                    "items",
                    {}
                )


                if not items:
                    continue


                rows = items.get(
                    "item",
                    []
                )


                # 1개만 들어온 경우 대비
                if isinstance(rows, dict):
                    rows = [rows]


                all_data.extend(rows)


                print(
                    "수집:",
                    month,
                    gu_name,
                    indicator_name
                )


                # API에 너무 빠른 요청 방지
                time.sleep(0.1)


            except Exception as e:

                print(
                    "오류:",
                    month,
                    gu_name,
                    indicator_name,
                    e
                )


# ============================
# DataFrame
# ============================

df = pd.DataFrame(all_data)


print()
print("총 데이터:", len(df))
print(df.head())


# ============================
# 중복 제거
# ============================

df = df.drop_duplicates()


# ============================
# CSV 저장
# ============================

df.to_csv(
    "부산_관광체류강도_전체.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("CSV 저장 완료")