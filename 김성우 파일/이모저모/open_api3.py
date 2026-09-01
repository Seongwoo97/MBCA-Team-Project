import requests
import pandas as pd
from urllib.parse import unquote
import time

url = "https://apis.data.go.kr/B551011/AreaTarDemDsService/areaTarExpDsList"

encoded_key = "GANimFHCZX2R5NDfBLBhJtZK9D8qKyq%2BLbBE8QM9ju7V%2FONCovF6rE6gNj1xfqnEY7fV7I1C3TazYUGRLJiGIw%3D%3D"
service_key = unquote(encoded_key)


# 부산 구·군
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


# 관광 소비 강도 지표
indicators = {
    "2201": "외지인 소비액",
    "2202": "전체 대비 외지인 소비액 비중",
    "2203": "방문량 대비 방문 소비액"
}


# 조회 기간
months = pd.period_range(
    start="2025-01",
    end="2026-07",
    freq="M"
).strftime("%Y%m")


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

                "tarExpDsIxCd": indicator_code,

                "_type": "json"
            }


            try:

                response = requests.get(
                    url,
                    params=params,
                    timeout=10
                )

                data = response.json()


                # 비정상 API 응답
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


                # 1건만 반환되는 경우
                if isinstance(rows, dict):
                    rows = [rows]


                all_data.extend(rows)


                print(
                    "수집 완료:",
                    month,
                    gu_name,
                    indicator_name
                )


                time.sleep(0.1)


            except Exception as e:

                print(
                    "오류:",
                    month,
                    gu_name,
                    indicator_name,
                    e
                )


# =========================
# DataFrame 생성
# =========================

df = pd.DataFrame(all_data)


# 중복 제거
df = df.drop_duplicates()


print()
print("총 데이터 수:", len(df))

print(df.head())


# =========================
# CSV 저장
# =========================

df.to_csv(
    "부산_관광소비강도_전체.csv",
    index=False,
    encoding="utf-8-sig"
)

print("CSV 저장 완료")