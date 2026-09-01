# ==============================================================
# 부산 사업자 증감 분석 HTML Dashboard - V9
# - Colab / ipywidgets 사용 안 함
# - Plotly.js를 HTML 내부에 1회만 삽입
# - 데이터 상수(DATA 등)도 1회만 선언
# - 지도는 외부 타일이 필요 없는 Plotly choropleth 사용
# - 생성된 HTML을 파일:// 로 직접 열어도 동작하도록 구성
# ==============================================================

from pathlib import Path
import json
import webbrowser

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.offline as pyo

try:
    from scipy.stats import friedmanchisquare, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    friedmanchisquare = None
    spearmanr = None
    SCIPY_AVAILABLE = False


# ==============================================================
# 0. 경로 설정
# ==============================================================

DATA_PATH = Path(
    r"C:\Users\Admin\mbca\1차 팀과제\데이터\부산_사업자현황_통합_202201_202606_업종6분류.csv"
)

DATA_DIR = DATA_PATH.parent
OUTPUT_PATH = DATA_DIR / "부산_사업자_증감대시보드_V9.html"

# SHP 파일을 직접 지정하려면 아래 None 대신 경로를 입력하세요.
# 예:
# SHP_PATH = Path(r"C:\Users\Admin\mbca\1차 팀과제\데이터\부산행정구역\부산_구군.shp")
SHP_PATH = Path(r"C:\Users\Admin\mbca\1차 팀과제\데이터\BND_SIGUNGU_PG\BND_SIGUNGU_PG.shp")


BUSAN_REGIONS = [
    "중구", "서구", "동구", "영도구",
    "부산진구", "동래구", "남구", "북구",
    "해운대구", "사하구", "금정구", "강서구",
    "연제구", "수영구", "사상구", "기장군",
]

INDUSTRY_ORDER = [
    "전체",
    "쇼핑업",
    "서비스업",
    "식음료업",
    "의료웰니스업",
    "운송업",
    "여행/숙박업",
]


# ==============================================================
# 1. CSV 읽기
# ==============================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949", "utf-8"]

    last_error = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"[OK] CSV 인코딩: {encoding}")
            return df
        except UnicodeDecodeError as e:
            last_error = e

    raise ValueError(
        "CSV 파일의 인코딩을 읽지 못했습니다."
    ) from last_error


if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다:\n{DATA_PATH}")


df = read_csv_auto(DATA_PATH)
print(f"[OK] CSV 불러오기 완료: {df.shape[0]:,}행 × {df.shape[1]}열")


# ==============================================================
# 2. 데이터 전처리
# ==============================================================

raw = df.copy()

required_columns = [
    "연월", "구분1", "구분3", "업종대분류",
    "당월①", "전월②", "전년동월③",
]

missing_columns = [c for c in required_columns if c not in raw.columns]
if missing_columns:
    raise KeyError(
        "CSV에 필요한 컬럼이 없습니다: " + ", ".join(missing_columns)
    )


# 숫자형 변환
for column in ["당월①", "전월②", "전년동월③"]:
    raw[column] = (
        raw[column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    raw[column] = pd.to_numeric(raw[column], errors="coerce")


# 날짜 변환
raw["연월"] = (
    raw["연월"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.strip()
    .str.zfill(6)
)

raw["기준일"] = pd.to_datetime(
    raw["연월"],
    format="%Y%m",
    errors="coerce",
)

raw = raw[raw["기준일"].notna()].copy()

raw["연도"] = raw["기준일"].dt.year.astype(int)
raw["월"] = raw["기준일"].dt.month.astype(int)
raw["연월표시"] = raw["기준일"].dt.strftime("%Y-%m")


# 지역명 표준화
raw["구분1"] = (
    raw["구분1"]
    .astype(str)
    .str.replace("부산광역시", "", regex=False)
    .str.replace("부산시", "", regex=False)
    .str.strip()
)

raw = raw[raw["구분1"].isin(BUSAN_REGIONS)].copy()

if raw.empty:
    raise ValueError(
        "부산 16개 구·군 데이터가 한 행도 남지 않았습니다. "
        "CSV의 '구분1' 값을 확인해주세요."
    )

print(
    "[OK] 분석기간:",
    raw["연월표시"].min(),
    "~",
    raw["연월표시"].max(),
)
print(f"[OK] 지역 수: {raw['구분1'].nunique()}개")


# ==============================================================
# 3. 전체 업종 데이터
# ==============================================================

overall = (
    raw[raw["구분3"].astype(str).str.strip() == "업종전체"]
    [[
        "연월", "기준일", "연도", "월", "연월표시",
        "구분1", "당월①", "전월②", "전년동월③",
    ]]
    .rename(
        columns={
            "구분1": "지역",
            "당월①": "사업자수",
            "전월②": "전월사업자수",
            "전년동월③": "전년동월사업자수",
        }
    )
)

overall["업종"] = "전체"

if overall.empty:
    raise ValueError(
        "'구분3 == 업종전체' 데이터가 없습니다. CSV의 구분3 값을 확인해주세요."
    )


# ==============================================================
# 4. 6개 업종대분류 집계
# ==============================================================

major_source = raw[
    raw["업종대분류"].notna()
    & raw["업종대분류"].astype(str).str.strip().ne("")
].copy()

major_source["업종대분류"] = major_source["업종대분류"].astype(str).str.strip()

major = (
    major_source
    .groupby(
        [
            "연월", "기준일", "연도", "월", "연월표시",
            "구분1", "업종대분류",
        ],
        as_index=False,
    )
    .agg(
        사업자수=("당월①", "sum"),
        전월사업자수=("전월②", "sum"),
        전년동월사업자수=("전년동월③", "sum"),
    )
    .rename(
        columns={
            "구분1": "지역",
            "업종대분류": "업종",
        }
    )
)


# ==============================================================
# 5. 전체 + 업종 통합
# ==============================================================

viz = pd.concat([overall, major], ignore_index=True)

# 화면에서 사용할 업종만 유지
viz = viz[viz["업종"].isin(INDUSTRY_ORDER)].copy()

viz = viz.sort_values(
    ["지역", "업종", "기준일"]
).reset_index(drop=True)

print("[OK] 대시보드 업종:", sorted(viz["업종"].unique().tolist()))
print(f"[OK] 대시보드 분석행: {len(viz):,}행")


# ==============================================================
# 6. 성장지표 계산
# ==============================================================

viz["전월성장률"] = np.where(
    viz["전월사업자수"] > 0,
    (viz["사업자수"] / viz["전월사업자수"] - 1) * 100,
    np.nan,
)

viz["전년동월성장률"] = np.where(
    viz["전년동월사업자수"] > 0,
    (viz["사업자수"] / viz["전년동월사업자수"] - 1) * 100,
    np.nan,
)

# 각 지역 × 업종의 첫 관측월을 기준 100으로 설정
viz["기준월사업자수"] = (
    viz.groupby(["지역", "업종"])["사업자수"]
    .transform("first")
)

viz["성장지수"] = np.where(
    viz["기준월사업자수"] > 0,
    viz["사업자수"] / viz["기준월사업자수"] * 100,
    np.nan,
)

viz["누적성장률"] = viz["성장지수"] - 100


# ==============================================================
# 7. 가설 검증 (현재 보유 데이터만 사용)
# ==============================================================
# 이 데이터는 신규사업자/폐업사업자 건수를 각각 분리한 자료가 아닙니다.
# 따라서 '사업자 수의 순증감'과 '업종별 증감 패턴'에 맞춰 검증합니다.


def format_p_value(p):
    """로그/상세 통계용 p-value 표기."""
    if p is None or not np.isfinite(p):
        return "-"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def format_p_friendly(p):
    """대시보드 메인 화면용, 비전공자 친화적 p-value 표기."""
    if p is None or not np.isfinite(p):
        return "-"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def make_hypothesis_results(viz_df):
    if not SCIPY_AVAILABLE:
        common = {
            "status": "검정 미실행",
            "status_class": "neutral",
            "headline": "통계 검정을 실행하지 못했습니다.",
            "interpretation": "SciPy 설치 후 검정 결과를 확인할 수 있습니다.",
            "evidence": [
                ("필요 작업", "pip install scipy"),
            ],
            "stat_detail": "SciPy 미설치",
        }
        return {
            "h1": {"kicker": "H1 · 지역 차이", "title": "지역별 사업자 증감률에는 차이가 있을 것이다.", **common},
            "h2": {"kicker": "H2 · 업종 차이", "title": "업종대분류별 사업자 증감 양상은 다를 것이다.", **common},
            "h3": {"kicker": "H3 · 단기/장기 흐름", "title": "단기 증감과 장기 누적 변화는 항상 같은 방향으로 움직이지 않을 것이다.", **common},
        }

    # H1: 월별 반복 관측 안에서 16개 지역의 전체 YoY 비교
    h1_pivot = (
        viz_df[(viz_df["업종"] == "전체") & viz_df["전년동월성장률"].notna()]
        .pivot(index="연월표시", columns="지역", values="전년동월성장률")
        .reindex(columns=BUSAN_REGIONS)
        .dropna()
    )
    if len(h1_pivot) >= 2 and h1_pivot.shape[1] == len(BUSAN_REGIONS):
        h1_stat, h1_p = friedmanchisquare(*[h1_pivot[c].to_numpy() for c in h1_pivot.columns])
        h1_supported = bool(h1_p < 0.05)
        h1_status = "지지" if h1_supported else "기각"
        h1_class = "supported" if h1_supported else "rejected"
        if h1_supported:
            h1_headline = "지역별 사업자 증감률 차이가 뚜렷하게 나타났습니다."
            h1_interpretation = "부산 16개 구·군이 모두 같은 흐름으로 움직인 것이 아니라, 지역마다 증가·감소 패턴에 차이가 있었습니다."
        else:
            h1_headline = "지역별 차이를 통계적으로 뚜렷하다고 보기 어려웠습니다."
            h1_interpretation = "현재 관측 범위에서는 부산 16개 구·군의 증감률 차이가 충분히 명확하게 확인되지 않았습니다."
        h1_evidence = [
            ("검정 방법", "Friedman 검정"),
            ("유의확률", format_p_friendly(h1_p)),
            ("비교 범위", f"{len(h1_pivot)}개월 × 16개 구·군"),
        ]
        h1_stat_detail = f"χ² = {h1_stat:.2f} · 정확한 p-value = {format_p_value(h1_p)}"
    else:
        h1_status, h1_class = "판정 보류", "neutral"
        h1_headline = "지역별 차이를 판정할 공통 관측이 부족합니다."
        h1_interpretation = "동일한 월에 16개 구·군을 함께 비교할 수 있는 관측치가 충분하지 않습니다."
        h1_evidence = [("검정 상태", "공통 관측 부족")]
        h1_stat_detail = "검정 미실행"

    # H2: 같은 지역×월 안에서 6개 업종의 YoY 비교
    six_industries = [x for x in INDUSTRY_ORDER if x != "전체"]
    h2_pivot = (
        viz_df[viz_df["업종"].isin(six_industries) & viz_df["전년동월성장률"].notna()]
        .pivot_table(index=["지역", "연월표시"], columns="업종", values="전년동월성장률", aggfunc="first")
        .reindex(columns=six_industries)
        .dropna()
    )
    if len(h2_pivot) >= 2 and h2_pivot.shape[1] == len(six_industries):
        h2_stat, h2_p = friedmanchisquare(*[h2_pivot[c].to_numpy() for c in h2_pivot.columns])
        h2_supported = bool(h2_p < 0.05)
        h2_status = "지지" if h2_supported else "기각"
        h2_class = "supported" if h2_supported else "rejected"
        if h2_supported:
            h2_headline = "업종대분류별 사업자 증감 패턴이 서로 다르게 나타났습니다."
            h2_interpretation = "같은 지역·시점이라도 쇼핑, 서비스, 식음료 등 업종에 따라 증가·감소 흐름이 동일하지 않았습니다."
        else:
            h2_headline = "업종별 차이를 통계적으로 뚜렷하다고 보기 어려웠습니다."
            h2_interpretation = "현재 관측 범위에서는 6개 업종대분류의 증감 패턴 차이가 충분히 명확하게 확인되지 않았습니다."
        h2_evidence = [
            ("검정 방법", "Friedman 검정"),
            ("유의확률", format_p_friendly(h2_p)),
            ("비교 범위", f"{len(h2_pivot):,}개 지역×월 × 6개 업종"),
        ]
        h2_stat_detail = f"χ² = {h2_stat:.2f} · 정확한 p-value = {format_p_value(h2_p)}"
    else:
        h2_status, h2_class = "판정 보류", "neutral"
        h2_headline = "업종별 차이를 판정할 공통 관측이 부족합니다."
        h2_interpretation = "동일한 지역·월에서 6개 업종을 함께 비교할 수 있는 관측치가 충분하지 않습니다."
        h2_evidence = [("검정 상태", "공통 관측 부족")]
        h2_stat_detail = "검정 미실행"

    # H3: 전체 업종의 MoM(단기) vs 누적증감률(장기)
    h3_df = viz_df[(viz_df["업종"] == "전체") & viz_df["전월성장률"].notna() & viz_df["누적성장률"].notna()].copy()
    if len(h3_df) >= 3:
        h3_rho, h3_p = spearmanr(h3_df["전월성장률"].to_numpy(), h3_df["누적성장률"].to_numpy())
        short_sign = np.sign(h3_df["전월성장률"].to_numpy())
        long_sign = np.sign(h3_df["누적성장률"].to_numpy())
        nonzero = (short_sign != 0) & (long_sign != 0)
        mismatch_rate = float((short_sign[nonzero] != long_sign[nonzero]).mean() * 100) if nonzero.any() else np.nan

        if np.isfinite(h3_rho) and np.isfinite(mismatch_rate):
            if abs(h3_rho) < 0.30 and mismatch_rate >= 20:
                h3_status, h3_class = "탐색적 확인", "partial"
            elif abs(h3_rho) < 0.50 and mismatch_rate >= 10:
                h3_status, h3_class = "부분 확인", "partial"
            else:
                h3_status, h3_class = "기각", "rejected"
        else:
            h3_status, h3_class = "판정 보류", "neutral"

        h3_headline = "단기 증감과 장기 누적 변화의 관계는 뚜렷하지 않았습니다."
        if np.isfinite(h3_p) and h3_p >= 0.05:
            significance_text = f"{format_p_friendly(h3_p)} · 통계적으로 유의하지 않음"
        elif np.isfinite(h3_p):
            significance_text = f"{format_p_friendly(h3_p)} · 통계적으로 유의함"
        else:
            significance_text = "-"

        h3_interpretation = (
            f"전월에 증가한 지역이 항상 장기적으로도 증가한 것은 아니었습니다. "
            f"실제로 비교 가능한 관측치 중 {mismatch_rate:.1f}%에서 단기와 장기의 증감 방향이 달랐습니다."
        )
        h3_evidence = [
            ("상관 분석", f"Spearman ρ = {h3_rho:.3f}"),
            ("유의확률", significance_text),
            ("방향 불일치", f"{mismatch_rate:.1f}%"),
            ("비교 범위", f"{len(h3_df):,}개 지역×월 관측"),
        ]
        h3_stat_detail = f"Spearman ρ = {h3_rho:.3f} · 정확한 p-value = {format_p_value(h3_p)} · 방향 불일치율 = {mismatch_rate:.1f}%"
    else:
        h3_status, h3_class = "판정 보류", "neutral"
        h3_headline = "단기·장기 흐름을 비교할 관측치가 부족합니다."
        h3_interpretation = "전월 증감률과 누적 증감률을 함께 비교할 수 있는 관측치가 충분하지 않습니다."
        h3_evidence = [("검정 상태", "관측치 부족")]
        h3_stat_detail = "검정 미실행"

    return {
        "h1": {
            "kicker": "H1 · 지역 차이",
            "title": "지역별 사업자 증감률에는 차이가 있을 것이다.",
            "status": h1_status,
            "status_class": h1_class,
            "headline": h1_headline,
            "interpretation": h1_interpretation,
            "evidence": h1_evidence,
            "stat_detail": h1_stat_detail,
        },
        "h2": {
            "kicker": "H2 · 업종 차이",
            "title": "업종대분류별 사업자 증감 양상은 다를 것이다.",
            "status": h2_status,
            "status_class": h2_class,
            "headline": h2_headline,
            "interpretation": h2_interpretation,
            "evidence": h2_evidence,
            "stat_detail": h2_stat_detail,
        },
        "h3": {
            "kicker": "H3 · 단기/장기 흐름",
            "title": "단기 증감과 장기 누적 변화는 항상 같은 방향으로 움직이지 않을 것이다.",
            "status": h3_status,
            "status_class": h3_class,
            "headline": h3_headline,
            "interpretation": h3_interpretation,
            "evidence": h3_evidence,
            "stat_detail": h3_stat_detail,
        },
    }


hypothesis_results = make_hypothesis_results(viz)
print("[OK] 가설 검증 결과")
for key, result in hypothesis_results.items():
    print(f"     - {key.upper()}: {result['status']} / {result['stat_detail']}")


def hypothesis_card_html(result):
    evidence_html = "".join(
        '<div class="hypothesis-evidence-row">'
        f'<span>{label}</span><strong>{value}</strong>'
        '</div>'
        for label, value in result["evidence"]
    )

    return (
        '<div class="hypothesis-card">'
            '<div class="hypothesis-card-top">'
                '<div class="hypothesis-heading">'
                    f'<div class="hypothesis-kicker">{result["kicker"]}</div>'
                    f'<div class="hypothesis-title">{result["title"]}</div>'
                '</div>'
                f'<span class="hypothesis-status {result["status_class"]}">{result["status"]}</span>'
            '</div>'
            '<div class="hypothesis-result-block">'
                '<div class="hypothesis-block-label">한 줄 결론</div>'
                f'<div class="hypothesis-headline">{result["headline"]}</div>'
            '</div>'
            '<div class="hypothesis-interpretation">'
                '<div class="hypothesis-block-label">쉽게 말하면</div>'
                f'<p>{result["interpretation"]}</p>'
            '</div>'
            '<div class="hypothesis-evidence">'
                '<div class="hypothesis-block-label">검증 근거</div>'
                f'{evidence_html}'
            '</div>'
            '<details class="hypothesis-details">'
                '<summary>상세 통계 보기</summary>'
                f'<div>{result["stat_detail"]}</div>'
            '</details>'
        '</div>'
    )


hypothesis_html = "".join(hypothesis_card_html(hypothesis_results[key]) for key in ["h1", "h2", "h3"])


# ============================================================================== 
# 8. 부산 행정구역 SHP 탐색 / 부산만 필터링 - V2
# ==============================================================================
# 이전 버전은 지역명 컬럼을 먼저 찾은 뒤 부산 코드를 검사해서,
# 실제 SHP에 부산 코드가 있어도 지역명 컬럼명이 예상과 다르면 탈락할 수 있었습니다.
# V2는 코드/시도명/geometry로 부산을 먼저 찾고, 그 subset 안에서 지역명 컬럼을 찾습니다.

BUSAN_SGG_PREFIX = "26"

NAME_COLUMNS = [
    "SIGUNGU_NM", "SIG_KOR_NM", "SGG_NM", "SGG_NAME",
    "ADM_NM", "ADM_NAME", "구군명", "시군구명", "지역", "NAME",
]

CODE_COLUMNS = [
    "SIGUNGU_CD", "SIG_CD", "SGG_CD", "SGG_CODE",
    "SIGUNGU_CODE", "ADM_CD", "CODE",
]

PROVINCE_COLUMNS = [
    "SIDO_NM", "CTP_KOR_NM", "CTP_NM", "SIDO_NAME",
    "시도명", "시도", "광역시도", "SIDO",
]

# 부산을 충분히 포함하면서 울산 중구 등은 가급적 제외하는 범위
BUSAN_LON_RANGE = (128.65, 129.40)
BUSAN_LAT_RANGE = (34.80, 35.46)


def clean_region_name(series: pd.Series) -> pd.Series:
    """
    시군구 이름을 정리합니다.

    중요: "부산진구"의 "부산"은 지우면 안 됩니다.
    따라서 "부산광역시" 또는 "부산시"가 문자열 맨 앞의
    시도 접두어로 붙어 있을 때만 제거합니다.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^\s*부산광역시\s*", "", regex=True)
    s = s.str.replace(r"^\s*부산시\s*", "", regex=True)
    return s.str.strip()


def normalize_admin_code(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
    )


def _candidate_columns(gdf, preferred, keywords=()):
    result = []
    for col in preferred:
        if col in gdf.columns and col != gdf.geometry.name:
            result.append(col)

    for col in gdf.columns:
        if col == gdf.geometry.name or col in result:
            continue
        upper = str(col).upper()
        if any(keyword in upper for keyword in keywords):
            result.append(col)
    return result


def detect_region_name_column(gdf: gpd.GeoDataFrame, mask=None):
    sample = gdf if mask is None else gdf.loc[mask]

    candidates = _candidate_columns(
        gdf,
        NAME_COLUMNS,
        keywords=("NM", "NAME", "SIG", "SGG", "ADM", "구", "군", "지역"),
    )

    for col in gdf.columns:
        if col == gdf.geometry.name or col in candidates:
            continue
        dtype = gdf[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            candidates.append(col)

    best_col = None
    best_score = -1

    for column in candidates:
        try:
            names = clean_region_name(sample[column])
            score = len(set(names[names.isin(BUSAN_REGIONS)]))
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_col = column

    if best_col is None or best_score < 10:
        return None
    return best_col


def mask_from_admin_code(gdf: gpd.GeoDataFrame):
    candidates = _candidate_columns(gdf, CODE_COLUMNS, keywords=("CD", "CODE"))

    for column in candidates:
        try:
            codes = normalize_admin_code(gdf[column])
            mask = codes.str.startswith(BUSAN_SGG_PREFIX, na=False)
            count = int(mask.sum())
        except Exception:
            continue

        if count >= 10:
            return mask, f"{column} 코드 앞자리 26"

    return None, None


def mask_from_province_name(gdf: gpd.GeoDataFrame):
    candidates = _candidate_columns(gdf, PROVINCE_COLUMNS, keywords=("SIDO", "CTP", "PROV"))

    for column in candidates:
        try:
            mask = gdf[column].astype(str).str.contains("부산", na=False)
            count = int(mask.sum())
        except Exception:
            continue

        if count >= 10:
            return mask, f"{column}에 '부산' 포함"

    return None, None


def mask_from_geometry(gdf: gpd.GeoDataFrame):
    if gdf.crs is None or len(gdf) == 0:
        return None, None

    try:
        temp = gdf[[gdf.geometry.name]].copy().to_crs(epsg=4326)
        points = temp.geometry.representative_point()
        mask = (
            points.x.between(*BUSAN_LON_RANGE)
            & points.y.between(*BUSAN_LAT_RANGE)
        )
        count = int(mask.sum())
    except Exception:
        return None, None

    if count >= 10:
        return mask, "geometry 대표점이 부산 범위 안"

    return None, None


def make_busan_mask(gdf: gpd.GeoDataFrame, name_col=None):
    mask, basis = mask_from_admin_code(gdf)
    if mask is not None:
        return mask, basis

    mask, basis = mask_from_province_name(gdf)
    if mask is not None:
        return mask, basis

    if name_col is not None:
        try:
            mask = gdf[name_col].astype(str).str.contains("부산", na=False)
            if int(mask.sum()) >= 10:
                return mask, f"{name_col} 명칭에 '부산' 포함"
        except Exception:
            pass

    mask, basis = mask_from_geometry(gdf)
    if mask is not None:
        return mask, basis

    if name_col is None:
        name_col = detect_region_name_column(gdf)

    if name_col is not None:
        names = clean_region_name(gdf[name_col])
        matched = names.isin(BUSAN_REGIONS)
        unique_count = len(set(names[matched]))
        if unique_count >= 10 and int(matched.sum()) == len(gdf):
            return matched, "부산 전용 SHP로 판정"

    return None, None


def list_shapefile_candidates(roots):
    seen = set()
    result = []

    for root in roots:
        root = Path(root)
        if not root.exists():
            continue

        for shp in root.rglob("*.shp"):
            key = str(shp.resolve()).lower()
            if key not in seen:
                seen.add(key)
                result.append(shp)

    return result


def find_busan_shapefile(*roots) -> Path:
    candidates = list_shapefile_candidates(roots)

    if not candidates:
        raise FileNotFoundError(
            "\nSHP 파일을 찾지 못했습니다.\n"
            "CSV 폴더 또는 프로젝트 폴더 아래에 .shp/.shx/.dbf/.prj 파일을 함께 두거나,\n"
            "코드 상단 SHP_PATH에 정확한 .shp 경로를 지정해주세요."
        )

    print(f"[CHECK] 발견한 SHP 파일: {len(candidates)}개")

    best_path = None
    best_score = -1
    best_basis = None
    best_name_col = None
    diagnostics = []

    for shp in candidates:
        try:
            temp = gpd.read_file(shp)
        except Exception as e:
            diagnostics.append((shp, f"읽기 실패: {type(e).__name__}"))
            continue

        # 중요: 지역명 컬럼보다 부산 범위를 먼저 찾습니다.
        mask, basis = make_busan_mask(temp, name_col=None)

        if mask is None:
            preliminary_name_col = detect_region_name_column(temp)
            if preliminary_name_col is not None:
                mask, basis = make_busan_mask(temp, preliminary_name_col)

        if mask is None:
            diagnostics.append(
                (shp, f"rows={len(temp):,}, 부산 판별 실패, cols={list(temp.columns)}")
            )
            continue

        name_col = detect_region_name_column(temp, mask=mask)
        if name_col is None:
            diagnostics.append(
                (shp, f"rows={len(temp):,}, 부산후보={int(mask.sum())}개, 지역명 컬럼 탐색 실패, 기준={basis}, cols={list(temp.columns)}")
            )
            continue

        names = clean_region_name(temp.loc[mask, name_col])
        matched_names = names[names.isin(BUSAN_REGIONS)]
        score = len(set(matched_names))

        diagnostics.append(
            (shp, f"rows={len(temp):,}, 부산후보={int(mask.sum())}개, name_col={name_col}, 지역매칭={score}/16, 기준={basis}")
        )

        if score > best_score:
            best_score = score
            best_path = shp
            best_basis = basis
            best_name_col = name_col

    if best_path is None or best_score < 10:
        print("\n[진단] 검사한 SHP 목록")
        for shp, info in diagnostics:
            print(f" - {shp}\n   {info}")

        raise RuntimeError(
            "\nSHP 파일은 발견했지만 부산 구·군 경계 파일을 자동 판별하지 못했습니다.\n"
            "위 [진단] 목록에서 부산 행정구역 SHP 경로를 확인한 뒤\n"
            "코드 상단 SHP_PATH = Path(r'...') 로 직접 지정해도 됩니다."
        )

    print(f"[OK] 행정구역 SHP 자동 선택: {best_path}")
    print(f"[OK] 부산 판별 기준: {best_basis}")
    print(f"[OK] 지역명 컬럼: {best_name_col} / 매칭 {best_score}/16")
    return best_path


if SHP_PATH is None:
    # 데이터 폴더 + 그 상위 프로젝트 폴더까지 탐색
    SHP_PATH = find_busan_shapefile(DATA_DIR, DATA_DIR.parent)

if not Path(SHP_PATH).exists():
    raise FileNotFoundError(f"SHP 파일을 찾을 수 없습니다:\n{SHP_PATH}")


# ==============================================================================
# 8. 지도 데이터 전처리 - 전국 SHP에서 부산 16개 구·군만 정확히 선택
# ==============================================================================
# 이 SHP는 전국 시군구 경계이므로 "중구/남구/북구/동구" 같은 이름이 여러 시도에
# 중복됩니다. 단순 이름 필터링이나 넓은 bounding box만 사용하면 울산/대구 등의
# 동명 구가 섞일 수 있습니다.
#
# 따라서 각 부산 구·군의 대략적인 중심 좌표를 기준으로, 같은 이름 후보 중
# 부산 중심에 가장 가까운 geometry 1개를 선택합니다.

BUSAN_REGION_CENTERS = {
    "중구": (129.0324, 35.1060),
    "서구": (129.0167, 35.0979),
    "동구": (129.0454, 35.1293),
    "영도구": (129.0679, 35.0912),
    "부산진구": (129.0531, 35.1631),
    "동래구": (129.0858, 35.2048),
    "남구": (129.0840, 35.1366),
    "북구": (128.9900, 35.1972),
    "해운대구": (129.1636, 35.1631),
    "사하구": (128.9748, 35.1046),
    "금정구": (129.0912, 35.2428),
    "강서구": (128.9800, 35.2120),
    "연제구": (129.0820, 35.1762),
    "수영구": (129.1131, 35.1454),
    "사상구": (128.9910, 35.1526),
    "기장군": (129.2223, 35.2446),
}


def select_busan_districts_from_nationwide_shp(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        raise ValueError("SHP 파일에 CRS 정보가 없습니다. .prj 파일을 확인해주세요.")

    # 실제 파일에서 확인된 컬럼을 최우선 사용
    if "SIGUNGU_NM" in gdf.columns:
        name_col = "SIGUNGU_NM"
    else:
        name_col = detect_region_name_column(gdf)

    if name_col is None:
        raise ValueError(
            "시군구 이름 컬럼을 찾지 못했습니다.\n"
            f"현재 SHP 컬럼: {gdf.columns.tolist()}"
        )

    wgs = gdf.to_crs(epsg=4326).copy()
    wgs["_지역명"] = clean_region_name(wgs[name_col])

    selected = []
    diagnostics = []

    for region in BUSAN_REGIONS:
        candidates = wgs[wgs["_지역명"] == region].copy()

        if candidates.empty:
            # 공백/특수문자 등 예상치 못한 표기를 확인하기 위한 진단
            similar = sorted(
                set(
                    wgs.loc[
                        wgs["_지역명"].astype(str).str.contains(
                            region.replace("구", "").replace("군", ""),
                            na=False,
                            regex=False,
                        ),
                        "_지역명",
                    ].astype(str)
                )
            )[:20]
            raise RuntimeError(
                f"SHP에서 '{region}' 후보를 찾지 못했습니다.\n"
                f"지역명 컬럼: {name_col}\n"
                f"비슷한 표기: {similar}"
            )

        target_lon, target_lat = BUSAN_REGION_CENTERS[region]
        reps = candidates.geometry.representative_point()
        dist2 = (reps.x - target_lon) ** 2 + (reps.y - target_lat) ** 2
        chosen_idx = dist2.idxmin()
        chosen = candidates.loc[[chosen_idx]].copy()
        chosen["지역"] = region
        selected.append(chosen)

        chosen_point = chosen.geometry.representative_point().iloc[0]
        diagnostics.append(
            (region, len(candidates), float(chosen_point.x), float(chosen_point.y))
        )

    gu = gpd.GeoDataFrame(
        pd.concat(selected, ignore_index=True),
        geometry=gdf.geometry.name,
        crs="EPSG:4326",
    )

    # 정확히 16개인지 검증
    if len(gu) != 16 or set(gu["지역"]) != set(BUSAN_REGIONS):
        raise RuntimeError(
            "부산 16개 구·군 선택에 실패했습니다.\n"
            f"현재 지역: {sorted(gu['지역'].tolist())}"
        )

    print(f"[OK] 지역명 컬럼: {name_col}")
    print("[OK] 전국 SHP에서 부산 16개 구·군을 위치 기반으로 선택했습니다.")
    for region, count, lon, lat in diagnostics:
        extra = f" (동명 후보 {count}개 중 선택)" if count > 1 else ""
        print(f"     - {region}: lon={lon:.4f}, lat={lat:.4f}{extra}")

    # GeoPandas 버전에 따라 활성 geometry 컬럼명이 이미 "geometry"인 상태에서
    # rename_geometry("geometry")를 다시 호출하면
    # ValueError: Column named geometry already exists 가 발생할 수 있습니다.
    # 따라서 geometry 컬럼을 명시적으로 정리한 새 GeoDataFrame을 반환합니다.
    geom_col = gu.geometry.name

    if geom_col == "geometry":
        result = gu[["지역", "geometry"]].copy()
        return gpd.GeoDataFrame(
            result,
            geometry="geometry",
            crs=gu.crs,
        )

    result = gu[["지역", geom_col]].copy()
    result = result.rename(columns={geom_col: "geometry"})

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=gu.crs,
    )


# 전국 시군구 SHP 읽기
gu_all = gpd.read_file(SHP_PATH)
print(f"[OK] SHP 불러오기 완료: {len(gu_all):,}개 geometry")
print(f"[OK] SHP 경로: {SHP_PATH}")

gu = select_busan_districts_from_nationwide_shp(gu_all)

# 지도 경량화: 미터 좌표계에서 단순화 후 다시 WGS84로
try:
    gu_simple = gu.to_crs(epsg=5186).copy()
    gu_simple["geometry"] = gu_simple["geometry"].simplify(
        tolerance=80,
        preserve_topology=True,
    )
    gu_simple = gu_simple.to_crs(epsg=4326)
except Exception as e:
    print("[WARN] EPSG:5186 단순화 실패. WGS84에서 소폭 단순화합니다:", e)
    gu_simple = gu.to_crs(epsg=4326).copy()
    gu_simple["geometry"] = gu_simple["geometry"].simplify(
        tolerance=0.0007,
        preserve_topology=True,
    )

gu_simple = gu_simple[["지역", "geometry"]].copy()

min_lon, min_lat, max_lon, max_lat = gu_simple.total_bounds
print(
    f"[OK] 부산 지도 경계: "
    f"lon {min_lon:.4f} ~ {max_lon:.4f}, "
    f"lat {min_lat:.4f} ~ {max_lat:.4f}"
)

# 부산 전체 범위가 대략 이 안에 들어와야 정상
if (
    min_lon < 128.65 or max_lon > 129.40
    or min_lat < 34.80 or max_lat > 35.48
):
    raise RuntimeError(
        "부산 외 지역 geometry가 섞였을 가능성이 있습니다.\n"
        f"현재 bounds: {gu_simple.total_bounds.tolist()}"
    )

busan_geojson = json.loads(gu_simple.to_json())
print(f"[OK] 최종 부산 지도 지역 수: {gu_simple['지역'].nunique()}개")


# ==============================================================
# 9. HTML용 데이터 생성
# ==============================================================

export_columns = [
    "연월표시", "연도", "월", "지역", "업종",
    "사업자수", "전월사업자수", "전년동월사업자수", "기준월사업자수",
    "전월성장률", "전년동월성장률", "누적성장률", "성장지수",
]

export_df = viz[export_columns].copy()

# DataFrame.to_json()을 사용하면 NaN이 JSON의 null로 안전하게 변환됨
records_json = export_df.to_json(
    orient="records",
    force_ascii=False,
)


# 연도별 대표값: 각 연도의 마지막 관측월
annual = (
    viz.sort_values("기준일")
    .groupby(["연도", "지역", "업종"], group_keys=False)
    .tail(1)
    .copy()
)

annual_export = annual[
    [
        "연도", "월", "연월표시", "지역", "업종", "사업자수",
        "전월성장률", "전년동월성장률", "누적성장률", "성장지수",
    ]
].copy()

annual_json = annual_export.to_json(
    orient="records",
    force_ascii=False,
)

geojson_json = json.dumps(
    busan_geojson,
    ensure_ascii=False,
    separators=(",", ":"),
)

regions_json = json.dumps(BUSAN_REGIONS, ensure_ascii=False)
industries_json = json.dumps(INDUSTRY_ORDER, ensure_ascii=False)


# ==============================================================
# 10. Plotly.js 로드
# ==============================================================

plotly_js = pyo.get_plotlyjs()

if len(plotly_js) < 100_000:
    raise RuntimeError(
        "Plotly.js 로딩에 실패한 것으로 보입니다. "
        "plotly 패키지를 업데이트해주세요: pip install -U plotly"
    )

print(f"[OK] Plotly.js 로드: {len(plotly_js):,} characters")


# ==============================================================
# 11. HTML 템플릿
# 중요:
# - f-string을 쓰지 않습니다.
# - Plotly.js placeholder는 딱 1번만 존재합니다.
# - DATA / ANNUAL / BUSAN_GEOJSON / REGIONS / INDUSTRIES도 딱 1번 선언합니다.
# ==============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>부산 사업자 증감 분석 Dashboard</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #f4f6f9;
    font-family: Arial, "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
    color: #172033;
}

.dashboard {
    width: 96%;
    max-width: 1650px;
    margin: 24px auto 70px auto;
}

.header {
    margin-bottom: 18px;
}

.header h1 {
    margin: 0 0 8px 0;
    font-size: 31px;
}

.header p {
    margin: 0;
    color: #687386;
    font-size: 14px;
}

.panel {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 3px 16px rgba(0, 0, 0, 0.055);
}

.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
}

.control-box {
    min-width: 165px;
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.control-box label {
    font-size: 12px;
    font-weight: 700;
    color: #697386;
}

select {
    height: 42px;
    padding: 0 12px;
    border: 1px solid #d6dbe3;
    border-radius: 8px;
    background: #ffffff;
    font-size: 14px;
}

.cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-top: 20px;
}

.card {
    border: 1px solid #e3e8ef;
    border-radius: 12px;
    padding: 18px;
    min-height: 112px;
}

.card-title {
    color: #7b8495;
    font-size: 12px;
    margin-bottom: 10px;
}

.card-value {
    font-size: 25px;
    font-weight: 700;
}

.card-sub {
    margin-top: 6px;
    font-size: 12px;
    color: #8b94a5;
}

.section-title {
    margin: 0 0 5px 0;
    font-size: 20px;
}

.section-description {
    margin: 0 0 15px 0;
    font-size: 13px;
    color: #7d8798;
}

.view-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin: 14px 0 8px 0;
    padding: 10px;
    border: 1px solid #e3e8ef;
    border-radius: 12px;
    background: #f8fafc;
}

.view-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.view-btn {
    height: 38px;
    padding: 0 16px;
    border: 1px solid #cfd7e3;
    border-radius: 9px;
    background: #ffffff;
    color: #344054;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
}

.view-btn:hover {
    border-color: #64748b;
    background: #f1f5f9;
}

.view-btn.active {
    border-color: #172033;
    background: #172033;
    color: #ffffff;
    box-shadow: 0 2px 7px rgba(23, 32, 51, 0.18);
}

.compare-region-box {
    display: none;
    align-items: center;
    gap: 8px;
}

.compare-region-box.visible {
    display: flex;
}

.compare-region-box label {
    font-size: 12px;
    font-weight: 700;
    color: #697386;
    white-space: nowrap;
}

.compare-region-box select {
    min-width: 140px;
    height: 38px;
}

.industry-quickbar {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin: 10px 0 8px 0;
    padding: 11px 12px;
    border: 1px solid #dbe3ec;
    border-radius: 12px;
    background: #ffffff;
}

.industry-quick-label {
    margin-right: 4px;
    color: #667085;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}

.industry-quick-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}

.industry-quick-btn {
    min-height: 34px;
    padding: 0 13px;
    border: 1px solid #cbd5e1;
    border-radius: 999px;
    background: #f8fafc;
    color: #344054;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}

.industry-quick-btn:hover {
    border-color: #64748b;
    background: #eef2f7;
}

.industry-quick-btn.active {
    border-color: #172033;
    background: #172033;
    color: #ffffff;
}

.industry-quick-btn[data-industry="전체"] {
    padding-left: 16px;
    padding-right: 16px;
}

.view-note {
    margin: 4px 0 4px 2px;
    color: #667085;
    font-size: 12px;
}

.timeline-bar {
    display: none;
    grid-template-columns: auto auto auto minmax(220px, 1fr) auto;
    align-items: center;
    gap: 10px;
    margin: 12px 0 8px 0;
    padding: 12px 14px;
    border: 1px solid #dbe3ec;
    border-radius: 12px;
    background: #f8fafc;
}

.timeline-bar.visible {
    display: grid;
}

.timeline-btn {
    height: 36px;
    padding: 0 13px;
    border: 1px solid #cbd5e1;
    border-radius: 9px;
    background: #ffffff;
    color: #243047;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
}

.timeline-btn:hover {
    border-color: #64748b;
    background: #eef2f7;
}

.timeline-btn.playing {
    background: #172033;
    border-color: #172033;
    color: #ffffff;
}

.timeline-range {
    width: 100%;
    accent-color: #172033;
    cursor: pointer;
}

.timeline-period {
    min-width: 92px;
    padding: 7px 10px;
    border-radius: 8px;
    background: #ffffff;
    border: 1px solid #e1e7ef;
    color: #344054;
    font-size: 12px;
    font-weight: 700;
    text-align: center;
    white-space: nowrap;
}

.integrated-panel {
    padding-bottom: 16px;
}

.integrated-panel .map-section-divider {
    height: 1px;
    background: #edf1f5;
    margin: 22px 0 20px 0;
}

.chart {
    width: 100%;
}

#map { height: 680px; }
#heatmap { height: 610px; }
#lineChart { height: 570px; }

.region-control {
    max-width: 260px;
    margin-bottom: 12px;
}

.status {
    display: none;
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 8px;
    background: #fff3f3;
    color: #b42318;
    font-size: 13px;
    white-space: pre-wrap;
}


.method-note { margin: 12px 0 0 0; padding: 12px 14px; border: 1px solid #d8e1ec; border-radius: 10px; background: #f8fafc; color: #475467; font-size: 12.5px; line-height: 1.6; }
.method-note strong { color: #172033; }

.hypothesis-intro {
    margin: 0 0 14px 0;
    padding: 14px 16px;
    border-radius: 11px;
    background: #f8fafc;
    color: #475467;
    font-size: 13px;
    line-height: 1.65;
}

.hypothesis-guide {
    display: flex;
    flex-wrap: wrap;
    gap: 9px 18px;
    align-items: center;
    margin: 0 0 18px 0;
    padding: 11px 14px;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    background: #ffffff;
    color: #667085;
    font-size: 12px;
}
.hypothesis-guide strong { color: #172033; }
.hypothesis-guide .guide-good { color: #166534; font-weight: 700; }
.hypothesis-guide .guide-neutral { color: #92400e; font-weight: 700; }

.hypothesis-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
}

.hypothesis-card {
    display: flex;
    flex-direction: column;
    border: 1px solid #dfe5ed;
    border-radius: 14px;
    padding: 18px;
    background: #ffffff;
    min-height: 360px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.035);
}

.hypothesis-card-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    padding-bottom: 14px;
    border-bottom: 1px solid #edf1f5;
}
.hypothesis-heading { min-width: 0; }
.hypothesis-kicker {
    margin-bottom: 5px;
    color: #667085;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.02em;
}
.hypothesis-title {
    font-size: 14px;
    font-weight: 800;
    line-height: 1.5;
    color: #172033;
}
.hypothesis-status {
    flex: 0 0 auto;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    white-space: nowrap;
}
.hypothesis-status.supported { background: #dcfce7; color: #166534; }
.hypothesis-status.partial { background: #fef3c7; color: #92400e; }
.hypothesis-status.rejected { background: #fee2e2; color: #991b1b; }
.hypothesis-status.neutral { background: #e2e8f0; color: #475569; }

.hypothesis-result-block {
    margin-top: 15px;
    padding: 13px 14px;
    border-radius: 10px;
    background: #f8fafc;
}
.hypothesis-block-label {
    margin-bottom: 6px;
    color: #7b8495;
    font-size: 10.5px;
    font-weight: 800;
    letter-spacing: 0.02em;
}
.hypothesis-headline {
    color: #172033;
    font-size: 14px;
    font-weight: 800;
    line-height: 1.55;
}

.hypothesis-interpretation {
    margin-top: 14px;
}
.hypothesis-interpretation p {
    margin: 0;
    color: #566176;
    font-size: 12.5px;
    line-height: 1.7;
}

.hypothesis-evidence {
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid #edf1f5;
}
.hypothesis-evidence-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 14px;
    padding: 5px 0;
    font-size: 12px;
    line-height: 1.5;
}
.hypothesis-evidence-row span { color: #7b8495; }
.hypothesis-evidence-row strong {
    color: #344054;
    text-align: right;
    font-weight: 750;
}

.hypothesis-details {
    margin-top: auto;
    padding-top: 14px;
    color: #667085;
    font-size: 11.5px;
}
.hypothesis-details summary {
    cursor: pointer;
    color: #667085;
    font-weight: 700;
    user-select: none;
}
.hypothesis-details[open] summary { margin-bottom: 7px; }
.hypothesis-details div {
    padding: 8px 10px;
    border-radius: 8px;
    background: #f8fafc;
    line-height: 1.55;
}

.hypothesis-footnote {
    margin-top: 15px;
    padding: 11px 13px;
    border-radius: 9px;
    background: #fffbeb;
    color: #7c5b17;
    font-size: 11.5px;
    line-height: 1.65;
}
@media (max-width: 900px) {
    .cards { grid-template-columns: repeat(2, 1fr); }
    .hypothesis-grid { grid-template-columns: 1fr; }
}

@media (max-width: 550px) {
    .cards { grid-template-columns: 1fr; }
    .control-box { width: 100%; }
    .view-toolbar { align-items: stretch; }
    .view-tabs { width: 100%; }
    .view-btn { flex: 1; padding: 0 10px; }
    .compare-region-box { width: 100%; }
    .compare-region-box select { flex: 1; }
    .industry-quickbar { align-items: flex-start; }
    .industry-quick-label { width: 100%; }
    .industry-quick-buttons { width: 100%; }
    .industry-quick-btn { flex: 1 1 auto; }
    .timeline-bar {
        grid-template-columns: 1fr 1fr;
    }
    .timeline-range {
        grid-column: 1 / -1;
        order: 5;
    }
    .timeline-period {
        min-width: 0;
    }
}
</style>

<!-- Plotly.js: 정확히 1번만 삽입 -->
<script>
__PLOTLY_JS__
</script>
</head>

<body>
<div class="dashboard">

    <div class="header">
        <h1>부산 지역별 사업자 증감 분석</h1>
        <p>2022년 1월 ~ 2026년 6월 · 부산 16개 구·군 · 업종별 사업자 수 증감 추이</p>
        <div class="method-note">
            <strong>분석 기준</strong> · 현재 데이터는 신규사업자와 폐업사업자 건수를 각각 분리한 자료가 아닙니다.
            따라서 본 대시보드는 <strong>월별 사업자 수의 순증감</strong>, 지역별 차이, 업종별 변화 패턴을 분석합니다.
        </div>
    </div>

    <div class="panel integrated-panel">
        <div class="controls">
            <div class="control-box">
                <label for="industrySelect">업종</label>
                <select id="industrySelect"></select>
            </div>

            <div class="control-box">
                <label for="metricSelect">증감 지표</label>
                <select id="metricSelect">
                    <option value="전년동월성장률">전년동월 증감률 (YoY)</option>
                    <option value="전월성장률">전월 증감률 (MoM)</option>
                    <option value="누적성장률">2022-01 대비 누적 증감률</option>
                </select>
            </div>

            <div class="control-box">
                <label for="yearSelect">연도</label>
                <select id="yearSelect"></select>
            </div>

            <div class="control-box">
                <label for="monthSelect">월</label>
                <select id="monthSelect"></select>
            </div>
        </div>

        <div class="cards">
            <div class="card">
                <div class="card-title">부산 사업자 수</div>
                <div class="card-value" id="totalBusiness">-</div>
                <div class="card-sub" id="selectedPeriod">-</div>
            </div>

            <div class="card">
                <div class="card-title">부산 전체 증감률</div>
                <div class="card-value" id="busanGrowth">-</div>
                <div class="card-sub">선택 지표 기준</div>
            </div>

            <div class="card">
                <div class="card-title">증감률 최고 지역</div>
                <div class="card-value" id="bestRegion">-</div>
                <div class="card-sub" id="bestGrowth">-</div>
            </div>

            <div class="card">
                <div class="card-title">증감률 최저 지역</div>
                <div class="card-value" id="worstRegion">-</div>
                <div class="card-sub" id="worstGrowth">-</div>
            </div>
        </div>

        <div id="status" class="status"></div>

        <div class="map-section-divider"></div>

        <h2 class="section-title">부산 지역별 사업자 증감 지도</h2>
        <p class="section-description">
            선택창과 지도를 한 화면에 배치했습니다. 아래 업종 버튼으로 전체·6개 업종대분류를 즉시 전환하고,
            월 이동 버튼·슬라이더·자동 재생으로 월별 지역 증감 변화를 바로 넘겨볼 수 있습니다.
        </p>

        <div class="view-toolbar">
            <div class="view-tabs">
                <button type="button" class="view-btn active" id="mapViewBtn">지도 보기</button>
                <button type="button" class="view-btn" id="regionCompareBtn">지역별 비교</button>
                <button type="button" class="view-btn" id="monthCompareBtn">월별 비교</button>
            </div>

            <div class="compare-region-box" id="compareRegionBox">
                <label for="compareRegionSelect">비교 지역</label>
                <select id="compareRegionSelect"></select>
            </div>
        </div>

        <div class="industry-quickbar" id="industryQuickBar">
            <span class="industry-quick-label">업종 바로 보기</span>
            <div class="industry-quick-buttons" id="industryQuickButtons"></div>
        </div>

        <div id="timelineBar" class="timeline-bar visible">
            <button type="button" class="timeline-btn" id="prevMonthBtn">◀ 이전 월</button>
            <button type="button" class="timeline-btn" id="playMonthBtn">▶ 월별 재생</button>
            <button type="button" class="timeline-btn" id="nextMonthBtn">다음 월 ▶</button>
            <input type="range" class="timeline-range" id="monthTimeline" min="0" max="0" value="0" step="1">
            <div class="timeline-period" id="timelinePeriodLabel">-</div>
        </div>

        <div id="mapViewNote" class="view-note">
            이전 월·다음 월·슬라이더·월별 재생으로 월별 지도 변화를 바로 확인할 수 있습니다.
        </div>

        <div id="map" class="chart"></div>
    </div>

    <div class="panel">
        <h2 class="section-title">지역 × 연도 증감 Heatmap</h2>
        <p class="section-description">
            각 연도의 마지막 관측월을 기준으로 부산 16개 구·군의 사업자 증감 흐름을 비교합니다.
        </p>
        <div id="heatmap" class="chart"></div>
    </div>

    <div class="panel">
        <h2 class="section-title">지역별 장기 사업자 수 추이</h2>
        <p class="section-description">
            선택한 지역에서 전체 사업자와 6개 업종대분류의 사업자 수가 기준월 대비 어떻게 변했는지 비교합니다.
        </p>

        <div class="region-control">
            <div class="control-box">
                <label for="regionSelect">지역 선택</label>
                <select id="regionSelect"></select>
            </div>
        </div>

        <div id="lineChart" class="chart"></div>
    </div>

    <div class="panel">
        <h2 class="section-title">가설 검증 결과</h2>
        <div class="hypothesis-intro">
            <strong>먼저 결론과 쉬운 해석을 확인하고, 필요할 때만 상세 통계를 펼쳐보세요.</strong><br>
            현재 보유한 사업자 수 데이터에서 확인 가능한 지역 차이·업종 차이·단기/장기 흐름을 검증했습니다.
        </div>
        <div class="hypothesis-guide">
            <strong>p-value 읽는 법</strong>
            <span class="guide-good">p &lt; 0.05 → 우연으로 보기 어려운 차이</span>
            <span class="guide-neutral">p ≥ 0.05 → 통계적으로 뚜렷하다고 보기 어려움</span>
        </div>
        <div class="hypothesis-grid">
__HYPOTHESIS_HTML__
        </div>
        <div class="hypothesis-footnote">
            <strong>해석 주의</strong> · H3의 p-value가 0.05보다 크다는 것은 “관계가 없다”가 확정됐다는 뜻이 아닙니다.
            현재 데이터에서는 뚜렷한 상관이 확인되지 않았고, 동시에 실제 증감 방향이 다른 사례가 있어 탐색적으로 해석합니다.
            신규·폐업의 개별 원인까지 설명하려면 별도의 신규사업자·폐업사업자 집계 데이터가 필요합니다.
        </div>
    </div>
</div>

<script>
// =============================================================
// 데이터: 여기에서만 1회 선언
// =============================================================
const DATA = __RECORDS_JSON__;
const ANNUAL = __ANNUAL_JSON__;
const BUSAN_GEOJSON = __GEOJSON_JSON__;
const REGIONS = __REGIONS_JSON__;
const INDUSTRIES = __INDUSTRIES_JSON__;

// =============================================================
// DOM
// =============================================================
const industrySelect = document.getElementById("industrySelect");
const metricSelect = document.getElementById("metricSelect");
const yearSelect = document.getElementById("yearSelect");
const monthSelect = document.getElementById("monthSelect");
const regionSelect = document.getElementById("regionSelect");
const statusBox = document.getElementById("status");

const mapViewBtn = document.getElementById("mapViewBtn");
const regionCompareBtn = document.getElementById("regionCompareBtn");
const monthCompareBtn = document.getElementById("monthCompareBtn");
const compareRegionBox = document.getElementById("compareRegionBox");
const compareRegionSelect = document.getElementById("compareRegionSelect");
const mapViewNote = document.getElementById("mapViewNote");
const timelineBar = document.getElementById("timelineBar");
const prevMonthBtn = document.getElementById("prevMonthBtn");
const playMonthBtn = document.getElementById("playMonthBtn");
const nextMonthBtn = document.getElementById("nextMonthBtn");
const monthTimeline = document.getElementById("monthTimeline");
const timelinePeriodLabel = document.getElementById("timelinePeriodLabel");
const industryQuickBar = document.getElementById("industryQuickBar");
const industryQuickButtons = document.getElementById("industryQuickButtons");

const ALL_PERIODS = [...new Set(
    DATA.map(d => String(d["연월표시"])).filter(Boolean)
)].sort();

let currentMapView = "map";
let playbackTimer = null;

function showStatus(message) {
    statusBox.style.display = "block";
    statusBox.textContent = message;
}

function clearStatus() {
    statusBox.style.display = "none";
    statusBox.textContent = "";
}

function isFiniteNumber(value) {
    return value !== null
        && value !== undefined
        && value !== ""
        && Number.isFinite(Number(value));
}

function numberFormat(value) {
    if (!isFiniteNumber(value)) return "-";
    return Number(value).toLocaleString("ko-KR");
}

function percentFormat(value) {
    if (!isFiniteNumber(value)) return "-";
    const n = Number(value);
    const sign = n > 0 ? "+" : "";
    return sign + n.toFixed(2) + "%";
}

function metricLabel(metric) {
    const labels = {
        "전년동월성장률": "전년동월 증감률 (YoY)",
        "전월성장률": "전월 증감률 (MoM)",
        "누적성장률": "기준월 대비 누적 증감률",
    };
    return labels[metric] || metric;
}

// =============================================================
// 업종 빠른 선택 버튼
// =============================================================
function renderIndustryQuickButtons() {
    industryQuickButtons.innerHTML = "";

    INDUSTRIES.forEach(industry => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "industry-quick-btn";
        button.dataset.industry = industry;
        button.textContent = industry;
        button.title = `${industry} 지도로 바로 전환`;

        button.addEventListener("click", () => {
            if (industrySelect.value === industry) {
                syncIndustryQuickButtons();
                return;
            }

            industrySelect.value = industry;
            syncIndustryQuickButtons();
            updateMainCharts();
        });

        industryQuickButtons.appendChild(button);
    });

    syncIndustryQuickButtons();
}

function syncIndustryQuickButtons() {
    const selected = industrySelect.value;

    industryQuickButtons
        .querySelectorAll(".industry-quick-btn")
        .forEach(button => {
            button.classList.toggle(
                "active",
                button.dataset.industry === selected
            );
        });
}

// =============================================================
// SELECT 초기화
// =============================================================
function initializeSelects() {
    industrySelect.innerHTML = "";
    INDUSTRIES.forEach(industry => {
        const option = document.createElement("option");
        option.value = industry;
        option.textContent = industry;
        industrySelect.appendChild(option);
    });

    regionSelect.innerHTML = "";
    compareRegionSelect.innerHTML = "";

    REGIONS.forEach(region => {
        const option = document.createElement("option");
        option.value = region;
        option.textContent = region;
        regionSelect.appendChild(option);

        const compareOption = document.createElement("option");
        compareOption.value = region;
        compareOption.textContent = region;
        compareRegionSelect.appendChild(compareOption);
    });

    const years = [...new Set(
        DATA.map(d => Number(d["연도"]))
            .filter(v => Number.isFinite(v))
    )].sort((a, b) => a - b);

    yearSelect.innerHTML = "";
    years.forEach(year => {
        const option = document.createElement("option");
        option.value = String(year);
        option.textContent = String(year);
        yearSelect.appendChild(option);
    });

    industrySelect.value = INDUSTRIES.includes("전체") ? "전체" : INDUSTRIES[0];
    metricSelect.value = "전년동월성장률";
    regionSelect.value = REGIONS.includes("해운대구") ? "해운대구" : REGIONS[0];
    compareRegionSelect.value = regionSelect.value;

    if (years.length > 0) {
        yearSelect.value = String(Math.max(...years));
    }

    updateMonths();
    renderIndustryQuickButtons();
}

function updateMonths() {
    const year = Number(yearSelect.value);

    const months = [...new Set(
        DATA
            .filter(d => Number(d["연도"]) === year)
            .map(d => Number(d["월"]))
            .filter(v => Number.isFinite(v))
    )].sort((a, b) => a - b);

    const oldMonth = Number(monthSelect.value);
    monthSelect.innerHTML = "";

    months.forEach(month => {
        const option = document.createElement("option");
        option.value = String(month);
        option.textContent = month + "월";
        monthSelect.appendChild(option);
    });

    if (months.length === 0) return;

    if (months.includes(oldMonth)) {
        monthSelect.value = String(oldMonth);
    } else {
        monthSelect.value = String(Math.max(...months));
    }
}


function currentPeriodString() {
    if (!yearSelect.value || !monthSelect.value) return "";
    return `${yearSelect.value}-${String(monthSelect.value).padStart(2, "0")}`;
}

function syncTimelineFromSelectors() {
    const current = currentPeriodString();
    const index = ALL_PERIODS.indexOf(current);

    monthTimeline.min = "0";
    monthTimeline.max = String(Math.max(ALL_PERIODS.length - 1, 0));

    if (index >= 0) {
        monthTimeline.value = String(index);
        timelinePeriodLabel.textContent = current.replace("-", "년 ") + "월";
    } else if (ALL_PERIODS.length > 0) {
        monthTimeline.value = String(ALL_PERIODS.length - 1);
        timelinePeriodLabel.textContent = ALL_PERIODS[ALL_PERIODS.length - 1].replace("-", "년 ") + "월";
    }
}

function applyPeriodByIndex(index, redraw = true) {
    if (ALL_PERIODS.length === 0) return;

    const safeIndex = Math.max(0, Math.min(Number(index), ALL_PERIODS.length - 1));
    const period = ALL_PERIODS[safeIndex];
    const [year, month] = period.split("-").map(Number);

    yearSelect.value = String(year);
    updateMonths();
    monthSelect.value = String(month);

    monthTimeline.value = String(safeIndex);
    timelinePeriodLabel.textContent = `${year}년 ${month}월`;

    if (redraw) {
        drawMap();
    }
}

function movePeriod(step) {
    const current = Number(monthTimeline.value || 0);
    const next = Math.max(0, Math.min(current + step, ALL_PERIODS.length - 1));
    applyPeriodByIndex(next, true);
}

function stopPlayback() {
    if (playbackTimer !== null) {
        clearInterval(playbackTimer);
        playbackTimer = null;
    }
    playMonthBtn.classList.remove("playing");
    playMonthBtn.textContent = "▶ 월별 재생";
}

function togglePlayback() {
    if (playbackTimer !== null) {
        stopPlayback();
        return;
    }

    playMonthBtn.classList.add("playing");
    playMonthBtn.textContent = "⏸ 정지";

    playbackTimer = setInterval(() => {
        let current = Number(monthTimeline.value || 0);
        let next = current + 1;

        if (next >= ALL_PERIODS.length) {
            next = 0;
        }

        applyPeriodByIndex(next, true);
    }, 900);
}

// =============================================================
// 색상 범위
// =============================================================
// 전체 기간의 극단치가 현재 지도의 색을 희석시키지 않도록
// 지도/지역비교에서는 '선택한 한 달의 16개 구·군'을 기준으로 범위를 잡습니다.
function getPeriodColorLimit(rows, metric) {
    const values = rows
        .filter(d => isFiniteNumber(d[metric]))
        .map(d => Math.abs(Number(d[metric])));

    const maxAbs = values.length > 0 ? Math.max(...values) : 0;

    const minimum = metric === "전월성장률"
        ? 0.5
        : metric === "전년동월성장률"
            ? 1.0
            : 2.0;

    return Math.max(maxAbs, minimum);
}

// Heatmap처럼 여러 연도를 한 번에 비교할 때만 전체 데이터 기반 범위를 사용합니다.
function getGlobalColorLimit(industry, metric) {
    const values = DATA
        .filter(d => d["업종"] === industry && isFiniteNumber(d[metric]))
        .map(d => Math.abs(Number(d[metric])))
        .sort((a, b) => a - b);

    if (values.length === 0) return 1;

    const index = Math.min(
        Math.floor(values.length * 0.90),
        values.length - 1
    );

    return Math.max(values[index], 1);
}

// 이전보다 채도와 대비를 높인 색상표
const GROWTH_COLORSCALE = [
    [0.00, "#991B1B"],
    [0.16, "#DC2626"],
    [0.34, "#F97316"],
    [0.47, "#FED7AA"],
    [0.50, "#F8FAFC"],
    [0.53, "#BBF7D0"],
    [0.66, "#4ADE80"],
    [0.84, "#16A34A"],
    [1.00, "#047857"],
];

// =============================================================
// KPI
// =============================================================
function updateCards(rows, metric) {
    if (rows.length === 0) {
        document.getElementById("totalBusiness").textContent = "-";
        document.getElementById("busanGrowth").textContent = "-";
        document.getElementById("bestRegion").textContent = "-";
        document.getElementById("bestGrowth").textContent = "-";
        document.getElementById("worstRegion").textContent = "-";
        document.getElementById("worstGrowth").textContent = "-";
        return;
    }

    const totalBusiness = rows.reduce(
        (sum, row) => sum + Number(row["사업자수"] || 0),
        0
    );

    let aggregateGrowth = null;

    if (metric === "전년동월성장률") {
        const previous = rows.reduce(
            (sum, row) => sum + Number(row["전년동월사업자수"] || 0),
            0
        );
        if (previous > 0) {
            aggregateGrowth = (totalBusiness / previous - 1) * 100;
        }
    } else if (metric === "전월성장률") {
        const previous = rows.reduce(
            (sum, row) => sum + Number(row["전월사업자수"] || 0),
            0
        );
        if (previous > 0) {
            aggregateGrowth = (totalBusiness / previous - 1) * 100;
        }
    } else {
        const baseline = rows.reduce(
            (sum, row) => sum + Number(row["기준월사업자수"] || 0),
            0
        );
        if (baseline > 0) {
            aggregateGrowth = (totalBusiness / baseline - 1) * 100;
        }
    }

    const validRows = rows.filter(row => isFiniteNumber(row[metric]));

    let best = null;
    let worst = null;

    if (validRows.length > 0) {
        best = [...validRows].sort(
            (a, b) => Number(b[metric]) - Number(a[metric])
        )[0];

        worst = [...validRows].sort(
            (a, b) => Number(a[metric]) - Number(b[metric])
        )[0];
    }

    document.getElementById("totalBusiness").textContent = numberFormat(totalBusiness);
    document.getElementById("busanGrowth").textContent = percentFormat(aggregateGrowth);
    document.getElementById("selectedPeriod").textContent =
        yearSelect.value + "년 " + monthSelect.value + "월 · " + industrySelect.value;

    document.getElementById("bestRegion").textContent = best ? best["지역"] : "-";
    document.getElementById("bestGrowth").textContent = best ? percentFormat(best[metric]) : "-";
    document.getElementById("worstRegion").textContent = worst ? worst["지역"] : "-";
    document.getElementById("worstGrowth").textContent = worst ? percentFormat(worst[metric]) : "-";
}

// =============================================================
// 지도
// 외부 Mapbox/Carto 타일을 사용하지 않는 완전 내장 choropleth
// =============================================================
function drawChoroplethMap() {
    const industry = industrySelect.value;
    const metric = metricSelect.value;
    const year = Number(yearSelect.value);
    const month = Number(monthSelect.value);

    const rows = DATA.filter(d =>
        d["업종"] === industry
        && Number(d["연도"]) === year
        && Number(d["월"]) === month
    );

    updateCards(rows, metric);

    if (rows.length === 0) {
        Plotly.purge("map");
        showStatus("선택한 조건에 해당하는 지도 데이터가 없습니다.");
        return;
    }

    const limit = getPeriodColorLimit(rows, metric);

    const trace = {
        type: "choropleth",
        geojson: BUSAN_GEOJSON,
        locations: rows.map(d => d["지역"]),
        featureidkey: "properties.지역",
        z: rows.map(d => isFiniteNumber(d[metric]) ? Number(d[metric]) : null),
        zmin: -limit,
        zmax: limit,
        zmid: 0,
        colorscale: GROWTH_COLORSCALE,
        marker: {
            line: {
                color: "#ffffff",
                width: 1.8,
            },
        },
        customdata: rows.map(d => [
            d["사업자수"],
            d["전월성장률"],
            d["전년동월성장률"],
            d["누적성장률"],
        ]),
        hovertemplate:
            "<b>%{location}</b>" +
            "<br>사업자 수: %{customdata[0]:,.0f}" +
            "<br>MoM: %{customdata[1]:.2f}%" +
            "<br>YoY: %{customdata[2]:.2f}%" +
            "<br>누적 증감률: %{customdata[3]:.2f}%" +
            "<extra></extra>",
        colorbar: {
            title: { text: "증감률 (%)" },
            thickness: 18,
            tickformat: ".1f",
            outlinewidth: 0,
        },
    };

    const layout = {
        title: {
            text: `${year}년 ${month}월 · ${industry} · ${metricLabel(metric)}`,
            x: 0.01,
            font: { size: 17 },
        },
        geo: {
            // 부산 부분만 화면에 고정합니다.
            visible: false,
            bgcolor: "#F8FAFC",
            projection: { type: "mercator" },
            center: { lat: 35.15, lon: 129.07 },
            lonaxis: { range: [128.65, 129.40] },
            lataxis: { range: [34.80, 35.48] },
        },
        margin: { l: 0, r: 0, t: 55, b: 0 },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        uirevision: "busan-map",
    };

    Plotly.react(
        "map",
        [trace],
        layout,
        { responsive: true, displaylogo: false }
    );

    clearStatus();

    const mapDiv = document.getElementById("map");
    mapDiv.removeAllListeners?.("plotly_click");
    mapDiv.on("plotly_click", eventData => {
        if (!eventData || !eventData.points || !eventData.points.length) return;
        const region = eventData.points[0].location;
        if (REGIONS.includes(region)) {
            regionSelect.value = region;
            compareRegionSelect.value = region;
            drawRegionTrend();
        }
    });
}


// =============================================================
// 지역별 비교: 선택한 월의 16개 구·군을 가로 막대로 비교
// =============================================================
function drawRegionComparison() {
    const industry = industrySelect.value;
    const metric = metricSelect.value;
    const year = Number(yearSelect.value);
    const month = Number(monthSelect.value);

    const rows = DATA
        .filter(d =>
            d["업종"] === industry
            && Number(d["연도"]) === year
            && Number(d["월"]) === month
            && isFiniteNumber(d[metric])
        )
        .sort((a, b) => Number(a[metric]) - Number(b[metric]));

    updateCards(rows, metric);

    if (rows.length === 0) {
        Plotly.purge("map");
        showStatus("선택한 조건에 해당하는 지역 비교 데이터가 없습니다.");
        return;
    }

    const limit = getPeriodColorLimit(rows, metric);
    const values = rows.map(d => Number(d[metric]));

    const trace = {
        type: "bar",
        orientation: "h",
        y: rows.map(d => d["지역"]),
        x: values,
        customdata: rows.map(d => [
            d["사업자수"],
            d["전월성장률"],
            d["전년동월성장률"],
            d["누적성장률"],
        ]),
        marker: {
            color: values,
            colorscale: GROWTH_COLORSCALE,
            cmin: -limit,
            cmax: limit,
            cmid: 0,
            line: { color: "rgba(255,255,255,0.9)", width: 1 },
            colorbar: {
                title: { text: "증감률 (%)" },
                thickness: 18,
                tickformat: ".1f",
                outlinewidth: 0,
            },
        },
        text: values.map(v => percentFormat(v)),
        textposition: "outside",
        cliponaxis: false,
        hovertemplate:
            "<b>%{y}</b>" +
            "<br>증감률: %{x:.2f}%" +
            "<br>사업자 수: %{customdata[0]:,.0f}" +
            "<br>MoM: %{customdata[1]:.2f}%" +
            "<br>YoY: %{customdata[2]:.2f}%" +
            "<br>누적 증감률: %{customdata[3]:.2f}%" +
            "<extra></extra>",
    };

    const padding = Math.max(limit * 0.18, 0.7);

    const layout = {
        title: {
            text: `${year}년 ${month}월 · ${industry} · 지역별 ${metricLabel(metric)}`,
            x: 0.01,
            font: { size: 17 },
        },
        xaxis: {
            title: "증감률 (%)",
            range: [-limit - padding, limit + padding],
            zeroline: true,
            zerolinecolor: "#475467",
            zerolinewidth: 1.2,
            gridcolor: "#E4E7EC",
        },
        yaxis: {
            title: "",
            automargin: true,
        },
        margin: { l: 85, r: 80, t: 60, b: 55 },
        paper_bgcolor: "white",
        plot_bgcolor: "#F8FAFC",
        bargap: 0.28,
    };

    Plotly.react(
        "map",
        [trace],
        layout,
        { responsive: true, displaylogo: false }
    );

    clearStatus();

    const chartDiv = document.getElementById("map");
    chartDiv.removeAllListeners?.("plotly_click");
    chartDiv.on("plotly_click", eventData => {
        if (!eventData?.points?.length) return;
        const region = eventData.points[0].y;
        if (REGIONS.includes(region)) {
            regionSelect.value = region;
            compareRegionSelect.value = region;
            drawRegionTrend();
        }
    });
}

// =============================================================
// 월별 비교: 선택 지역의 전체 월 시계열 비교
// =============================================================
function drawMonthlyComparison() {
    const industry = industrySelect.value;
    const metric = metricSelect.value;
    const region = compareRegionSelect.value || regionSelect.value;
    const year = Number(yearSelect.value);
    const month = Number(monthSelect.value);

    // 상단 KPI는 현재 선택한 연도·월 기준을 유지합니다.
    const periodRows = DATA.filter(d =>
        d["업종"] === industry
        && Number(d["연도"]) === year
        && Number(d["월"]) === month
    );
    updateCards(periodRows, metric);

    const rows = DATA
        .filter(d =>
            d["업종"] === industry
            && d["지역"] === region
        )
        .sort((a, b) =>
            String(a["연월표시"]).localeCompare(String(b["연월표시"]))
        );

    if (rows.length === 0) {
        Plotly.purge("map");
        showStatus("선택한 조건에 해당하는 월별 비교 데이터가 없습니다.");
        return;
    }

    const x = rows.map(d => d["연월표시"]);
    const y = rows.map(d => isFiniteNumber(d[metric]) ? Number(d[metric]) : null);

    const currentPeriod =
        `${yearSelect.value}-${String(monthSelect.value).padStart(2, "0")}`;

    const currentRow = rows.find(d => d["연월표시"] === currentPeriod);

    const traces = [
        {
            type: "scatter",
            mode: "lines+markers",
            name: region,
            showlegend: false,
            x: x,
            y: y,
            line: {
                color: "#2563EB",
                width: 3,
            },
            marker: {
                color: "#1D4ED8",
                size: 6,
            },
            customdata: rows.map(d => [
                d["사업자수"],
                d["전월성장률"],
                d["전년동월성장률"],
                d["누적성장률"],
            ]),
            hovertemplate:
                `<b>${region}</b>` +
                "<br>%{x}" +
                `<br>${metricLabel(metric)}: %{y:.2f}%` +
                "<br>사업자 수: %{customdata[0]:,.0f}" +
                "<br>MoM: %{customdata[1]:.2f}%" +
                "<br>YoY: %{customdata[2]:.2f}%" +
                "<br>누적 증감률: %{customdata[3]:.2f}%" +
                "<extra></extra>",
        },
    ];

    if (currentRow && isFiniteNumber(currentRow[metric])) {
        traces.push({
            type: "scatter",
            mode: "markers",
            name: "현재 선택 월",
            showlegend: false,
            x: [currentPeriod],
            y: [Number(currentRow[metric])],
            marker: {
                color: "#F59E0B",
                size: 14,
                line: { color: "#92400E", width: 2 },
            },
            hovertemplate:
                `<b>${currentPeriod}</b>` +
                `<br>${metricLabel(metric)}: %{y:.2f}%` +
                "<extra></extra>",
        });
    }

    const layout = {
        title: {
            text: `${region} · ${industry} · 월별 ${metricLabel(metric)}`,
            x: 0.01,
            xanchor: "left",
            y: 0.98,
            yanchor: "top",
            font: { size: 17 },
        },
        hovermode: "x unified",
        xaxis: {
            title: "",
            gridcolor: "#E4E7EC",
            rangeslider: { visible: false },
        },
        yaxis: {
            title: "증감률 (%)",
            zeroline: true,
            zerolinecolor: "#475467",
            zerolinewidth: 1.2,
            gridcolor: "#E4E7EC",
        },
        margin: { l: 70, r: 30, t: 82, b: 62 },
        paper_bgcolor: "white",
        plot_bgcolor: "#F8FAFC",
        showlegend: false,
    };

    Plotly.react(
        "map",
        traces,
        layout,
        { responsive: true, displaylogo: false }
    );

    clearStatus();
}

// =============================================================
// 지도 영역 보기 모드 전환
// =============================================================
function setMapView(view) {
    currentMapView = view;

    [
        ["map", mapViewBtn],
        ["region", regionCompareBtn],
        ["month", monthCompareBtn],
    ].forEach(([name, button]) => {
        button.classList.toggle("active", name === view);
    });

    compareRegionBox.classList.toggle("visible", view === "month");
    timelineBar.classList.toggle("visible", view === "map");

    if (view !== "map") {
        stopPlayback();
    }

    if (view === "map") {
        mapViewNote.textContent =
            "이전 월·다음 월·슬라이더·월별 재생으로 지도를 시간 순서대로 넘겨볼 수 있습니다. 지도에서 구·군을 클릭하면 아래 장기 사업자 수 추이도 연결됩니다.";
    } else if (view === "region") {
        mapViewNote.textContent =
            "선택한 연도·월의 부산 16개 구·군을 한 화면에서 비교합니다. 막대를 클릭하면 해당 지역의 장기 추이로 연결됩니다.";
    } else {
        mapViewNote.textContent =
            "선택 지역의 2022-01~2026-06 월별 흐름을 비교합니다. 주황색 점은 상단에서 선택한 현재 월입니다.";
    }

    drawMap();
}

function drawMap() {
    if (currentMapView === "region") {
        drawRegionComparison();
        return;
    }

    if (currentMapView === "month") {
        drawMonthlyComparison();
        return;
    }

    drawChoroplethMap();
}

// =============================================================
// Heatmap
// =============================================================
function drawHeatmap() {
    const industry = industrySelect.value;
    const metric = metricSelect.value;

    const rows = ANNUAL.filter(d => d["업종"] === industry);

    const years = [...new Set(
        rows.map(d => Number(d["연도"]))
            .filter(v => Number.isFinite(v))
    )].sort((a, b) => a - b);

    const z = REGIONS.map(region =>
        years.map(year => {
            const row = rows.find(d =>
                d["지역"] === region
                && Number(d["연도"]) === year
            );

            if (!row || !isFiniteNumber(row[metric])) return null;
            return Number(row[metric]);
        })
    );

    const flatValues = z
        .flat()
        .filter(v => isFiniteNumber(v))
        .map(v => Math.abs(Number(v)))
        .sort((a, b) => a - b);

    let limit = 1;
    if (flatValues.length > 0) {
        const idx = Math.min(
            Math.floor(flatValues.length * 0.90),
            flatValues.length - 1
        );
        limit = Math.max(flatValues[idx], 1);
    }

    const text = z.map(row =>
        row.map(value => isFiniteNumber(value) ? Number(value).toFixed(1) : "")
    );

    const trace = {
        type: "heatmap",
        x: years,
        y: REGIONS,
        z: z,
        zmin: -limit,
        zmax: limit,
        zmid: 0,
        text: text,
        texttemplate: "%{text}",
        textfont: { size: 12 },
        colorscale: GROWTH_COLORSCALE,
        colorbar: {
            title: { text: "증감률 (%)" },
            thickness: 16,
        },
        hovertemplate:
            "<b>%{y}</b>" +
            "<br>연도: %{x}" +
            "<br>증감률: %{z:.2f}%" +
            "<extra></extra>",
    };

    const layout = {
        title: {
            text: `${industry} · ${metricLabel(metric)}`,
            x: 0.01,
            font: { size: 16 },
        },
        margin: { l: 80, r: 45, t: 55, b: 55 },
        xaxis: { title: "연도", dtick: 1 },
        yaxis: { title: "지역", autorange: "reversed", automargin: true },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
    };

    Plotly.react(
        "heatmap",
        [trace],
        layout,
        { responsive: true, displaylogo: false }
    );
}

// =============================================================
// 지역별 장기 성장 추이
// =============================================================
function drawRegionTrend() {
    const region = regionSelect.value;
    const rows = DATA.filter(d => d["지역"] === region);

    const traces = [];

    INDUSTRIES.forEach(industry => {
        const industryRows = rows
            .filter(d => d["업종"] === industry)
            .sort((a, b) => String(a["연월표시"]).localeCompare(String(b["연월표시"])));

        if (industryRows.length === 0) return;

        traces.push({
            type: "scatter",
            mode: "lines",
            name: industry,
            x: industryRows.map(d => d["연월표시"]),
            y: industryRows.map(d => isFiniteNumber(d["성장지수"]) ? Number(d["성장지수"]) : null),
            customdata: industryRows.map(d => [
                d["사업자수"],
                d["전년동월성장률"],
                d["누적성장률"],
            ]),
            line: {
                width: industry === "전체" ? 3.4 : 2,
            },
            hovertemplate:
                `<b>${industry}</b>` +
                "<br>%{x}" +
                "<br>사업자 수 지수: %{y:.2f}" +
                "<br>사업자 수: %{customdata[0]:,.0f}" +
                "<br>YoY: %{customdata[1]:.2f}%" +
                "<br>누적: %{customdata[2]:.2f}%" +
                "<extra></extra>",
        });
    });

    const layout = {
        title: {
            text: `${region} 업종별 사업자 수 지수`,
            x: 0.01,
            font: { size: 17 },
        },
        hovermode: "x unified",
        xaxis: { title: "" },
        yaxis: { title: "사업자 수 지수 (첫 관측월 = 100)" },
        shapes: [
            {
                type: "line",
                xref: "paper",
                x0: 0,
                x1: 1,
                y0: 100,
                y1: 100,
                line: { color: "#999", dash: "dash", width: 1 },
            },
        ],
        legend: {
            orientation: "h",
            y: 1.12,
            x: 0,
        },
        margin: { l: 65, r: 30, t: 90, b: 60 },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
    };

    Plotly.react(
        "lineChart",
        traces,
        layout,
        { responsive: true, displaylogo: false }
    );
}

// =============================================================
// 업데이트
// =============================================================
function updateMainCharts() {
    try {
        drawMap();
    } catch (error) {
        console.error("지도 오류:", error);
        showStatus("지도 렌더링 오류: " + error.message);
    }

    try {
        drawHeatmap();
    } catch (error) {
        console.error("Heatmap 오류:", error);
        showStatus("Heatmap 렌더링 오류: " + error.message);
    }
}

// =============================================================
// 이벤트
// =============================================================
industrySelect.addEventListener("change", () => {
    syncIndustryQuickButtons();
    updateMainCharts();
});
metricSelect.addEventListener("change", updateMainCharts);

yearSelect.addEventListener("change", () => {
    stopPlayback();
    updateMonths();
    syncTimelineFromSelectors();
    drawMap();
});

monthSelect.addEventListener("change", () => {
    stopPlayback();
    syncTimelineFromSelectors();
    drawMap();
});

prevMonthBtn.addEventListener("click", () => {
    stopPlayback();
    movePeriod(-1);
});

nextMonthBtn.addEventListener("click", () => {
    stopPlayback();
    movePeriod(1);
});

playMonthBtn.addEventListener("click", togglePlayback);

monthTimeline.addEventListener("input", () => {
    stopPlayback();
    applyPeriodByIndex(Number(monthTimeline.value), true);
});

regionSelect.addEventListener("change", () => {
    compareRegionSelect.value = regionSelect.value;
    drawRegionTrend();

    if (currentMapView === "month") {
        drawMap();
    }
});

compareRegionSelect.addEventListener("change", () => {
    regionSelect.value = compareRegionSelect.value;
    drawRegionTrend();

    if (currentMapView === "month") {
        drawMap();
    }
});

mapViewBtn.addEventListener("click", () => setMapView("map"));
regionCompareBtn.addEventListener("click", () => setMapView("region"));
monthCompareBtn.addEventListener("click", () => setMapView("month"));

// =============================================================
// 초기 실행
// =============================================================
function initializeDashboard() {
    if (typeof Plotly === "undefined") {
        throw new Error("Plotly.js가 로드되지 않았습니다.");
    }

    if (!Array.isArray(DATA) || DATA.length === 0) {
        throw new Error("HTML에 삽입된 DATA가 비어 있습니다.");
    }

    console.log("[Dashboard] DATA rows:", DATA.length);
    console.log("[Dashboard] ANNUAL rows:", ANNUAL.length);
    console.log("[Dashboard] GeoJSON features:", BUSAN_GEOJSON?.features?.length);

    initializeSelects();
    syncTimelineFromSelectors();
    setMapView("map");
    drawHeatmap();
    drawRegionTrend();
}

try {
    initializeDashboard();
} catch (error) {
    console.error("대시보드 초기화 오류:", error);
    showStatus("대시보드 초기화 오류: " + error.message);
}
</script>
</body>
</html>
"""


# ==============================================================
# 12. Placeholder 치환
# ==============================================================

html = HTML_TEMPLATE

replacements = {
    "__PLOTLY_JS__": plotly_js,
    "__RECORDS_JSON__": records_json,
    "__ANNUAL_JSON__": annual_json,
    "__GEOJSON_JSON__": geojson_json,
    "__REGIONS_JSON__": regions_json,
    "__INDUSTRIES_JSON__": industries_json,
    "__HYPOTHESIS_HTML__": hypothesis_html,
}

for placeholder, value in replacements.items():
    expected_count = html.count(placeholder)
    if expected_count != 1:
        raise RuntimeError(
            f"템플릿의 {placeholder} 개수가 1개가 아닙니다: {expected_count}개"
        )
    html = html.replace(placeholder, value)

remaining = [key for key in replacements if key in html]
if remaining:
    raise RuntimeError(
        "HTML에 치환되지 않은 placeholder가 남아 있습니다: " + str(remaining)
    )

# 생성 HTML에 핵심 데이터/Plotly 코드가 실제 포함됐는지 확인
if "const DATA = [" not in html:
    raise RuntimeError("생성 HTML에 DATA 배열이 정상 삽입되지 않았습니다.")

if "Plotly" not in html:
    raise RuntimeError("생성 HTML에 Plotly.js가 정상 삽입되지 않았습니다.")

print("[OK] HTML placeholder 치환 완료")


# ==============================================================
# 13. HTML 저장
# ==============================================================

OUTPUT_PATH.write_text(html, encoding="utf-8")

size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024

print()
print("=" * 70)
print("[완료]")
print("HTML:", OUTPUT_PATH)
print(f"파일 크기: {size_mb:.2f} MB")
print("=" * 70)


# ==============================================================
# 14. 브라우저 자동 실행
# - 이번 버전은 외부 지도 타일을 쓰지 않으므로 file:// 직접 실행 가능
# ==============================================================

webbrowser.open(OUTPUT_PATH.resolve().as_uri())