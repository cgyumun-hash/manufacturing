"""2.2 데이터 품질 확인 — 필수 열, 빈 값, 무한대, 중복, 같은 시각, 시간 역행, 0.1초보다 긴 공백.

출력: 표_품질_점검.csv, 표_중복행.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import DT, SENSORS, read_raw  # noqa: E402
from common.paths import save_table  # noqa: E402

REQUIRED = ["TimeStamp", *SENSORS, "Equipment_state"]
checks, duplicates = {}, []
for kind in ["normal", "anomaly"]:
    raw = read_raw(kind)
    t = pd.to_datetime(raw["TimeStamp"], errors="coerce")
    values = raw[SENSORS].to_numpy(dtype=float)
    dup_mask = raw.duplicated(keep="first")
    clean = raw.assign(TimeStamp=t).dropna().drop_duplicates().sort_values("TimeStamp", kind="stable")
    delta = clean.TimeStamp.diff().dt.total_seconds().iloc[1:]
    checks[kind] = {
        "필수 열(시각, 센서 3개, 라벨)": "있음" if all(c in raw.columns for c in REQUIRED) else "없음",
        "빈 값·읽기 실패": int(raw[REQUIRED].isna().sum().sum() + t.isna().sum()),
        "무한대 값": int(np.isinf(values).sum()),
        "완전히 같은 행(추가분)": int(dup_mask.sum()),
        "같은 시각이 두 번": int(t.duplicated().sum()),
        "시간이 거꾸로 가거나 0초 간격": int((delta <= 0).sum()),
        "0.1초보다 긴 공백": int((~np.isclose(delta, DT, atol=1e-9, rtol=0)).sum()),
    }
    if dup_mask.any():
        dup_rows = raw[raw.duplicated(keep=False)].assign(**{"class": kind})
        duplicates.append(dup_rows[["class", *REQUIRED]])

result = {"필수 열(시각, 센서 3개, 라벨)": "통과", "빈 값·읽기 실패": "통과 — 처리 불필요", "무한대 값": "통과 — 처리 불필요",
          "완전히 같은 행(추가분)": "주의 — 분할 전에 1행만 남기고 제거", "같은 시각이 두 번": "위 중복행을 지우면 해결",
          "시간이 거꾸로 가거나 0초 간격": "통과", "0.1초보다 긴 공백": "주의 — segment로 나눔 (2.5)"}
table = pd.DataFrame([{"점검 항목": item, "정상": checks["normal"][item], "이상": checks["anomaly"][item], "결과·처리": result[item]}
                      for item in result])
save_table(table, HERE / "표_품질_점검.csv")
save_table(pd.concat(duplicates), HERE / "표_중복행.csv")
print(table.to_string(index=False))
