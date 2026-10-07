"""원데이터 로드와 segment 분리 (2장 규칙).

- 완전히 같은 행(중복)은 1개만 남긴다.
- 시각 순서로 정렬한 뒤, 이웃 기록 간격이 정확히 0.1초가 아니면 그 자리에서 segment를 나눈다.
- 끊긴 구간을 보간하거나 이어 붙이지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import ANOMALY_CSV, NORMAL_CSV

SENSORS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
SENSOR_KO = {"AI0_Vibration": "AI0 상부 진동", "AI1_Vibration": "AI1 하부 진동", "AI2_Current": "AI2 전류"}
DT = 0.1
FILES = {"normal": NORMAL_CSV, "anomaly": ANOMALY_CSV}
LABEL_KO = {"normal": "정상", "anomaly": "이상"}


def read_raw(kind: str) -> pd.DataFrame:
    """원본 CSV를 그대로 읽는다 (품질 점검용)."""
    frame = pd.read_csv(FILES[kind])
    return frame.drop(columns=[c for c in frame.columns if c.startswith("Unnamed")])


def load_frames() -> dict[str, pd.DataFrame]:
    """정제한 정상·이상 데이터. interval(앞 기록과의 간격)과 segment_id(1부터)를 붙인다."""
    frames = {}
    for kind in FILES:
        raw = read_raw(kind)
        frame = raw.copy()
        frame["TimeStamp"] = pd.to_datetime(frame["TimeStamp"])
        frame = frame.dropna().drop_duplicates().sort_values("TimeStamp", kind="stable").reset_index(drop=True)
        delta = frame.TimeStamp.diff().dt.total_seconds()
        continuous = np.isclose(delta.to_numpy(), DT, atol=1e-9, rtol=0)
        continuous[0] = False
        frame["interval"] = delta
        frame["segment_id"] = (~continuous).cumsum()
        frame.attrs["raw_rows"] = len(raw)
        frames[kind] = frame
    return frames


def load_segments() -> list[dict]:
    """모델 학습용 segment 목록. group 이름은 'normal_5', 'anomaly_12'처럼 붙는다."""
    segments = []
    for kind, label in (("normal", 0), ("anomaly", 1)):
        frame = pd.read_csv(FILES[kind], index_col=0)
        frame["TimeStamp"] = pd.to_datetime(frame["TimeStamp"], errors="coerce")
        frame = frame.dropna(subset=["TimeStamp", *SENSORS]).drop_duplicates()
        frame = frame.sort_values("TimeStamp", kind="stable")
        delta = frame["TimeStamp"].diff().dt.total_seconds()
        segment_id = (delta.isna() | (delta <= 0) | (delta.sub(DT).abs() > 1e-9)).cumsum()
        for number, (_, group) in enumerate(frame.groupby(segment_id, sort=False), 1):
            values = group[SENSORS].to_numpy(dtype=np.float32)
            if np.isfinite(values).all():
                segments.append({
                    "group": f"{kind}_{number}",
                    "label": label,
                    "values": values,
                    "samples": len(values),
                    "start_time": group.TimeStamp.iloc[0],
                })
    return segments
