"""원데이터 로드와 segment 분리.

- 시각·센서 값이 빈 행 제거, 완전히 같은 행(중복)은 1개만 남김
- 시각 순서로 정렬한 뒤 이웃 기록 간격이 정확히 0.1초가 아니면 그 자리에서 segment를 나눔 (보간하지 않음)
"""
import numpy as np
import pandas as pd

from .config import ANOMALY_CSV, DT, MIN_SAMPLES, NORMAL_CSV, SENSORS


def load_segments() -> list:
    segments = []
    for kind, path, label in (("normal", NORMAL_CSV, 0), ("anomaly", ANOMALY_CSV, 1)):
        frame = pd.read_csv(path, index_col=0)
        frame["TimeStamp"] = pd.to_datetime(frame["TimeStamp"], errors="coerce")
        frame = frame.dropna(subset=["TimeStamp", *SENSORS]).drop_duplicates()
        frame = frame.sort_values("TimeStamp", kind="stable")
        delta = frame["TimeStamp"].diff().dt.total_seconds()
        segment_id = (delta.isna() | (delta <= 0) | (delta.sub(DT).abs() > 1e-9)).cumsum()
        for number, (_, group) in enumerate(frame.groupby(segment_id, sort=False), 1):
            values = group[SENSORS].to_numpy(dtype=np.float32)
            if np.isfinite(values).all():
                segments.append({"group": f"{kind}_{number}", "label": label, "values": values,
                                 "samples": len(values), "start_time": group["TimeStamp"].iloc[0],
                                 "end_time": group["TimeStamp"].iloc[-1]})
    return segments


def eligible_segments() -> list:
    return [s for s in load_segments() if s["samples"] >= MIN_SAMPLES]
