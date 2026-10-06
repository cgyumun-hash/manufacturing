# 프레스 이상 탐지 스토리텔링 분석

스토리텔링 순서에 맞춘 분석 결과만 이 폴더에서 관리한다. 기존 프로젝트 루트의 실험 파일은 이동하거나 삭제하지 않고, 이후 새로 생성하는 표·이미지·설명 문서는 해당 단계의 하위 폴더에 저장한다.

## 폴더 구조

```text
storytelling_analysis/
├── README.md
└── 02_data_quality/
    ├── 2.2_데이터_품질_확인.md
    ├── scripts/
    │   └── analyze_data_quality.py
    ├── tables/
    │   └── 데이터 품질 점검 CSV
    └── figures/
        └── 원격 확인용 PNG 이미지
```

## 진행 상태

| 스토리 순서 | 상태 | 결과 위치 |
|---|---|---|
| 1. 상황 설명 | 초안 완료 | 프로젝트 루트의 전체 스토리텔링 문서 |
| 2.2 데이터 품질 확인 | 완료 | `02_data_quality/` |
| 정상·이상 탐색 및 시각화 | 예정 | 다음 단계에서 별도 폴더 생성 |
| 모델 성능 비교 | 기존 결과 정리 예정 | 다음 단계 |
| 해석·한계 | 기존 결과 정리 예정 | 다음 단계 |
| 실시간 조기 탐지 확장 | 설계 예정 | 다음 단계 |

## 2.2 재실행

프로젝트 루트에서 다음 명령을 실행한다.

```bash
python storytelling_analysis/02_data_quality/scripts/analyze_data_quality.py
```

보고서와 CSV, PNG는 `02_data_quality/` 아래에 덮어쓴다.

