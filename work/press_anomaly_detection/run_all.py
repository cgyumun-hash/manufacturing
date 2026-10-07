"""학습 → 기준값 결정 → test 예측을 한 번에 실행한다 (Windows · macOS · Linux 공통).

실행: python run_all.py
"""
import time

import predict_test
import train
import valid

if __name__ == "__main__":
    started = time.perf_counter()
    for step, module in (("1/3 학습", train), ("2/3 기준값 결정", valid), ("3/3 test 예측·평가", predict_test)):
        print(f"\n===== {step} =====", flush=True)
        module.main()
    print(f"\n전체 완료: {time.perf_counter() - started:.1f}초", flush=True)
