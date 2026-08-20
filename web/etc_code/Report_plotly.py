
# -*- coding: utf-8 -*-
"""
계수 이상 진단 — 온도 pct 계수가 큰 이유 분석
─────────────────────────────────────────
온도 pct 계수가 비정상적으로 크면(예: 3.86) 다중공선성 의심.
  · 큰 계수들이 서로 상쇄되는지
  · 온도 pct들 간 상관이 높은지
  · 온도 합산 효과는 정상인지

실행:
  python diagnose_coef.py          # frame
  python diagnose_coef.py slurry
"""
import sys
import numpy as np
import pandas as pd
import train_inverse_dual as dual


def diagnose(name):
    cfg = dual.CONFIG
    model, scaler, meta = dual.load_profile_model(cfg, name)
    FEATURES = meta['feature_cols']
    coef = model.coef_
    profile_cols = meta['profile_cols']

    # 온도 pct 계수만
    temp_idx = [i for i, f in enumerate(FEATURES) if f in profile_cols]
    temp_coef = coef[temp_idx]
    temp_names = [FEATURES[i] for i in temp_idx]

    print(f"\n{'='*56}\n[{name}] 온도 pct 계수 진단\n{'='*56}")
    print(f"\n온도 계수 값:")
    for n, c in zip(temp_names, temp_coef):
        bar = '█' * int(abs(c) * 3)
        print(f"  {n:24s} {c:+.3f}  {bar}")

    # 1. 큰 계수 확인
    max_abs = np.max(np.abs(temp_coef))
    print(f"\n[1] 최대 절대 계수: {max_abs:.3f}")
    if max_abs > 1.0:
        print(f"    ⚠ 1.0 초과 — 비정상적으로 큼 (다중공선성 의심)")

    # 2. 상쇄 여부 (합 vs 절대값 합)
    coef_sum = np.sum(temp_coef)
    abs_sum = np.sum(np.abs(temp_coef))
    print(f"\n[2] 계수 합: {coef_sum:+.3f}  vs  절대값 합: {abs_sum:.3f}")
    if abs_sum > 3 * abs(coef_sum) and abs_sum > 1:
        print(f"    ⚠ 큰 계수들이 서로 상쇄됨 → 다중공선성 확실")
        print(f"       (개별은 크지만 합치면 {coef_sum:+.3f}로 작음)")
    else:
        print(f"    → 상쇄 없음, 계수가 순수하게 큼")

    # 3. 온도 pct 간 상관 (공선성 직접 확인)
    print(f"\n[3] 온도 pct 간 상관 (학습 데이터)")
    df = dual.build_dataset(cfg, profile_cols)
    temp_data = df[profile_cols].dropna()
    if len(temp_data) > 10:
        corr = temp_data.corr().values
        # 대각선 제외 평균 상관
        off_diag = corr[~np.eye(len(corr), dtype=bool)]
        mean_corr = np.mean(np.abs(off_diag))
        max_corr = np.max(np.abs(off_diag))
        print(f"    평균 |상관|: {mean_corr:.3f},  최대: {max_corr:.3f}")
        if mean_corr > 0.8:
            print(f"    ⚠ 온도 pct들이 강하게 상관 → 공선성 원인 확인")
        elif mean_corr > 0.5:
            print(f"    · 중간 상관 (일부 공선성)")
        else:
            print(f"    · 상관 낮음")

    # 4. 결론
    print(f"\n{'='*56}\n[결론]")
    if max_abs > 1.0 and abs_sum > 3 * abs(coef_sum):
        print("  다중공선성으로 개별 계수가 큼/상쇄됨.")
        print("  → 개별 pct 계수는 해석 주의. '온도 전체'로 봐야 함.")
        print(f"  → 온도 순 효과(합): {coef_sum:+.3f} (이게 실제 방향)")
        if coef_sum < 0:
            print("    온도↑ → BOW↓ (공정 지식 일치) ✓")
        else:
            print("    온도↑ → BOW↑ (확인 필요)")
    else:
        print("  계수가 정상 범위이거나 상쇄 없음.")
    print('='*56)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'frame'
    diagnose(which)
