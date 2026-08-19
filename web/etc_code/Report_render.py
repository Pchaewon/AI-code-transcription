# -*- coding: utf-8 -*-
"""
Frame + Slurry 각각 별도 모델 학습 + 역산 (분리 방식)
─────────────────────────────────────────
관찰: frame+slurry를 한 모델에 넣으면 R² 0.1 (feature 희석),
      각각 따로 모델 만들면 각 0.2+ → 분리가 유리.

구조:
  · Frame 모델: set_frame_temp 12개 + roll_조건 + 장비더미 → BOW
  · Slurry 모델: set_slurry_temp 12개 + roll_조건 + 장비더미 → BOW
  · 각 모델로 해당 프로파일만 역산

앙상블 옵션:
  · 예측 시 두 모델 평균 (예측 정확도용)
  · 역산은 각 프로파일 독립 (frame은 frame모델, slurry는 slurry모델)
"""
import os
import os.path as pt
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from rolling_run_features import add_rolling_run_features

CONFIG = {
    'input_csv':  r'./data/mixed_pt.csv',
    'model_dir':  r'./apc_model_dual',
    'process_time': '13.3Hr',
    'target':     'avg_bow_bf_total',
    'wire_col':   'fdc_new_wire_id',
    'eqp_col':    'eqp_nm_3200',
    'date_col':   'date_3200',
    # 두 프로파일 (각각 별도 모델)
    'profiles': {
        'frame':  [f'set_frame_temp_{p}pct' for p in
                   [0,10,20,30,40,50,60,70,80,90,99,100]],
        'slurry': [f'set_slurry_temp_{p}pct' for p in
                   [0,10,20,30,40,50,60,70,80,90,99,100]],
    },
    'roll_source_cols': ['fdc_set_tension','fdc_wait_time','fdc_ingot_len',
                         'range_slurry_temp_10_0'],
    'lag': 2, 'window': 10, 'min_runs': 3,
    'use_eqp_dummy': True,
    'ridge_alpha': 5.0,
    'split_ratio': 0.8,
    'lambda_smooth': 0.0,
    'inverse_target_eqps': ['BSWS38','BSWS42','BSWS44'],
    'inverse_target_bow': 1.75,
    'test_csv': r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',
    'inverse_out_csv': r'./inverse_dual_by_eqp.csv',
    'encoding': 'utf-8',
}


def build_dataset(cfg, profile_cols):
    """특정 프로파일용 rolling 데이터."""
    df = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    if cfg['process_time']:
        df = df[df['process_time'] == cfg['process_time']]
    roll_cfg = {
        'wire_col': cfg['wire_col'], 'date_col': cfg['date_col'],
        'eqp_col': cfg['eqp_col'], 'target_col': cfg['target'],
        'feature_cols': profile_cols + cfg['roll_source_cols'],
        'lag': cfg['lag'], 'window': cfg['window'], 'min_runs': cfg['min_runs'],
    }
    return add_rolling_run_features(df, roll_cfg)


def train_one_profile(cfg, name, profile_cols):
    """단일 프로파일 모델 학습·저장."""
    df = build_dataset(cfg, profile_cols)
    EQP = cfg['eqp_col']; TARGET = cfg['target']; DATE = cfg['date_col']
    roll_cols = [f'roll_{c}' for c in cfg['roll_source_cols']]
    base_feats = profile_cols + roll_cols

    sub = df[base_feats + [TARGET, DATE, EQP]].dropna().copy()
    sub = sub.sort_values(DATE).reset_index(drop=True)

    if cfg['use_eqp_dummy']:
        dummies = pd.get_dummies(sub[EQP], prefix='eqp')
        FEATURES = base_feats + list(dummies.columns)
        sub = pd.concat([sub, dummies], axis=1)
        eqp_cols = list(dummies.columns)
    else:
        FEATURES = base_feats; eqp_cols = []

    X = sub[FEATURES].values.astype(float); y = sub[TARGET].values
    si = int(len(sub) * cfg['split_ratio'])
    sc = StandardScaler().fit(X[:si])
    m = Ridge(alpha=cfg['ridge_alpha']).fit(sc.transform(X[:si]), y[:si])
    pred = m.predict(sc.transform(X[si:]))
    r2 = r2_score(y[si:], pred); mae = mean_absolute_error(y[si:], pred)
    print(f"  [{name}] N={len(sub)}, 시간분할 R²={r2:.4f}, MAE={mae:.4f}")

    # 전체 재학습
    sc_full = StandardScaler().fit(X)
    m_full = Ridge(alpha=cfg['ridge_alpha']).fit(sc_full.transform(X), y)

    mdir = pt.join(cfg['model_dir'], name)
    os.makedirs(mdir, exist_ok=True)
    with open(pt.join(mdir, 'model.pkl'), 'wb') as f: pickle.dump(m_full, f)
    with open(pt.join(mdir, 'scaler.pkl'), 'wb') as f: pickle.dump(sc_full, f)

    def stats(a):
        return {'mean': float(np.mean(a)), 'std': float(np.std(a)),
                'q01': float(np.quantile(a,0.01)), 'q99': float(np.quantile(a,0.99))}
    meta = {
        'name': name, 'target': TARGET, 'feature_cols': FEATURES,
        'profile_cols': profile_cols, 'roll_cols': roll_cols,
        'eqp_cols': eqp_cols, 'eqp_prefix': 'eqp_',
        'x_stats': {c: stats(sub[c].values) for c in base_feats},
        'metrics': {'r2_time': round(r2,4), 'mae': round(mae,4)},
    }
    with open(pt.join(mdir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {'name': name, 'r2': round(r2,4), 'mae': round(mae,4), 'n': len(sub)}


def train_all_process_times(cfg, process_times=('13.3Hr', '18.5Hr')):
    """
    13.3Hr / 18.5Hr 각각 별도 모델 학습.
    구조: model_dir/13.3Hr/frame,slurry  +  model_dir/18.5Hr/frame,slurry
    """
    import copy
    base_dir = cfg['model_dir']
    all_results = {}
    for pt_val in process_times:
        print(f"\n{'#'*56}\n# process_time = {pt_val}\n{'#'*56}")
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg['process_time'] = pt_val
        sub_cfg['model_dir'] = f"{base_dir}/{pt_val}"   # 하위 폴더
        results = train(sub_cfg)
        all_results[pt_val] = results
    print(f"\n{'='*56}\n전체 완료: {list(process_times)}\n{'='*56}")
    for pt_val, res in all_results.items():
        print(f"  [{pt_val}]")
        for r in res:
            print(f"    {r['name']}: R²={r['r2']}, MAE={r['mae']} (N={r['n']})")
    return all_results


def train(cfg):
    os.makedirs(cfg['model_dir'], exist_ok=True)
    print(f"[분리 학습] frame / slurry 각각 별도 모델\n")
    results = []
    for name, cols in cfg['profiles'].items():
        # 프로파일 컬럼 존재 확인
        df_head = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                              encoding_errors='replace', nrows=5)
        avail = [c for c in cols if c in df_head.columns]
        if len(avail) == 0:
            print(f"  [{name}] 컬럼 없음 (예: {cols[0]}) — 스킵")
            continue
        results.append(train_one_profile(cfg, name, cols))
    # 요약
    print(f"\n{'='*50}\n분리 학습 요약\n{'='*50}")
    for r in results:
        print(f"  {r['name']}: R²={r['r2']}, MAE={r['mae']} (N={r['n']})")
    print(f"\n💾 저장: {cfg['model_dir']}/")
    return results


def evaluate_ensemble(cfg):
    """
    앙상블(frame+slurry 예측 평균)이 각 단독보다 나은지 시간분할로 평가.
    ★ 앙상블 이득 판정:
      · 앙상블 R² > max(frame, slurry) → 상호보완적, 앙상블 채택
      · 앙상블 R² ≈ 단독 → 같은 정보, 하나만 써도 됨
    """
    print(f"\n{'='*50}\n앙상블 평가 (frame+slurry 예측 평균)\n{'='*50}")
    # 공통 rolling 데이터 (frame 기준, roll_조건 공통)
    frame_cols = cfg['profiles']['frame']
    slurry_cols = cfg['profiles']['slurry']
    df = build_dataset(cfg, frame_cols + slurry_cols)  # 둘 다 포함해 결측 정렬
    EQP = cfg['eqp_col']; TARGET = cfg['target']; DATE = cfg['date_col']
    roll_cols = [f'roll_{c}' for c in cfg['roll_source_cols']]

    need = frame_cols + slurry_cols + roll_cols + [TARGET, DATE, EQP]
    need = [c for c in need if c in df.columns]
    sub = df[need].dropna().sort_values(DATE).reset_index(drop=True)
    si = int(len(sub) * cfg['split_ratio'])
    test = sub.iloc[si:]
    if len(test) < 10:
        print("  평가 샘플 부족"); return

    # 각 모델 로드 후 test 예측
    def model_predict(name, prof_cols):
        try:
            model, scaler, meta = load_profile_model(cfg, name)
        except FileNotFoundError:
            return None
        FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
        eqp_cols = meta['eqp_cols']; pfx = meta['eqp_prefix']
        mprofile = meta['profile_cols']; mroll = meta['roll_cols']
        # test 각 행 예측
        dummies = pd.get_dummies(test[EQP], prefix='eqp')
        preds = []
        for _, row in test.iterrows():
            def gv(c):
                if c in eqp_cols:
                    return 1.0 if c == f"{pfx}{row[EQP]}" else 0.0
                if c in mprofile:
                    return float(row[c]) if c in row and pd.notna(row[c]) \
                           else X_STATS.get(c,{}).get('mean',0.0)
                if c in mroll:
                    return float(row[c]) if c in row and pd.notna(row[c]) \
                           else X_STATS.get(c,{}).get('mean',0.0)
                return float(X_STATS.get(c,{}).get('mean',0.0))
            x = np.array([gv(c) for c in FEATURES]).reshape(1,-1)
            preds.append(float(model.predict(scaler.transform(x))[0]))
        return np.array(preds)

    y_true = test[TARGET].values
    p_frame = model_predict('frame', frame_cols)
    p_slurry = model_predict('slurry', slurry_cols)

    if p_frame is None or p_slurry is None:
        print("  모델 없음 — train 먼저 실행"); return

    r2_frame = r2_score(y_true, p_frame)
    r2_slurry = r2_score(y_true, p_slurry)
    # 앙상블: 단순 평균
    p_ens = (p_frame + p_slurry) / 2
    r2_ens = r2_score(y_true, p_ens)
    mae_ens = mean_absolute_error(y_true, p_ens)

    # 가중 평균 최적 탐색 (frame 비중 w)
    best_w, best_r2 = 0.5, r2_ens
    for w in np.linspace(0, 1, 21):
        r2_w = r2_score(y_true, w*p_frame + (1-w)*p_slurry)
        if r2_w > best_r2:
            best_r2, best_w = r2_w, w

    print(f"  frame 단독:    R²={r2_frame:.4f}")
    print(f"  slurry 단독:   R²={r2_slurry:.4f}")
    print(f"  앙상블(평균):  R²={r2_ens:.4f}, MAE={mae_ens:.4f}")
    print(f"  앙상블(최적):  R²={best_r2:.4f} (frame 비중 {best_w:.2f})")
    print()
    base = max(r2_frame, r2_slurry)
    if r2_ens > base + 0.01:
        print(f"  ✅ 앙상블 이득 (+{r2_ens-base:.3f}) → 상호보완적, 앙상블 채택")
    elif best_r2 > base + 0.01:
        print(f"  ⭕ 가중 앙상블만 이득 (frame {best_w:.2f}) → 가중 평균 권장")
    else:
        print(f"  ⚠ 앙상블 이득 미미 → 두 모델 정보 유사. "
              f"예측은 단독으로 충분 (역산은 각각 유지)")
    return {'frame': r2_frame, 'slurry': r2_slurry,
            'ensemble': r2_ens, 'best_w': best_w, 'best_r2': best_r2}


# ═══════════════════════════════════════
# 역산 (프로파일별 해당 모델 사용)
# ═══════════════════════════════════════
def load_profile_model(cfg, name):
    mdir = pt.join(cfg['model_dir'], name)
    with open(pt.join(mdir, 'model.pkl'), 'rb') as f: model = pickle.load(f)
    with open(pt.join(mdir, 'scaler.pkl'), 'rb') as f: scaler = pickle.load(f)
    with open(pt.join(mdir, 'meta.json'), encoding='utf-8') as f: meta = json.load(f)
    return model, scaler, meta


def inverse_profile(cfg, name, target_bow, roll_values, eqp_name,
                    current_profile=None):
    """해당 프로파일 모델로 역산."""
    model, scaler, meta = load_profile_model(cfg, name)
    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    profile_cols = meta['profile_cols']; roll_cols = meta['roll_cols']
    eqp_cols = meta['eqp_cols']; pfx = meta['eqp_prefix']

    def gv(c, override):
        if c in eqp_cols:
            return 1.0 if c == f'{pfx}{eqp_name}' else 0.0
        if c in profile_cols and override is not None:
            return float(override[profile_cols.index(c)])
        if c in roll_cols:
            return float(roll_values.get(c, X_STATS.get(c,{}).get('mean',0.0)))
        return float(X_STATS.get(c,{}).get('mean',0.0))

    def predict(vec):
        x = np.array([gv(c, vec) for c in FEATURES]).reshape(1,-1)
        return float(model.predict(scaler.transform(x))[0])

    def objective(vec):
        loss = (predict(vec) - target_bow)**2
        if cfg['lambda_smooth'] > 0:
            loss += cfg['lambda_smooth'] * np.sum(np.diff(vec)**2)
        return loss

    if current_profile is not None:
        x0 = np.array(current_profile)
    else:
        x0 = np.array([X_STATS[c]['mean'] for c in profile_cols])
    bounds = [(X_STATS[c]['q01'], X_STATS[c]['q99']) for c in profile_cols]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter':300,'ftol':1e-9})
    rec = {c: round(float(v),4) for c,v in zip(profile_cols, res.x)}
    return {'profile': name, 'recipe': rec,
            'predicted_bow': round(predict(res.x),4), 'target_bow': target_bow}


def predict_ensemble(cfg, roll_values, eqp_name, frame_prof, slurry_prof):
    """두 모델 예측 평균 (모니터링용)."""
    preds = []
    for name, prof in [('frame', frame_prof), ('slurry', slurry_prof)]:
        try:
            model, scaler, meta = load_profile_model(cfg, name)
        except FileNotFoundError:
            continue
        FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
        profile_cols = meta['profile_cols']; roll_cols = meta['roll_cols']
        eqp_cols = meta['eqp_cols']; pfx = meta['eqp_prefix']
        def gv(c):
            if c in eqp_cols:
                return 1.0 if c == f'{pfx}{eqp_name}' else 0.0
            if c in profile_cols and prof is not None:
                return float(prof[profile_cols.index(c)])
            if c in roll_cols:
                return float(roll_values.get(c, X_STATS.get(c,{}).get('mean',0.0)))
            return float(X_STATS.get(c,{}).get('mean',0.0))
        x = np.array([gv(c) for c in FEATURES]).reshape(1,-1)
        preds.append(float(model.predict(scaler.transform(x))[0]))
    return float(np.mean(preds)) if preds else None


def inverse_by_equipment(cfg):
    """장비별 frame+slurry 역산 (각 모델)."""
    test_df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                          encoding_errors='replace')
    if cfg['process_time']:
        test_df = test_df[test_df['process_time'] == cfg['process_time']]

    # frame 기준 rolling (roll_조건은 공통)
    roll_cfg = {
        'wire_col': cfg['wire_col'], 'date_col': cfg['date_col'],
        'eqp_col': cfg['eqp_col'], 'target_col': cfg['target'],
        'feature_cols': cfg['profiles']['frame'] + cfg['roll_source_cols'],
        'lag': cfg['lag'], 'window': cfg['window'], 'min_runs': cfg['min_runs'],
    }
    test_df = add_rolling_run_features(test_df, roll_cfg)
    roll_cols = [f'roll_{c}' for c in cfg['roll_source_cols']]

    EQP = cfg['eqp_col']; tb = cfg['inverse_target_bow']
    rows = []
    for eqp in cfg['inverse_target_eqps']:
        esub = test_df[test_df[EQP] == eqp].dropna(subset=roll_cols)
        if len(esub) == 0:
            print(f"  {eqp}: 직전 평균 있는 행 없음"); continue
        print(f"  {eqp}: {len(esub)}개 lot")
        for _, row in esub.iterrows():
            roll_values = {rc: float(row[rc]) for rc in roll_cols}
            out = {'eqp': eqp, 'target_bow': tb}
            # frame 역산
            fr = inverse_profile(cfg, 'frame', tb, roll_values, eqp)
            out['frame_pred_bow'] = fr['predicted_bow']
            for c, v in fr['recipe'].items():
                out[f'rec_{c}'] = v
            # slurry 역산 (모델 있으면)
            if os.path.exists(pt.join(cfg['model_dir'], 'slurry')):
                sl = inverse_profile(cfg, 'slurry', tb, roll_values, eqp)
                out['slurry_pred_bow'] = sl['predicted_bow']
                for c, v in sl['recipe'].items():
                    out[f'rec_{c}'] = v
            if cfg['wire_col'] in row:
                out['wire_id'] = row[cfg['wire_col']]
            rows.append(out)
    res = pd.DataFrame(rows)
    res.to_csv(cfg['inverse_out_csv'], index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {cfg['inverse_out_csv']} ({len(res)} lot)")
    return res


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if mode == 'train':
        train(CONFIG)
        evaluate_ensemble(CONFIG)   # 학습 후 앙상블 평가 자동 실행
    elif mode == 'train_all':
        # ★ 13.3Hr / 18.5Hr 각각 학습 (하위 폴더)
        train_all_process_times(CONFIG)
    elif mode == 'ensemble':
        evaluate_ensemble(CONFIG)
    elif mode == 'by_eqp':
        inverse_by_equipment(CONFIG)
    else:
        # 단일 예시
        rv = {f'roll_{c}': v for c, v in
              [('fdc_set_tension',0.8),('fdc_wait_time',45),
               ('fdc_ingot_len',38),('range_slurry_temp_10_0',2.3)]}
        for name in ['frame','slurry']:
            if os.path.exists(pt.join(CONFIG['model_dir'], name)):
                r = inverse_profile(CONFIG, name, 1.75, rv, 'BSWS38')
                print(f"\n[{name}] 예측BOW={r['predicted_bow']}")
