# -*- coding: utf-8 -*-
"""
report_data.py — 리포트용 데이터 로딩 (Plotly 버전)
──────────────────────────────────────────────────
· 실측: 가공시간(process_time) 통합, 시간순 전체 트렌드
· x축 4단계: 시간(mm/dd hh) → wire id → wire lifetime → blk_id
· 추천: 가장 최근 끝난 가공시간 1개만
"""
import pandas as pd
import numpy as np

PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

ACT_FRAME = 'FRAME_IN_TEMP_{p}pct'
ACT_SLURRY = 'SLURRY_IN_TEMP_{p}pct'
ACT_WG_L = 'SHIFT_AMOUNT_WIREGUIDE_L_{p}pct'
ACT_WG_R = 'SHIFT_AMOUNT_WIREGUIDE_R_{p}pct'

STORE_CFG = {
    'eqp':  'eqp_nm_3200',
    'wire': 'fdc_new_wire_id',
    'date': 'date_3200',
    'bow':  'avg_bow_bf_total',
    'warp': 'avg_warp_bf_total',
    'bow_seed': 'avg_bow_bf_seed', 'bow_mid': 'avg_bow_bf_mid', 'bow_tail': 'avg_bow_bf_tail',
    'warp_seed': 'avg_warp_bf_seed', 'warp_mid': 'avg_warp_bf_mid', 'warp_tail': 'avg_warp_bf_tail',
    'ingot': 'fdc_ingot_len', 'wait': 'fdc_wait_time', 'warm': 'fdc_warm_up_time',
    'lifetime': 'WIREGUIDE_LIFE_TIME',   # wire lifetime (x축 3단계)
    'blk': 'BLK_NO',                     # blk id (x축 4단계)
    'pt': 'process_time',
}

RECENT_N = 10   # 최근 N lot

MISSING = {-1.0, -1, -999, -9999}


def _real(df, name):
    """대소문자 무관 컬럼 조회."""
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower())


def _fmt_date(raw):
    """YYYYMMDDHHMMSS... → 'MM-DD HH' (시간까지)."""
    digits = ''.join(ch for ch in str(raw) if ch.isdigit())
    if len(digits) >= 10:
        return f"{digits[4:6]}-{digits[6:8]} {digits[8:10]}h"
    return str(raw)[:13]


def _profile_row(row, tmpl, df_cols_lower):
    """한 행에서 0~100pct 프로파일 리스트."""
    out = []
    for p in PCTS:
        col = df_cols_lower.get(tmpl.format(p=p).lower())
        v = row.get(col) if col else None
        if v is None or (isinstance(v, float) and np.isnan(v)) or v in MISSING:
            out.append(None)
        else:
            out.append(round(float(v), 3))
    return out


def load_actuals(store_path):
    """장비별 실측 데이터 로딩 — 가공시간 통합, 시간순 전체."""
    df = pd.read_csv(store_path)
    C = STORE_CFG
    cl = {c.lower(): c for c in df.columns}

    if C['eqp'] not in df.columns:
        print(f"  ⚠ {C['eqp']} 컬럼 없음")
        return {}

    LOT = 'lot_id'
    has_pt = C['pt'] in df.columns
    date_col = C['date']
    wire_col = C['wire']
    life_col = _real(df, C['lifetime'])
    blk_col = _real(df, C['blk'])

    acts = {}
    for eqp, g in df.groupby(C['eqp']):
        g = g.copy()
        if date_col in g.columns:
            g = g.sort_values(date_col)

        has_lot = LOT in g.columns
        # 최근 N lot
        if has_lot:
            lot_order = list(dict.fromkeys(g[LOT].astype(str).tolist()))
            recent = set(lot_order[-RECENT_N:])
            g = g[g[LOT].astype(str).isin(recent)]

        # 각 lot(행)을 시간순 point로 — 가공시간 통합
        points = []
        for _, r in g.iterrows():
            rd = r.to_dict()
            pt_val = str(r.get(C['pt'], '')) if has_pt else ''
            pt = {
                'wire': str(r.get(wire_col, '')),
                'date': _fmt_date(r.get(date_col, '')),
                'date_raw': str(r.get(date_col, '')),
                'lifetime': (round(float(r[life_col]), 1)
                             if life_col and pd.notna(r.get(life_col)) else ''),
                'blk': str(r.get(blk_col, '')) if blk_col else '',
                'lot': str(r.get(LOT, '')) if has_lot else '',
                'process_time': pt_val,
                # 품질
                'bow': _num(r.get(C['bow'])),
                'bow_seed': _num(r.get(C['bow_seed'])), 'bow_mid': _num(r.get(C['bow_mid'])), 'bow_tail': _num(r.get(C['bow_tail'])),
                'warp': _num(r.get(C['warp'])),
                'warp_seed': _num(r.get(C['warp_seed'])), 'warp_mid': _num(r.get(C['warp_mid'])), 'warp_tail': _num(r.get(C['warp_tail'])),
                # 단일값 인자
                'ingot': _num(r.get(C['ingot'])), 'wait': _num(r.get(C['wait'])), 'warm': _num(r.get(C['warm'])),
                # 프로파일 인자 (0~100pct)
                'frame': _profile_row(rd, ACT_FRAME, cl),
                'slurry': _profile_row(rd, ACT_SLURRY, cl),
                'wg_l': _profile_row(rd, ACT_WG_L, cl),
                'wg_r': _profile_row(rd, ACT_WG_R, cl),
            }
            points.append(pt)

        # 가장 최근 끝난 가공시간
        last_pt = points[-1]['process_time'] if points else ''

        acts[str(eqp)] = {
            'eqp': str(eqp),
            'points': points,
            'n_lots': len(points),
            'last_process_time': last_pt,
        }
    return acts


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or f in MISSING:
            return None
        return round(f, 3)
    except (ValueError, TypeError):
        return None


def load_recommend(csv_path):
    """추천 로딩 — 장비별 가장 최근 끝난 가공시간 1개만."""
    df = pd.read_csv(csv_path)
    eqp_col = 'eqp' if 'eqp' in df.columns else df.columns[0]
    has_pt = 'process_time' in df.columns
    cl = {c.lower(): c for c in df.columns}

    # 장비별로 모든 process_time 행 수집
    by_eqp = {}
    for _, r in df.iterrows():
        eqp = str(r.get(eqp_col, '?'))
        frame = _profile_row(r.to_dict(), 'rec_set_frame_temp_{p}pct', cl)
        slurry = _profile_row(r.to_dict(), 'rec_set_slurry_temp_{p}pct', cl)
        if all(v is None for v in frame) and all(v is None for v in slurry):
            continue
        pt_val = str(r.get('process_time', '')) if has_pt else ''
        rec = {
            'process_time': pt_val,
            'wire': str(r.get('latest_wire', '')),
            'waf': int(r['n_waf_used']) if 'n_waf_used' in r.index and pd.notna(r['n_waf_used']) else 0,
            'bow': _pred_bow(r),
            'rec_frame': [v if v is not None else 0 for v in frame],
            'rec_slurry': [v if v is not None else 0 for v in slurry],
            'frame_bow_rec': _num(r.get('frame_bow_with_recipe')),
            'slurry_bow_rec': _num(r.get('slurry_bow_with_recipe')),
        }
        by_eqp.setdefault(eqp, []).append(rec)
    return by_eqp


def _pred_bow(r):
    for c in ['frame_pred_bow', 'slurry_pred_bow', 'predicted_bow', 'target_bow']:
        if c in r.index and pd.notna(r[c]):
            return round(float(r[c]), 3)
    return 1.25


def merge(recs_by_eqp, acts):
    """추천(최근 가공시간 1개) + 실측(통합) 결합."""
    out = []
    all_eqps = set(recs_by_eqp) | set(acts)
    for eqp in sorted(all_eqps):
        a = acts.get(eqp)
        rec_list = recs_by_eqp.get(eqp, [])
        # 가장 최근 끝난 가공시간에 해당하는 추천 선택
        chosen = None
        if a and rec_list:
            last_pt = a['last_process_time']
            for rec in rec_list:
                if rec['process_time'] == last_pt:
                    chosen = rec
                    break
        if chosen is None and rec_list:
            chosen = rec_list[-1]  # 못 찾으면 마지막

        out.append({
            'eqp': eqp,
            'recommend': chosen,
            'actual': a,
        })
    return out
