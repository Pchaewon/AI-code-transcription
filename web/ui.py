"""
사용:
  python build_recommend_report.py
  python build_recommend_report.py ./recommend_future.csv ./data/field_store.csv ./reports/recipe.html
"""
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ── 설정 ──
PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RANGE = 0.15
TARGET_BOW = 1.25
SPEC_LO, SPEC_HI = 1.0, 1.5      # 양품 스펙
PROCESS_TIME = '13.3Hr'
RECENT_N = 10                    # 실제 trend/프로파일에 쓸 최근 lot 수

REC_FRAME = 'rec_set_frame_temp_{p}pct'
REC_SLURRY = 'rec_set_slurry_temp_{p}pct'
ACT_FRAME = 'FRAME_IN_TEMP_{p}pct'        # 실제 입구온도 (trace 실측)
ACT_SLURRY = 'SLURRY_IN_TEMP_{p}pct'
ACT_WG_L = 'SHIFT_AMOUNT_WIREGUIDE_L_{p}pct'   # wire guide L 프로파일
ACT_WG_R = 'SHIFT_AMOUNT_WIREGUIDE_R_{p}pct'   # wire guide R 프로파일

STORE_CFG = {
    'eqp':  'eqp_nm_3200',
    'wire': 'fdc_new_wire_id',
    'date': 'date_3200',
    'bow':  'avg_bow_bf_total',
    'warp': 'avg_warp_bf_total',
    'bow_seed': 'avg_bow_bf_seed',
    'bow_mid':  'avg_bow_bf_mid',
    'bow_tail': 'avg_bow_bf_tail',
    'warp_seed':'avg_warp_bf_seed',
    'warp_mid': 'avg_warp_bf_mid',
    'warp_tail':'avg_warp_bf_tail',
    'ingot':'fdc_ingot_len',
    'wait': 'fdc_wait_time',
    'warm': 'fdc_warm_up_time',
}


def _profile(row, tmpl):
    out = []
    for p in PCTS:
        v = row.get(tmpl.format(p=p))
        out.append(None if pd.isna(v) else round(float(v), 2))
    return out


def _pred_bow(row):
    for c in ['frame_pred_bow', 'slurry_pred_bow']:
        if c in row.index and pd.notna(row[c]):
            return round(float(row[c]), 3)
    return TARGET_BOW


def load_recommend(csv_path):
    df = pd.read_csv(csv_path)
    eqp_col = 'eqp' if 'eqp' in df.columns else df.columns[0]
    has_pt = 'process_time' in df.columns
    recs = {}
    for _, r in df.iterrows():
        frame = _profile(r, REC_FRAME)
        slurry = _profile(r, REC_SLURRY)
        if all(v is None for v in frame) and all(v is None for v in slurry):
            print(f"  ⚠ {r.get(eqp_col)}: 추천 온도 없음 — 스킵")
            continue
        eqp = str(r.get(eqp_col, '?'))
        pt_val = str(r.get('process_time', '')) if has_pt else ''

        def _numval(col):
            v = r.get(col)
            return round(float(v), 3) if v is not None and pd.notna(v) else None

        rec_one = {
            'process_time': pt_val,
            'waf': int(r['n_waf_used']) if 'n_waf_used' in r.index
                   and pd.notna(r['n_waf_used']) else 0,
            'wire': str(r.get('latest_wire', '')),
            'bow': _pred_bow(r),
            'rec_frame': [v if v is not None else 0 for v in frame],
            'rec_slurry': [v if v is not None else 0 for v in slurry],
            # 추천 레시피를 x인자로 넣었을 때 예상 BOW (frame/slurry 각각)
            'frame_bow_rec': _numval('frame_bow_with_recipe'),
            'slurry_bow_rec': _numval('slurry_bow_with_recipe'),
        }

        # 장비별로 process_time 목록 누적
        if eqp not in recs:
            recs[eqp] = {'eqp': eqp, 'pts': []}
        recs[eqp]['pts'].append(rec_one)

    # 대표값(카드 상단·요약용): 첫 process_time 기준으로 평탄화
    for eqp, d in recs.items():
        first = d['pts'][0]
        d['waf'] = first['waf']
        d['wire'] = first['wire']
        d['bow'] = first['bow']
        d['rec_frame'] = first['rec_frame']
        d['rec_slurry'] = first['rec_slurry']
    return recs


def load_actuals(store_path):
    """field_store에서 장비별 최근 N wire의 실제값 추출."""
    if not os.path.exists(store_path):
        print(f"  ⚠ field_store 없음: {store_path} — 실제 영역 생략")
        return {}
    df = pd.read_csv(store_path)
    C = STORE_CFG
    if C['eqp'] not in df.columns:
        print(f"  ⚠ {C['eqp']} 컴럼 없음 — 실제 영역 생략")
        return {}

    # 대소문자 무관 컬럼 조회 (FRAME_IN_TEMP_0pct vs frame_in_temp_0pct 등)
    col_lookup = {c.lower(): c for c in df.columns}
    def realcol(name):
        return col_lookup.get(name.lower())
    MISSING_SENTINELS = {-1.0, -1, -999, -9999}  # placeholder 결측값

    acts = {}
    LOT = 'lot_id'
    has_pt = 'process_time' in df.columns
    # eqp + process_time 단위로 그룹 (process_time 없으면 eqp만)
    group_keys = [C['eqp']] + (['process_time'] if has_pt else [])
    for gk, g in df.groupby(group_keys):
        if has_pt:
            eqp, pt_val = gk if isinstance(gk, tuple) else (gk, '')
        else:
            eqp, pt_val = gk, ''
        acts_key = f"{eqp}|{pt_val}" if has_pt else str(eqp)

        if C['date'] in g.columns:
            g = g.sort_values(C['date'])

        wire_col = C['wire']
        has_lot = LOT in g.columns

        # 최근 N lot 선택 (lot 등장 순서 기준, 최근 것)
        if has_lot:
            lot_order = list(dict.fromkeys(g[LOT].astype(str).tolist()))
            recent_lots = set(lot_order[-RECENT_N:])
            # 최근 lot만 남기고, 그 lot이 속한 wire 순서 유지
            g = g[g[LOT].astype(str).isin(recent_lots)]
            recent_wires = list(dict.fromkeys(g[wire_col].astype(str).tolist()))
        else:
            # lot 없으면 기존처럼 wire 기준
            wire_order = list(dict.fromkeys(g[wire_col].astype(str).tolist()))
            recent_wires = wire_order[-RECENT_N:]

        def lot_profiles(sub, tmpl):
            """sub(한 wire의 행들)에서 lot별 프로파일 리스트 반환."""
            out = []
            def clean(v):
                if pd.isna(v) or float(v) in MISSING_SENTINELS:
                    return None
                return round(float(v), 2)
            if has_lot:
                for lot, lg in sub.groupby(LOT, sort=False):
                    prof = []
                    for p in PCTS:
                        rc = realcol(tmpl.format(p=p))
                        v = lg[rc].mean() if rc else None
                        prof.append(clean(v) if v is not None else None)
                    if not all(v is None for v in prof):
                        out.append({'lot': str(lot),
                                    'prof': [v if v is not None else 0 for v in prof]})
            else:
                for _, r in sub.iterrows():
                    prof = []
                    for p in PCTS:
                        rc = realcol(tmpl.format(p=p))
                        v = r.get(rc) if rc else None
                        prof.append(clean(v) if v is not None else None)
                    if not all(v is None for v in prof):
                        out.append({'lot': '',
                                    'prof': [v if v is not None else 0 for v in prof]})
            return out

        def lot_scalars(sub, name):
            """lot별 단일값 리스트."""
            key = C[name]
            out = []
            if has_lot:
                for lot, lg in sub.groupby(LOT, sort=False):
                    v = lg[key].mean() if key in lg.columns else None
                    out.append({'lot': str(lot),
                                'val': round(float(v), 1) if pd.notna(v) else None})
            else:
                for _, r in sub.iterrows():
                    v = r.get(key)
                    out.append({'lot': '',
                                'val': round(float(v), 1) if pd.notna(v) else None})
            return out

        # wire별로 lot 계층 구성
        wire_blocks = []      # [{wire, frame:[{lot,prof}], slurry:[...], ...}]
        bow_wires = []
        # total/seed/mid/tail 4개 계열 × bow/warp
        series = {k: [] for k in
                  ['bow', 'bow_seed', 'bow_mid', 'bow_tail',
                   'warp', 'warp_seed', 'warp_mid', 'warp_tail']}
        def lot_scalars_col(sub, col_nm):
            """지정 컬럼의 lot별 단일값 리스트."""
            out = []
            if has_lot and col_nm and col_nm in sub.columns:
                for lot, lg in sub.groupby(LOT, sort=False):
                    v = lg[col_nm].mean()
                    out.append({'lot': str(lot),
                                'val': round(float(v), 3) if pd.notna(v) else None})
            elif col_nm and col_nm in sub.columns:
                for _, r in sub.iterrows():
                    v = r.get(col_nm)
                    out.append({'lot': '',
                                'val': round(float(v), 3) if pd.notna(v) else None})
            return out

        for w in recent_wires:
            sub = g[g[wire_col].astype(str) == w]
            wire_blocks.append({
                'wire': w,
                'frame':  lot_profiles(sub, ACT_FRAME),
                'slurry': lot_profiles(sub, ACT_SLURRY),
                'wg_l':   lot_profiles(sub, ACT_WG_L),
                'wg_r':   lot_profiles(sub, ACT_WG_R),
                'ingot':  lot_scalars(sub, 'ingot'),
                'wait':   lot_scalars(sub, 'wait'),
                'warm':   lot_scalars(sub, 'warm'),
                # bow/warp 4계열 lot별 단일값 (wire>lot 트렌드용)
                'bow':       lot_scalars_col(sub, C.get('bow')),
                'bow_seed':  lot_scalars_col(sub, C.get('bow_seed')),
                'bow_mid':   lot_scalars_col(sub, C.get('bow_mid')),
                'bow_tail':  lot_scalars_col(sub, C.get('bow_tail')),
                'warp':      lot_scalars_col(sub, C.get('warp')),
                'warp_seed': lot_scalars_col(sub, C.get('warp_seed')),
                'warp_mid':  lot_scalars_col(sub, C.get('warp_mid')),
                'warp_tail': lot_scalars_col(sub, C.get('warp_tail')),
            })
            # (기존 wire 단위 series도 유지 — 호환용)
            for key in series:
                col_nm = C.get(key)
                if col_nm and col_nm in sub.columns:
                    v = sub[col_nm].mean()
                    series[key].append(round(float(v), 3) if pd.notna(v) else None)
                else:
                    series[key].append(None)
            bow_wires.append(w)

        # 실제 표시된 lot 총수 (blocks의 frame lot 합)
        n_lots = sum(len(b.get('frame', [])) for b in wire_blocks)
        acts[acts_key] = {
            'eqp': str(eqp),
            'process_time': pt_val,
            'wires': bow_wires,
            'n_lots': n_lots,
            'bow':  series['bow'],
            'warp': series['warp'],
            'bow_seed': series['bow_seed'], 'bow_mid': series['bow_mid'], 'bow_tail': series['bow_tail'],
            'warp_seed': series['warp_seed'], 'warp_mid': series['warp_mid'], 'warp_tail': series['warp_tail'],
            'blocks': wire_blocks,   # wire>lot 계층
            'has_lot': has_lot,
        }
    return acts


def merge(recs, acts):
    """추천의 각 process_time 블록에 해당 (eqp, process_time) 실측을 매칭."""
    out = []
    for eqp, r in recs.items():
        # 각 process_time 블록(pts)에 실측 붙이기
        for p in r.get('pts', []):
            pt = p.get('process_time', '')
            key = f"{eqp}|{pt}"
            # process_time별 실측, 없으면 pt 없는 키(eqp)도 시도
            p['actual'] = acts.get(key) or acts.get(str(eqp))
        # 카드 대표 실측 (요약/상단용): 첫 블록의 실측
        if r.get('pts'):
            r['actual'] = r['pts'][0].get('actual')
        else:
            r['actual'] = acts.get(str(eqp))
        out.append(r)
    return out


def render_html(records):
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    frame_start = records[0]['rec_frame'][0] if records else 28.0
    # 실제로 그려지는 lot 수 (장비별 최댓값)
    lot_counts = [r['actual'].get('n_lots', 0)
                  for r in records if r.get('actual')]
    actual_max_wire = max(lot_counts) if lot_counts else 0
    return (TEMPLATE
            .replace('__DATA__', json.dumps(records, ensure_ascii=False))
            .replace('__PCTS__', json.dumps(PCTS))
            .replace('__RANGE__', str(RANGE))
            .replace('__NEQP__', str(len(records)))
            .replace('__TARGET__', str(TARGET_BOW))
            .replace('__SPECLO__', str(SPEC_LO))
            .replace('__SPECHI__', str(SPEC_HI))
            .replace('__PTIME__', PROCESS_TIME)
            .replace('__FSTART__', str(frame_start))
            .replace('__RECENTN__', str(actual_max_wire))
            .replace('__STAMP__', stamp))


TEMPLATE = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wire Saw APC — 온도 Recipe 추천 리포트</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#1a1a1a; --ink-soft:#404040; --ink-faint:#808080;
    --paper:#ffffff; --panel:#f2f2f2; --panel-line:#dcdcdc; --line:#cccccc;
    --frame:#1a1a1a; --frame-soft:#eeeeee;
    --slurry:#1a1a1a; --slurry-soft:#eeeeee;
    --target:#1a1a1a; --spec:#666666; --actual:#4d4d4d;
    --low:#4d4d4d; --low-bg:#f0f0f0;
    --rec-bg:#ececec; --act-bg:#f5f5f5;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Noto Sans KR',system-ui,sans-serif;color:var(--ink);background:var(--panel);line-height:1.6;padding:0 0 80px;}
  code,.mono{font-family:'JetBrains Mono',monospace;}
  .wrap{max-width:1280px;margin:0 auto;padding:0 16px;}
  .toolbar{position:sticky;top:0;z-index:50;background:rgba(255,255,255,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--panel-line);padding:11px 0;}
  .toolbar .wrap{display:flex;align-items:center;justify-content:space-between;gap:16px;}
  .toolbar .t-title{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--ink-faint);letter-spacing:.06em;}
  .btn{font-family:'Noto Sans KR',sans-serif;font-size:13px;font-weight:700;background:var(--frame);color:#fff;border:none;padding:9px 18px;border-radius:7px;cursor:pointer;}
  .btn:hover{background:#0c4a70;}
  .masthead{background:var(--ink);color:#fff;padding:38px 0 30px;border-bottom:4px solid var(--frame);}
  .eyebrow{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.28em;text-transform:uppercase;color:#7fb8dc;font-weight:500;margin-bottom:13px;}
  .masthead h1{font-size:30px;font-weight:900;letter-spacing:-.01em;line-height:1.2;}
  .masthead .sub{margin-top:9px;color:#a9bccb;font-size:14.5px;font-weight:300;max-width:660px;}
  .meta-row{display:flex;flex-wrap:wrap;gap:26px;margin-top:24px;padding-top:20px;border-top:1px solid rgba(255,255,255,.14);}
  .meta-item .k{font-family:'JetBrains Mono',monospace;color:#6f8598;font-size:11px;letter-spacing:.12em;text-transform:uppercase;display:block;margin-bottom:4px;}
  .meta-item .v{color:#dbe6ee;font-weight:500;font-size:13.5px;}
  .summary{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0 4px;}
  /* 요약 테이블 */
  .summary-tbl-wrap{background:var(--paper);border:1px solid var(--panel-line);border-radius:11px;margin-top:22px;padding:18px 20px 16px;box-shadow:0 1px 2px rgba(18,32,46,.03);}
  .sum-title{font-size:15px;font-weight:700;margin-bottom:14px;display:flex;align-items:center;gap:12px;}
  .sum-hint{font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:400;color:var(--ink-faint);letter-spacing:.03em;}
  .summary-tbl{width:100%;border-collapse:collapse;font-size:13px;}
  .summary-tbl th{font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-faint);font-weight:500;text-align:left;padding:7px 10px;border-bottom:1.5px solid var(--panel-line);}
  .summary-tbl th:nth-child(n+2){text-align:right;}
  .sum-row{cursor:pointer;transition:background .12s;border-left:3px solid transparent;}
  .sum-row:hover{background:var(--panel);}
  .sum-row td{padding:9px 10px;border-bottom:1px solid #eef1f4;}
  .sum-eqp{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:14px;}
  .sum-num{text-align:right;font-family:'JetBrains Mono',monospace;}
  .sum-range{color:var(--ink-faint);font-size:12px;}
  .sum-status{text-align:right;}
  .sum-arrow{text-align:right;color:var(--ink-faint);font-size:18px;width:20px;}
  .sum-badge{font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:700;padding:3px 9px;border-radius:5px;letter-spacing:.03em;}
  /* 흑백 톤 상태 강조: 농도 + 좌측 바 */
  .badge-ok{background:#eef0f2;color:#5a6b7a;}
  .badge-warn{background:#dfe3e7;color:#2d3a46;border:1px solid #b8bfc6;}
  .badge-out{background:#2d3a46;color:#fff;}
  .badge-none{background:#f4f6f8;color:#a9b2bb;}
  .sum-out{border-left-color:#2d3a46;background:#fafbfc;}
  .sum-warn{border-left-color:#8a939c;}
  .sum-legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;padding-top:11px;border-top:1px solid var(--panel-line);font-size:11.5px;color:var(--ink-soft);}
  .sum-legend span{display:flex;align-items:center;gap:6px;}
  .sum-legend i{width:12px;height:12px;border-radius:3px;display:inline-block;}
  .sum-legend .lg-out{background:#2d3a46;}
  .sum-legend .lg-warn{background:#dfe3e7;border:1px solid #b8bfc6;}
  .sum-legend .lg-ok{background:#eef0f2;}
  .stat{background:var(--paper);border:1px solid var(--panel-line);border-radius:9px;padding:16px 18px;}
  .stat .sk{font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-faint);margin-bottom:7px;}
  .stat .sv{font-size:24px;font-weight:900;letter-spacing:-.01em;}
  .stat .su{font-size:12px;color:var(--ink-faint);font-weight:400;margin-left:3px;}
  .note{background:var(--low-bg);border-left:3px solid var(--low);padding:13px 18px;margin:22px 0 6px;border-radius:0 6px 6px 0;font-size:13.5px;color:#6b5d1f;}
  .note strong{font-weight:700;}
  .eqp{background:var(--paper);border:1px solid var(--panel-line);border-radius:11px;margin-top:22px;overflow:hidden;box-shadow:0 1px 2px rgba(18,32,46,.03);}
  .eqp-head{display:flex;align-items:center;gap:16px;padding:20px 26px;border-bottom:1px solid var(--panel-line);background:linear-gradient(180deg,#fbfcfd,#fff);}
  .eqp-name{font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;}
  .eqp-head .grow{flex:1;}
  .home-btn{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;color:#2d3a46;
    background:#eef0f2;border:1px solid var(--panel-line);border-radius:6px;padding:5px 12px;cursor:pointer;
    transition:background .15s;}
  .home-btn:hover{background:#dde2e6;}
  .badge{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.05em;padding:5px 11px;border-radius:6px;}
  .badge-waf{background:var(--low-bg);color:var(--low);border:1px solid #e6dcae;}
  .badge-time{background:var(--frame-soft);color:var(--frame);margin-left:8px;}
  .sec-label{display:flex;align-items:center;gap:10px;padding:12px 26px;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:700;}
  .sec-rec{background:var(--rec-bg);color:var(--frame);border-top:1px solid var(--panel-line);border-bottom:1px solid var(--panel-line);}
  .sec-act{background:var(--act-bg);color:#8a6d4a;border-top:1px solid var(--panel-line);border-bottom:1px solid var(--panel-line);}
  .sec-label .tag{font-size:9px;padding:2px 7px;border-radius:4px;color:#fff;letter-spacing:.06em;}
  .sec-rec .tag{background:var(--frame);}
  .sec-act .tag{background:#8a6d4a;}
  .bow-band{display:flex;align-items:center;gap:20px;padding:16px 26px;background:var(--frame-soft);flex-wrap:wrap;}
  /* process_time별 추천 블록 */
  .pt-block{border-bottom:1px solid var(--panel-line);}
  .pt-block:last-child{border-bottom:none;}
  .pt-block-full{border-bottom:3px solid var(--panel-line);padding-bottom:8px;margin-bottom:8px;}
  .pt-block-full:last-child{border-bottom:none;}
  .pt-badge{display:inline-block;background:#2d3a46;color:#fff;font-family:'JetBrains Mono',monospace;
    font-size:11px;font-weight:700;padding:3px 10px;border-radius:5px;letter-spacing:.3px;}
  .pt-badge-sm{display:inline-block;background:#eef0f2;color:#2d3a46;font-family:'JetBrains Mono',monospace;
    font-size:9.5px;font-weight:700;padding:2px 7px;border-radius:4px;}
  .rec-bow{margin-left:10px;font-size:11px;font-weight:400;color:var(--frame);font-family:'JetBrains Mono',monospace;}
  .rec-bow b{font-size:12.5px;color:#0f5c8c;}
  .bow-band .lbl{font-size:12.5px;color:var(--ink-soft);font-weight:500;}
  .bow-val{font-size:22px;font-weight:900;}
  .bow-range{font-family:'JetBrains Mono',monospace;font-size:15px;color:var(--frame);font-weight:700;}
  .bow-target{margin-left:auto;font-size:12px;color:var(--ink-faint);}
  .bow-target b{color:var(--target);font-weight:700;}
  .profiles{display:grid;grid-template-columns:1fr 1fr;gap:0;}
  @media(max-width:760px){.profiles{grid-template-columns:1fr;}}
  .profile{padding:20px 24px;}
  .profile:first-child{border-right:1px solid var(--panel-line);}
  @media(max-width:760px){.profile:first-child{border-right:none;border-bottom:1px solid var(--panel-line);}}
  .profile h3, .xf h3{font-size:14px;font-weight:700;display:flex;align-items:center;gap:9px;margin-bottom:4px;font-family:'Noto Sans KR',system-ui,sans-serif;}
  .profile h3 .sw, .xf h3 .sw{width:11px;height:11px;border-radius:3px;}
  .profile .unit, .xf .unit{font-size:11.5px;color:var(--ink-faint);font-family:'JetBrains Mono',monospace;margin-bottom:12px;}
  .xfactor{padding:4px 0;}
  .xf{padding:16px 24px;border-bottom:1px solid var(--panel-line);}
  .xf:last-child{border-bottom:none;}
  .chart{width:100%;height:150px;margin-bottom:6px;cursor:zoom-in;transition:height .2s ease;}
  .chart-tall{height:170px;}
  /* 클릭 인라인 확대 */
  .chart.zoomed{height:auto;min-height:420px;cursor:zoom-out;background:#fff;
    box-shadow:0 4px 24px rgba(18,32,46,.12);border:1px solid var(--panel-line);
    border-radius:8px;padding:8px;position:relative;z-index:5;}
  .chart-horizon.zoomed{min-height:480px;}
  .tbl-wrap{width:100%;overflow-x:auto;margin-top:8px;}
  .tbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:9.5px;table-layout:fixed;}
  .tbl th{background:var(--panel);color:var(--ink-faint);font-weight:500;padding:4px 1px;text-align:center;border-bottom:1px solid var(--panel-line);}
  .tbl td{padding:4px 1px;text-align:center;border-bottom:1px solid #eef1f4;color:var(--ink-soft);}
  .tbl td.v{font-weight:700;color:var(--ink);}
  .tbl th:first-child,.tbl td:first-child{width:24px;color:var(--ink-faint);}
  .trend-row{display:grid;grid-template-columns:1fr 1fr;gap:0;}
  @media(max-width:760px){.trend-row{grid-template-columns:1fr;}}
  /* Bow/Warp seed/mid/tail 4분할 (2열) */
  .quad-grid{display:grid;grid-template-columns:1fr 1fr;gap:0;}
  .quad-grid .qcell{padding:16px 20px;border-right:1px solid var(--panel-line);border-bottom:1px solid var(--panel-line);}
  .quad-grid .qcell:nth-child(2n){border-right:none;}
  .quad-grid .qcell:nth-last-child(-n+2){border-bottom:none;}
  .quad-grid .qcell h3{font-size:13px;font-weight:700;display:flex;align-items:center;gap:8px;margin-bottom:3px;font-family:'Noto Sans KR',system-ui,sans-serif;}
  .quad-grid .qcell h3 .sw{width:10px;height:10px;border-radius:3px;}
  .quad-grid .qcell .unit{font-size:10.5px;color:var(--ink-faint);font-family:'JetBrains Mono',monospace;margin-bottom:8px;}
  @media(max-width:760px){.quad-grid{grid-template-columns:1fr;}.quad-grid .qcell{border-right:none;}}
  .cap{font-size:11px;color:var(--ink-faint);margin-top:2px;}
  .cond-row{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border-top:1px solid var(--panel-line);}
  @media(max-width:620px){.cond-row{grid-template-columns:1fr;}}
  .cond{padding:16px 20px;border-right:1px solid var(--panel-line);}
  .cond:last-child{border-right:none;}
  .cond h4{font-size:12px;font-weight:700;color:var(--ink-soft);margin-bottom:8px;font-family:'JetBrains Mono',monospace;}
  .no-actual{padding:18px 26px;font-size:13px;color:var(--ink-faint);background:var(--act-bg);}
  .foot{max-width:1280px;margin:34px auto 0;padding:22px 16px 0;border-top:1px solid var(--line);font-size:11.5px;color:var(--ink-faint);display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;font-family:'JetBrains Mono',monospace;}
  @media(max-width:620px){.summary{grid-template-columns:repeat(2,1fr);}.masthead h1{font-size:24px;}.wrap{padding:0 18px;}}
  @media print{body{background:#fff;}.toolbar{display:none;}.eqp{box-shadow:none;break-inside:avoid;}.masthead,.bow-band,.badge,.tbl th,.stat,.sec-rec,.sec-act{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
</style>
</head>
<body>
<header class="masthead"><div class="wrap">
  <div class="eyebrow">Wire Saw APC · Temperature Recipe Recommendation</div>
  <h1>온도 Recipe 추천 리포트</h1>
  <p class="sub">위쪽 <b>추천</b>(미래·역산), 아래쪽 <b>실제 추이</b>(최근 wire·실측). 최근 상태를 보고 추천 수용을 판단합니다.</p>
  <div class="meta-row">
    <div class="meta-item"><span class="k">Process Time</span><span class="v">__PTIME__</span></div>
    <div class="meta-item"><span class="k">Target BOW</span><span class="v">__TARGET__ µm</span></div>
    <div class="meta-item"><span class="k">양품 스펙</span><span class="v">__SPECLO__ ~ __SPECHI__ µm</span></div>
    <div class="meta-item"><span class="k">생성 시각</span><span class="v">__STAMP__</span></div>
  </div>
</div></header>
<div class="wrap">
  <div class="summary">
    <div class="stat"><div class="sk">추천 장비</div><div class="sv">__NEQP__<span class="su">/ 10대</span></div></div>
    <div class="stat"><div class="sk">Target BOW</div><div class="sv">__TARGET__<span class="su">µm</span></div></div>
    <div class="stat"><div class="sk">예상 범위</div><div class="sv">±__RANGE__<span class="su">µm</span></div></div>
    <div class="stat"><div class="sk">실제 추이(최대)</div><div class="sv">__RECENTN__<span class="su">lot</span></div></div>
  </div>
  <div id="cards"></div>
</div>
<div class="foot"><span>WIRESAW_APC · DUAL_MODEL · SLSQP_INVERSE + FIELD_ACTUALS</span><span>GENERATED __STAMP__</span></div>
<script>
const PCTS=__PCTS__, DATA=__DATA__, RANGE=__RANGE__, TARGET=__TARGET__;
const SPEC_LO=__SPECLO__, SPEC_HI=__SPECHI__;

function lineChart(values,color,soft){
  const W=300,H=150,padL=34,padR=10,padT=14,padB=24;
  const mn=Math.min(...values),mx=Math.max(...values),sp=(mx-mn)||1;
  const lo=mn-sp*0.15,hi=mx+sp*0.15,rng=hi-lo;
  const x=i=>padL+(W-padL-padR)*(i/(values.length-1)),y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  let grid='';for(let g=0;g<=2;g++){const val=lo+rng*g/2,yy=y(val);
    grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
    grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(1)}</text>`;}
  let xl='';[0,5,10].forEach(i=>{if(i<values.length)xl+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${PCTS[i]}</text>`;});
  const pts=values.map((v,i)=>`${x(i)},${y(v)}`).join(' ');
  const area=`M${padL},${H-padB} L`+values.map((v,i)=>`${x(i)},${y(v)}`).join(' L')+` L${W-padR},${H-padB} Z`;
  const dots=values.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="2.5" fill="${color}"/>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}">${grid}<path d="${area}" fill="${soft}" opacity="0.6"/><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>${dots}${xl}</svg>`;
}

// 사진 방식: wire별 구획을 가로로 이어 각 구획 안에서 0->100pct 진행
// wire > lot > pct 3계층 프로파일
//  blocks: [{wire:'W1', lots:[[...11pct], [...]]}, ...]  (각 wire의 lot 프로파일들)
function horizonChart(blocks,color,recProf,opts){
  opts=opts||{};
  const W=760,H=220,padL=42,padR=12,padT=16,padB=58,wireGap=0.12,lotGap=0.12;
  let all=blocks.flatMap(b=>b.lots.flat());
  if(recProf && recProf.length) all=all.concat(recProf);  // 추천선도 범위에 포함
  if(!all.length)return '<div class="cap">데이터 없음</div>';
  let lo,hi;
  if(opts.fixLo!=null && opts.fixHi!=null){
    lo=opts.fixLo; hi=opts.fixHi;
  } else {
    const mn=Math.min(...all),mx=Math.max(...all),sp=(mx-mn)||1;
    lo=mn-sp*0.12; hi=mx+sp*0.12;
  }
  const rng=(hi-lo)||1;
  const nW=blocks.length, plotW=W-padL-padR, wireSlot=plotW/nW;
  const y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  // y grid: 고정 스케일이면 major/minor, 아니면 4분할
  let grid='';
  if(opts.major){
    if(opts.minor){
      for(let val=Math.ceil(lo/opts.minor)*opts.minor; val<=hi+1e-9; val+=opts.minor){
        const yy=y(val);
        grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#f0f2f4" stroke-width="0.6"/>`;
      }
    }
    for(let val=Math.ceil(lo/opts.major)*opts.major; val<=hi+1e-9; val+=opts.major){
      const yy=y(val);
      grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#dde2e6" stroke-width="1"/>`;
      grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${(+val.toFixed(2))}</text>`;
    }
  } else {
    for(let g=0;g<=3;g++){const val=lo+rng*g/3,yy=y(val);
      grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
      grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(2)}</text>`;}
  }
  let body='';
  let lastLotRegion=null;  // 최신 wire 최신 lot 구간 (추천선 그릴 위치)
  blocks.forEach((blk,wi)=>{
    const wireX=padL+wireSlot*wi;
    if(wi%2===1) body+=`<rect x="${wireX}" y="${padT}" width="${wireSlot}" height="${H-padT-padB}" fill="#f6f6f6"/>`;
    if(wi>0) body+=`<line x1="${wireX}" y1="${padT-4}" x2="${wireX}" y2="${H-padB}" stroke="#b8bfc6" stroke-width="1.6"/>`;
    const inner=wireSlot*(1-wireGap), wStart=wireX+wireSlot*wireGap/2;
    const nL=blk.lots.length, lotSlot=inner/nL;
    blk.lots.forEach((prof,li)=>{
      const lotX=wStart+lotSlot*li, lotInner=lotSlot*(1-lotGap), lStart=lotX+lotSlot*lotGap/2;
      if(li>0) body+=`<line x1="${lotX}" y1="${padT}" x2="${lotX}" y2="${H-padB}" stroke="#e5e8eb" stroke-width="0.8" stroke-dasharray="2 2"/>`;
      const xx=i=>lStart+lotInner*(i/(prof.length-1));
      const pts=prof.map((v,i)=>`${xx(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
      body+=`<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/>`;
      body+=prof.map((v,i)=>`<circle cx="${xx(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="1.3" fill="${color}"/>`).join('');
      // 최신 wire의 최신 lot 위치 기억
      if(wi===blocks.length-1 && li===blk.lots.length-1){
        lastLotRegion={lStart, lotInner, npts:prof.length};
      }
    });
    const wid=(blk.wire||'').toString(), cx=wireX+wireSlot/2;
    body+=`<text x="${cx}" y="${H-padB+16}" text-anchor="end" font-family="JetBrains Mono" font-size="8" fill="#7a8896" transform="rotate(-40 ${cx} ${H-padB+16})">${wid}</text>`;
  });
  // ── 추천선 오버레이 (빨강) — 최신 lot 구간 위에 ──
  if(recProf && recProf.length && lastLotRegion){
    const {lStart, lotInner, npts}=lastLotRegion;
    const n=recProf.length;
    const xx=i=>lStart+lotInner*(i/(n-1));
    const pts=recProf.map((v,i)=>`${xx(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    body+=`<polyline points="${pts}" fill="none" stroke="#d92d20" stroke-width="2.2" stroke-linejoin="round"/>`;
    body+=recProf.map((v,i)=>`<circle cx="${xx(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="1.8" fill="#d92d20"/>`).join('');
    // 추천 라벨
    body+=`<text x="${xx(n-1).toFixed(1)}" y="${(y(recProf[n-1])-6).toFixed(1)}" text-anchor="end" font-family="JetBrains Mono" font-size="8.5" font-weight="700" fill="#d92d20">추천</text>`;
  }
  const legendRec = (recProf && recProf.length) ?
    `<line x1="${W-padR-120}" y1="12" x2="${W-padR-104}" y2="12" stroke="#d92d20" stroke-width="2.2"/><text x="${W-padR-100}" y="15" font-family="JetBrains Mono" font-size="8.5" fill="#d92d20">추천 recipe</text>` : '';
  const xaxis=`<text x="${padL+plotW/2}" y="${H-4}" text-anchor="middle" font-family="JetBrains Mono" font-size="8.5" fill="#7a8896">Wire ID &gt; lot &gt; pct 0→100% (굵은선=wire, 얕은선=lot, 빨강=추천)</text>`;
  return `<svg class="chart chart-horizon" viewBox="0 0 ${W} ${H}">${grid}${body}${legendRec}${xaxis}</svg>`;
}

// 단일값 인자: wire > lot 계층, lot마다 막대 하나
//  blocks: [{wire:'W1', vals:[v1,v2,...]}, ...]
// wire>lot 점+선 트렌드 (각 lot=점 하나, 시간순, wire 경계 표시)
function lotTrendChart(blocks,color,opts){
  opts=opts||{};
  const W=760,H=200,padL=42,padR=12,padT=16,padB=58,wireGap=0.10;
  const all=blocks.flatMap(b=>b.vals).filter(v=>v!=null);
  if(!all.length)return '<div class="cap">데이터 없음</div>';
  let lo,hi;
  if(opts.fixLo!=null && opts.fixHi!=null){
    // 고정 스케일
    lo=opts.fixLo; hi=opts.fixHi;
  } else {
    let mn=Math.min(...all),mx=Math.max(...all);
    if(opts.target!=null){mn=Math.min(mn,opts.target);mx=Math.max(mx,opts.target);}
    if(opts.specLo!=null){mn=Math.min(mn,opts.specLo);mx=Math.max(mx,opts.specHi);}
    const sp=(mx-mn)||1; lo=mn-sp*0.15; hi=mx+sp*0.15;
  }
  const rng=(hi-lo)||1;
  const nW=blocks.length, plotW=W-padL-padR, wireSlot=plotW/nW;
  const y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  // grid: 고정 스케일이면 major/minor 눈금, 아니면 4분할
  let grid='';
  if(opts.major){
    // 보조축(minor, 얇은 실선) 먼저
    if(opts.minor){
      for(let val=Math.ceil(lo/opts.minor)*opts.minor; val<=hi+1e-9; val+=opts.minor){
        const yy=y(val);
        grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#f0f2f4" stroke-width="0.6"/>`;
      }
    }
    // 주축(major)
    for(let val=Math.ceil(lo/opts.major)*opts.major; val<=hi+1e-9; val+=opts.major){
      const yy=y(val);
      grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#dde2e6" stroke-width="1"/>`;
      grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${(+val.toFixed(2))}</text>`;
    }
  } else {
    for(let g=0;g<=3;g++){const val=lo+rng*g/3,yy=y(val);
      grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
      grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(2)}</text>`;}
  }
  // 스펙 밴드 + target
  let band='';
  if(opts.specLo!=null){band=`<rect x="${padL}" y="${y(opts.specHi)}" width="${W-padL-padR}" height="${y(opts.specLo)-y(opts.specHi)}" fill="#000" opacity="0.04"/>
    <line x1="${padL}" y1="${y(opts.specLo)}" x2="${W-padR}" y2="${y(opts.specLo)}" stroke="#666" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>
    <line x1="${padL}" y1="${y(opts.specHi)}" x2="${W-padR}" y2="${y(opts.specHi)}" stroke="#666" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>`;}
  let tline='';
  if(opts.target!=null){tline=`<line x1="${padL}" y1="${y(opts.target)}" x2="${W-padR}" y2="${y(opts.target)}" stroke="#1a1a1a" stroke-width="1.2" stroke-dasharray="6 3"/>
    <text x="${W-padR}" y="${y(opts.target)-4}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#1a1a1a">target ${opts.target}</text>`;}
  // 배경: wire 음영 + 경계 (band/tline보다 먼저 그려서 안 가리게)
  let bg='';
  blocks.forEach((blk,wi)=>{
    const wireX=padL+wireSlot*wi;
    if(wi%2===1) bg+=`<rect x="${wireX}" y="${padT}" width="${wireSlot}" height="${H-padT-padB}" fill="#f6f6f6"/>`;
    if(wi>0) bg+=`<line x1="${wireX}" y1="${padT-4}" x2="${wireX}" y2="${H-padB}" stroke="#b8bfc6" stroke-width="1.6"/>`;
  });
  // 전경: 각 lot 점·선·라벨
  let body='', allPts=[];
  blocks.forEach((blk,wi)=>{
    const wireX=padL+wireSlot*wi;
    const inner=wireSlot*(1-wireGap), wStart=wireX+wireSlot*wireGap/2;
    const nL=blk.vals.length, lotSlot=inner/Math.max(nL,1);
    blk.vals.forEach((v,li)=>{
      const cx=wStart+lotSlot*(li+0.5);
      if(v!=null){allPts.push([cx,y(v)]);}
    });
    const wid=(blk.wire||'').toString(), cx=wireX+wireSlot/2;
    body+=`<text x="${cx}" y="${H-padB+16}" text-anchor="end" font-family="JetBrains Mono" font-size="8" fill="#7a8896" transform="rotate(-40 ${cx} ${H-padB+16})">${wid}</text>`;
  });
  // 점 잇는 선 (시간순 전체)
  if(allPts.length>1){
    const line=allPts.map(p=>`${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
    body+=`<polyline points="${line}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/>`;
  }
  // 점
  body+=allPts.map(p=>`<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="2.8" fill="${color}"/>`).join('');
  const xaxis=`<text x="${padL+plotW/2}" y="${H-4}" text-anchor="middle" font-family="JetBrains Mono" font-size="8.5" fill="#7a8896">Wire ID &gt; lot (굵은선=wire, 각 점=lot, 시간순)</text>`;
  // 순서: 음영/경계(bg) → grid → 스펙밴드/target(band,tline) → 점·선(body)
  return `<svg class="chart chart-horizon" viewBox="0 0 ${W} ${H}">${bg}${grid}${band}${tline}${body}${xaxis}</svg>`;
}

function barSlotChart(blocks,color,unit){
  const W=760,H=200,padL=42,padR=12,padT=16,padB=58,wireGap=0.12,lotGap=0.25;
  const all=blocks.flatMap(b=>b.vals).filter(v=>v!=null);
  if(!all.length)return '<div class="cap">데이터 없음</div>';
  const mn=Math.min(...all),mx=Math.max(...all),sp=(mx-mn)||1;
  const lo=mn-sp*0.15,hi=mx+sp*0.15,rng=hi-lo;
  const nW=blocks.length, plotW=W-padL-padR, wireSlot=plotW/nW;
  const y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  let grid='';for(let g=0;g<=3;g++){const val=lo+rng*g/3,yy=y(val);
    grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
    grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(1)}</text>`;}
  let body='';
  blocks.forEach((blk,wi)=>{
    const wireX=padL+wireSlot*wi;
    if(wi%2===1) body+=`<rect x="${wireX}" y="${padT}" width="${wireSlot}" height="${H-padT-padB}" fill="#f6f6f6"/>`;
    if(wi>0) body+=`<line x1="${wireX}" y1="${padT-4}" x2="${wireX}" y2="${H-padB}" stroke="#b8bfc6" stroke-width="1.6"/>`;
    const inner=wireSlot*(1-wireGap), wStart=wireX+wireSlot*wireGap/2;
    const nL=blk.vals.length, lotSlot=inner/nL, barW=lotSlot*(1-lotGap);
    blk.vals.forEach((v,li)=>{
      const lotX=wStart+lotSlot*li+lotSlot*lotGap/2;
      if(v!=null){
        const yy=y(v), h=(H-padB)-yy;
        body+=`<rect x="${lotX.toFixed(1)}" y="${yy.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" fill="${color}" opacity="0.75" rx="1"/>`;
        // 막대 위에 값 레이블
        const lblX=lotX+barW/2, lblY=yy-3;
        const lblTxt=(Math.abs(v)>=100)?v.toFixed(0):v.toFixed(1);
        body+=`<text x="${lblX.toFixed(1)}" y="${lblY.toFixed(1)}" text-anchor="middle" font-family="JetBrains Mono" font-size="7.5" fill="#4a5560">${lblTxt}</text>`;
      }
    });
    const wid=(blk.wire||'').toString(), cx=wireX+wireSlot/2;
    body+=`<text x="${cx}" y="${H-padB+16}" text-anchor="end" font-family="JetBrains Mono" font-size="8" fill="#7a8896" transform="rotate(-40 ${cx} ${H-padB+16})">${wid}</text>`;
  });
  const xaxis=`<text x="${padL+plotW/2}" y="${H-4}" text-anchor="middle" font-family="JetBrains Mono" font-size="8.5" fill="#7a8896">Wire ID &gt; lot (굵은선=wire, 시간순) · ${unit}</text>`;
  return `<svg class="chart chart-horizon" viewBox="0 0 ${W} ${H}">${grid}${body}${xaxis}</svg>`;
}

// 단일 라인 트렌드 (계열 하나)
function trendChart(values,wires,opts){
  opts=opts||{};
  const W=460,H=180,padL=38,padR=12,padT=16,padB=42;
  const vals=values.filter(v=>v!=null);if(!vals.length)return '<div class="cap">데이터 없음</div>';
  let mn=Math.min(...vals),mx=Math.max(...vals);
  if(opts.target!=null){mn=Math.min(mn,opts.target);mx=Math.max(mx,opts.target);}
  if(opts.specLo!=null){mn=Math.min(mn,opts.specLo);mx=Math.max(mx,opts.specHi);}
  const sp=(mx-mn)||1,lo=mn-sp*0.12,hi=mx+sp*0.12,rng=hi-lo;
  const n=values.length;
  const x=i=>padL+(W-padL-padR)*(n>1?i/(n-1):0.5),y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  let grid='';for(let g=0;g<=3;g++){const val=lo+rng*g/3,yy=y(val);
    grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
    grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(2)}</text>`;}
  let band='';
  if(opts.specLo!=null){band=`<rect x="${padL}" y="${y(opts.specHi)}" width="${W-padL-padR}" height="${y(opts.specLo)-y(opts.specHi)}" fill="#000" opacity="0.04"/>
    <line x1="${padL}" y1="${y(opts.specLo)}" x2="${W-padR}" y2="${y(opts.specLo)}" stroke="#666" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>
    <line x1="${padL}" y1="${y(opts.specHi)}" x2="${W-padR}" y2="${y(opts.specHi)}" stroke="#666" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>`;}
  let tline='';
  if(opts.target!=null){tline=`<line x1="${padL}" y1="${y(opts.target)}" x2="${W-padR}" y2="${y(opts.target)}" stroke="#1a1a1a" stroke-width="1.2" stroke-dasharray="6 3"/>
    <text x="${W-padR}" y="${y(opts.target)-4}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#1a1a1a">target ${opts.target}</text>`;}
  let xl='';const step=Math.ceil(n/5);
  values.forEach((v,i)=>{if(i%step===0||i===n-1){const w=(wires[i]||'').slice(-4);
    xl+=`<text x="${x(i)}" y="${H-22}" text-anchor="middle" font-family="JetBrains Mono" font-size="8" fill="#9aa6b2">${w}</text>`;}});
  const color=opts.color||'#0f5c8c';
  let seg=[],segs=[];
  values.forEach((v,i)=>{if(v==null){if(seg.length){segs.push(seg);seg=[];}}else seg.push(`${x(i)},${y(v)}`);});
  if(seg.length)segs.push(seg);
  const line=segs.map(s=>`<polyline points="${s.join(' ')}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`).join('');
  const dots=values.map((v,i)=>v==null?'':`<circle cx="${x(i)}" cy="${y(v)}" r="2.6" fill="${color}"/>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" style="height:180px">${grid}${band}${tline}${line}${dots}${xl}<text x="${padL}" y="${H-6}" font-family="JetBrains Mono" font-size="8" fill="#c3ccd4">← wire (과거→최근) →</text></svg>`;
}

function tbl(values){
  const head='<tr><th>pct</th>'+PCTS.map(p=>`<th>${p}</th>`).join('')+'</tr>';
  const row='<tr><td>°C</td>'+values.map(v=>`<td class="v">${v.toFixed(2)}</td>`).join('')+'</tr>';
  return `<div class="tbl-wrap"><table class="tbl">${head}${row}</table></div>`;
}
function condTrend(label,arr,unit){
  const vals=arr.filter(v=>v!=null);
  const last=vals.length?vals[vals.length-1]:'—';
  const spark=(function(){
    if(!vals.length)return '';const W=120,H=34,mn=Math.min(...vals),mx=Math.max(...vals),sp=(mx-mn)||1;
    const x=i=>2+(W-4)*(i/Math.max(1,arr.length-1)),y=v=>2+(H-4)*(1-(v-mn)/sp);
    const pts=arr.map((v,i)=>v==null?null:`${x(i)},${y(v)}`).filter(Boolean).join(' ');
    return `<svg width="${W}" height="${H}"><polyline points="${pts}" fill="none" stroke="#5a6b7a" stroke-width="1.5"/></svg>`;
  })();
  return `<div class="cond"><h4>${label}</h4>${spark}<div class="cap">최근값 <b>${last}</b> ${unit}</div></div>`;
}

// 한 실측 세트(a)의 HTML 생성 (process_time 블록 안에서 재사용)
// recFrame/recSlurry: 그 블록의 추천 프로파일 (X-Factor에 빨간선 오버레이)
function buildActualHTML(a, recFrame, recSlurry){
  if(!a) return `<div class="no-actual">⚠ 이 조건의 field_store 실제 데이터가 없어 실제 영역을 생략합니다.</div>`;
  let actualHTML='';
  {
    // blocks(wire>lot)에서 인자별 구조 추출
    const B=a.blocks||[];
    const prof=(key)=>B.map(b=>({wire:b.wire, lots:(b[key]||[]).map(x=>x.prof)}))
                        .filter(b=>b.lots.length);
    const scal=(key)=>B.map(b=>({wire:b.wire, vals:(b[key]||[]).map(x=>x.val)}))
                        .filter(b=>b.vals.length);
    const frB=prof('frame'), slB=prof('slurry'), wlB=prof('wg_l'), wrB=prof('wg_r');
    const inB=scal('ingot'), waB=scal('wait'), wmB=scal('warm');
    // bow/warp 4계열 (wire>lot 트렌드용)
    const bowB=scal('bow'), bowSeedB=scal('bow_seed'), bowMidB=scal('bow_mid'), bowTailB=scal('bow_tail');
    const warpB=scal('warp'), warpSeedB=scal('warp_seed'), warpMidB=scal('warp_mid'), warpTailB=scal('warp_tail');
    const nW=a.wires.length;
    const nL=a.n_lots||0;

    actualHTML=`
    <div class="sec-label sec-act"><span class="tag">ACTUAL</span> 실제 Bow 추이 · 최근 ${nL} lot</div>
    <div class="quad-grid">
      <div class="qcell">
        <h3><span class="sw" style="background:var(--actual)"></span>Bow · total</h3>
        <div class="unit">avg_bow_bf_total · wire&gt;lot</div>
        ${lotTrendChart(bowB,'#0f5c8c',{target:TARGET,specLo:SPEC_LO,specHi:SPEC_HI,fixLo:-4,fixHi:5,major:1})}
      </div>
      <div class="qcell">
        <h3><span class="sw" style="background:var(--actual)"></span>Bow · seed</h3>
        <div class="unit">avg_bow_bf_seed · wire&gt;lot</div>
        ${lotTrendChart(bowSeedB,'#0f5c8c',{target:TARGET,specLo:SPEC_LO,specHi:SPEC_HI,fixLo:-4,fixHi:5,major:1})}
      </div>
      <div class="qcell">
        <h3><span class="sw" style="background:var(--actual)"></span>Bow · mid</h3>
        <div class="unit">avg_bow_bf_mid · wire&gt;lot</div>
        ${lotTrendChart(bowMidB,'#0f5c8c',{target:TARGET,specLo:SPEC_LO,specHi:SPEC_HI,fixLo:-4,fixHi:5,major:1})}
      </div>
      <div class="qcell">
        <h3><span class="sw" style="background:var(--actual)"></span>Bow · tail</h3>
        <div class="unit">avg_bow_bf_tail · wire&gt;lot</div>
        ${lotTrendChart(bowTailB,'#0f5c8c',{target:TARGET,specLo:SPEC_LO,specHi:SPEC_HI,fixLo:-4,fixHi:5,major:1})}
      </div>
    </div>
    <div class="sec-label sec-act" style="border-top:1px solid var(--panel-line)"><span class="tag">ACTUAL</span> 실제 Warp 추이 · 최근 ${nL} lot</div>
    <div class="quad-grid">
      <div class="qcell">
        <h3><span class="sw" style="background:var(--actual)"></span>Warp · total</h3>
        <div class="unit">avg_warp_bf_total · wire&gt;lot</div>
        ${lotTrendChart(warpB,'#b8531f',{fixLo:0,fixHi:20,major:4,minor:2})}
      </div>
      <div class="qcell">
        <h3><span class="sw" style="background:var(--actual)"></span>Warp · seed</h3>
        <div class="unit">avg_warp_bf_seed · wire&gt;lot</div>
        ${lotTrendChart(warpSeedB,'#b8531f',{fixLo:0,fixHi:20,major:4,minor:2})}
      </div>
      <div class="qcell">
        <h3><span class="sw" style="background:var(--actual)"></span>Warp · mid</h3>
        <div class="unit">avg_warp_bf_mid · wire&gt;lot</div>
        ${lotTrendChart(warpMidB,'#b8531f',{fixLo:0,fixHi:20,major:4,minor:2})}
      </div>
      <div class="qcell">
        <h3><span class="sw" style="background:var(--actual)"></span>Warp · tail</h3>
        <div class="unit">avg_warp_bf_tail · wire&gt;lot</div>
        ${lotTrendChart(warpTailB,'#b8531f',{fixLo:0,fixHi:20,major:4,minor:2})}
      </div>
    </div>
    <div class="sec-label sec-act" style="border-top:1px solid var(--panel-line);background:#ececec;color:#4d4d4d"><span class="tag" style="background:#4d4d4d">X-FACTOR</span> 실제 인자 · wire &gt; lot 계층 (pct 0→100%, 시간순)</div>
    <div class="xfactor">
      <div class="xf">
        <h3><span class="sw" style="background:var(--frame)"></span>Frame Temp</h3>
        <div class="unit">frame_in_temp · 실측 · 최근 ${nL} lot · wire&gt;lot&gt;pct</div>
        ${horizonChart(frB,'#0f5c8c',recFrame,{fixLo:27.5,fixHi:31,major:0.5})}
      </div>
      <div class="xf">
        <h3><span class="sw" style="background:var(--slurry)"></span>Slurry Temp</h3>
        <div class="unit">slurry_in_temp · 실측 · 최근 ${nL} lot · wire&gt;lot&gt;pct</div>
        ${horizonChart(slB,'#2563a8',recSlurry,{fixLo:19,fixHi:30,major:1})}
      </div>
      ${wlB.length?`<div class="xf">
        <h3><span class="sw" style="background:#5a6b7a"></span>Wire Guide L</h3>
        <div class="unit">shift_amount_wireguide_l · wire&gt;lot&gt;pct</div>
        ${horizonChart(wlB,'#5a6b7a',null,{fixLo:-20,fixHi:20,major:5})}
      </div>`:''}
      ${wrB.length?`<div class="xf">
        <h3><span class="sw" style="background:#8a6d4a"></span>Wire Guide R</h3>
        <div class="unit">shift_amount_wireguide_r · wire&gt;lot&gt;pct</div>
        ${horizonChart(wrB,'#8a6d4a',null,{fixLo:-20,fixHi:20,major:5})}
      </div>`:''}
      <div class="xf">
        <h3><span class="sw" style="background:#0f766e"></span>ingot_len</h3>
        <div class="unit">fdc_ingot_len · wire&gt;lot 단일값</div>
        ${barSlotChart(inB,'#0f766e','mm')}
      </div>
      <div class="xf">
        <h3><span class="sw" style="background:#0f766e"></span>wait_time</h3>
        <div class="unit">fdc_wait_time · wire&gt;lot 단일값</div>
        ${barSlotChart(waB,'#7a8896','')}
      </div>
      <div class="xf">
        <h3><span class="sw" style="background:#0f766e"></span>warm_up_time</h3>
        <div class="unit">fdc_warm_up_time · wire&gt;lot 단일값</div>
        ${barSlotChart(wmB,'#b8531f','')}
      </div>
    </div>`;
    return actualHTML;
  }
}

function card(d){
  return `<section class="eqp" id="eqp-${d.eqp}">
    <div class="eqp-head"><span class="eqp-name">${d.eqp}</span><div class="grow"></div>
      <span class="badge badge-waf">WAF ${d.waf}개</span><span class="badge badge-time">__PTIME__</span>
      <button class="home-btn" onclick="document.getElementById('summary-top').scrollIntoView({behavior:'smooth',block:'start'})">↑ 요약으로</button></div>

    ${(d.pts||[{process_time:'',bow:d.bow,wire:d.wire,rec_frame:d.rec_frame,rec_slurry:d.rec_slurry,actual:d.actual}]).map(p=>{
      const plo=(p.bow-RANGE).toFixed(2), phi=(p.bow+RANGE).toFixed(2);
      const ptLabel = p.process_time ? `<span class="pt-badge">${p.process_time}</span>` : '';
      return `
      <div class="pt-block-full">
        <div class="sec-label sec-rec"><span class="tag">RECOMMEND</span> 추천 · 미래 lot (역산) ${ptLabel}</div>
        <div class="bow-band"><span class="lbl">예상 BOW</span><span class="bow-val">${p.bow.toFixed(2)}</span>
          <span class="bow-range">${plo} ~ ${phi} µm</span>
          <span class="bow-target">Target <b>${TARGET}</b> · 최근 wire <code>${p.wire}</code></span></div>
        <div class="profiles">
          <div class="profile"><h3><span class="sw" style="background:var(--frame)"></span>① Frame Temp 추천${p.frame_bow_rec!=null?`<span class="rec-bow">추천 사용시 예상 BOW <b>${p.frame_bow_rec.toFixed(3)}</b></span>`:''}</h3>
            <div class="unit">rec_set_frame_temp · °C · 0→100pct</div>${lineChart(p.rec_frame,'#0f5c8c','#cfe2ef')}${tbl(p.rec_frame)}</div>
          <div class="profile"><h3><span class="sw" style="background:var(--slurry)"></span>② Slurry Temp 추천${p.slurry_bow_rec!=null?`<span class="rec-bow">추천 사용시 예상 BOW <b>${p.slurry_bow_rec.toFixed(3)}</b></span>`:''}</h3>
            <div class="unit">rec_set_slurry_temp · °C · 0→100pct</div>${lineChart(p.rec_slurry,'#b8531f','#f0d9c9')}${tbl(p.rec_slurry)}</div>
        </div>
        ${buildActualHTML(p.actual, p.rec_frame, p.rec_slurry)}
      </div>`;
    }).join('')}
  </section>`;
}
// 요약 테이블: 장비×process_time별 추천/실제 비교 + 스펙 이탈 강조 + 클릭 이동
function summaryTable(records){
  const rows = [];
  const avg=arr=>{const v=(arr||[]).filter(x=>x!=null&&!isNaN(x)); return v.length?v.reduce((a,b)=>a+b,0)/v.length:null;};
  // 50,60pct 구간만 평균 (PCTS에서 값 50,60의 위치)
  const avg5060=arr=>{
    if(!arr) return null;
    const idx=[PCTS.indexOf(50),PCTS.indexOf(60)].filter(i=>i>=0);
    const v=idx.map(i=>arr[i]).filter(x=>x!=null&&!isNaN(x));
    return v.length?v.reduce((a,b)=>a+b,0)/v.length:null;
  };
  records.forEach(d=>{
    // 실제 최근 BOW (장비 공통)
    let lastBow=null;
    if(d.actual && d.actual.bow){
      for(let i=d.actual.bow.length-1;i>=0;i--){
        if(d.actual.bow[i]!=null){lastBow=d.actual.bow[i]; break;}
      }
    }
    let status='ok', statusTxt='정상';
    if(lastBow!=null){
      if(lastBow < SPEC_LO || lastBow > SPEC_HI){status='out'; statusTxt='스펙 이탈';}
      else if(lastBow < SPEC_LO+0.1 || lastBow > SPEC_HI-0.1){status='warn'; statusTxt='주의';}
    } else {status='none'; statusTxt='실제 없음';}

    // process_time별로 행 생성
    const pts = d.pts || [{process_time:'', bow:d.bow, rec_frame:d.rec_frame, waf:d.waf, actual:d.actual}];
    pts.forEach(p=>{
      const lo=(p.bow-RANGE).toFixed(2), hi=(p.bow+RANGE).toFixed(2);
      // 추천 vs 실측 판정: Frame 50,60pct 구간 평균의 delta (전제: start 28도)
      let recVerdict='—', recStatus='none', recDiff=null;
      // 이 process_time의 실측 frame (최신 lot) 50,60pct 평균
      let actFrameAvg=null;
      const pa = p.actual || d.actual;
      if(pa && pa.blocks && pa.blocks.length){
        const lastBlk=pa.blocks[pa.blocks.length-1];
        const frameLots=lastBlk.frame||[];
        if(frameLots.length) actFrameAvg=avg5060(frameLots[frameLots.length-1].prof);
      }
      const recAvg=avg5060(p.rec_frame);
      if(actFrameAvg!=null && recAvg!=null){
        recDiff=Math.abs(recAvg-actFrameAvg);
        if(recDiff>=0.3){recVerdict='변경 요망'; recStatus='out';}
        else{recVerdict='트렌드 유지'; recStatus='ok';}
      }
      rows.push({eqp:d.eqp, pt:p.process_time||'—', bow:p.bow, lo, hi, waf:p.waf,
                 lastBow, status, statusTxt, recVerdict, recStatus, recDiff});
    });
  });
  const body = rows.map(r=>`
    <tr class="sum-row sum-${r.status}" onclick="document.getElementById('eqp-${r.eqp}').scrollIntoView({behavior:'smooth',block:'start'})">
      <td class="sum-eqp">${r.eqp}</td>
      <td class="sum-status"><span class="pt-badge-sm">${r.pt}</span></td>
      <td class="sum-num">${r.bow.toFixed(2)}</td>
      <td class="sum-num sum-range">${r.lo}~${r.hi}</td>
      <td class="sum-num">${r.waf}</td>
      <td class="sum-num">${r.lastBow!=null?r.lastBow.toFixed(3):'—'}</td>
      <td class="sum-status"><span class="sum-badge badge-${r.status}">${r.statusTxt}</span></td>
      <td class="sum-num sum-range">${r.recDiff!=null?r.recDiff.toFixed(2):'—'}</td>
      <td class="sum-status"><span class="sum-badge badge-${r.recStatus}">${r.recVerdict}</span></td>
      <td class="sum-arrow">›</td>
    </tr>`).join('');
  return `
  <div class="summary-tbl-wrap" id="summary-top">
    <div class="sum-title">장비별 요약 <span class="sum-hint">행 클릭 → 상세로 이동</span></div>
    <table class="summary-tbl">
      <thead><tr>
        <th>장비</th><th>가공시간</th><th>예상 BOW</th><th>예상 범위</th><th>WAF</th>
        <th>실제 최근 BOW</th><th>상태</th><th>추천 차이</th><th>추천 판정</th><th></th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table>
    <div class="sum-legend">
      <span><i class="lg-out"></i>스펙 이탈 (${SPEC_LO}~${SPEC_HI} 벗어남)</span>
      <span><i class="lg-warn"></i>주의 (경계 근처)</span>
      <span><i class="lg-ok"></i>정상</span>
      <span style="margin-left:16px"><i class="lg-out"></i>변경 요망 (추천-실측 평균차 ≥ 0.3)</span>
      <span><i class="lg-ok"></i>트렌드 유지 (< 0.3)</span>
    </div>
  </div>`;
}

document.getElementById('cards').innerHTML = summaryTable(DATA) + DATA.map(card).join('');

// 그래프 클릭 → 인라인 확대 토글 (다시 클릭하면 축소)
document.addEventListener('click', function(e){
  const svg = e.target.closest('.chart');
  if(!svg) return;
  // 요약 테이블 행 클릭 이동과 충돌 방지 (차트만)
  svg.classList.toggle('zoomed');
});
</script>
</body>
</html>'''


def main():
    args = sys.argv[1:]
    rec_csv   = args[0] if len(args) > 0 else './recommend_future.csv'
    store_csv = args[1] if len(args) > 1 else './data/field_store.csv'
    out_path  = args[2] if len(args) > 2 else './reports/recipe_report.html'

    if not os.path.exists(rec_csv):
        print(f"❌ 추천 CSV 없음: {rec_csv}"); sys.exit(1)

    print(f"[리포트] 추천: {rec_csv}")
    recs = load_recommend(rec_csv)
    if not recs:
        print("❌ 추천 0건 — 리포트 생성 안 함"); sys.exit(1)

    print(f"[리포트] 실제: {store_csv}")
    acts = load_actuals(store_csv)
    records = merge(recs, acts)

    html = render_html(records)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ 리포트 생성: {out_path} ({len(records)}개 장비)")
    for r in records:
        has_act = '실제O' if r.get('actual') else '실제X'
        print(f"   · {r['eqp']}: WAF {r['waf']}개, 예측 BOW {r['bow']}, {has_act}")


if __name__ == '__main__':
    main()
