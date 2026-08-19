# -*- coding: utf-8 -*-
"""
build_recommend_report.py  (추천 + 실제 통합판)
─────────────────────────────────────────
두 소스를 읽어 장비별 리포트 HTML 생성.

  recommend_future.csv → ①②③ 추천 (rec_ 온도, pred_bow)   [미래·역산]
  field_store.csv      → 실제 영역                          [과거·실측]
      · Bow/Warp Trend (최근 N wire, Total)
      · X-Factor: 실제 frame/slurry 온도 프로파일 (최근 N wire 겹침)
      · X-Factor: 단일값 조건 (ingot/wait/warmup) 최근 N wire 추세

  두 소스는 eqp(장비)로 매칭. field_store 없으면 추천만 표시.

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
    'lifetime': 'WIREGUIDE_LIFE_TIME',
    'blk': 'BLK_NO',
    'pt': 'process_time',
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
    # eqp 단위로만 그룹 (가공시간 통합 — process_time은 lot별 속성으로 유지)
    group_keys = [C['eqp']]
    for gk, g in df.groupby(group_keys):
        eqp = gk[0] if isinstance(gk, tuple) else gk
        pt_val = ''  # 통합이므로 대표 pt 없음
        acts_key = str(eqp)

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

        def lot_strs_col(sub, col_nm, as_int=False):
            """지정 컬럼의 lot별 대표 문자열 리스트 (lot 순서).
            as_int=True면 숫자를 정수 문자열로 (250.0 → '250')."""
            def fmt(v):
                if pd.isna(v):
                    return ''
                if as_int:
                    try:
                        return str(int(round(float(v))))
                    except (ValueError, TypeError):
                        return str(v)
                return str(v)
            out = []
            real = None
            if col_nm:
                low = {c.lower(): c for c in sub.columns}
                real = low.get(col_nm.lower())
            if has_lot and real:
                for lot, lg in sub.groupby(LOT, sort=False):
                    out.append(fmt(lg[real].iloc[0]))
            elif real:
                for _, r in sub.iterrows():
                    out.append(fmt(r.get(real)))
            return out

        def wire_date(sub):
            """wire의 대표 날짜(첫 행)를 YYMMDD HH로."""
            dc = C.get('date')
            if not dc or dc not in sub.columns or len(sub) == 0:
                return ''
            raw = str(sub[dc].iloc[0])
            digits = ''.join(ch for ch in raw if ch.isdigit())
            if len(digits) >= 10:
                yy, mm, dd = digits[2:4], digits[4:6], digits[6:8]
                hh = digits[8:10]
                return f"{yy}{mm}{dd} {hh}"
            return raw[:11]

        def fmt_date_val(raw):
            """단일 날짜값을 YYMMDD HH로."""
            digits = ''.join(ch for ch in str(raw) if ch.isdigit())
            if len(digits) >= 10:
                return f"{digits[2:4]}{digits[4:6]}{digits[6:8]} {digits[8:10]}"
            return str(raw)[:11]

        def lot_dates(sub):
            """lot별 날짜 리스트 (YYMMDD HH, lot 순서)."""
            dc = C.get('date')
            if not dc or dc not in sub.columns:
                return []
            out = []
            if has_lot:
                for lot, lg in sub.groupby(LOT, sort=False):
                    out.append(fmt_date_val(lg[dc].iloc[0]))
            else:
                for _, r in sub.iterrows():
                    out.append(fmt_date_val(r.get(dc)))
            return out

        for w in recent_wires:
            sub = g[g[wire_col].astype(str) == w]
            wire_blocks.append({
                'wire': w,
                'date': wire_date(sub),
                'dates': lot_dates(sub),
                'lifetimes': lot_strs_col(sub, C.get('lifetime'), as_int=True),
                'blks':      lot_strs_col(sub, C.get('blk')),
                'pts':       lot_strs_col(sub, C.get('pt')),
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
        # 마지막(최근) 가공시간: 마지막 wire block의 마지막 lot pt
        last_pt = ''
        for b in reversed(wire_blocks):
            pts_list = b.get('pts', [])
            if pts_list:
                last_pt = str(pts_list[-1])
                break
        acts[acts_key] = {
            'eqp': str(eqp),
            'process_time': pt_val,
            'last_process_time': last_pt,
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
    """가공시간 통합 실측 + 마지막 끝난 가공시간의 추천만 선택."""
    out = []
    for eqp, r in recs.items():
        a = acts.get(str(eqp))
        r['actual'] = a
        # 마지막 끝난 가공시간
        last_pt = a['last_process_time'] if a else ''
        # 그 가공시간에 해당하는 추천 1개 선택
        chosen = None
        for p in r.get('pts', []):
            if str(p.get('process_time', '')) == last_pt:
                chosen = p
                break
        if chosen is None and r.get('pts'):
            chosen = r['pts'][-1]  # 못 찾으면 마지막 pts
        # 대표 추천값을 카드 상단용으로 평탄화
        if chosen:
            r['chosen_pt'] = chosen.get('process_time', '')
            r['bow'] = chosen.get('bow', r.get('bow'))
            r['wire'] = chosen.get('wire', r.get('wire'))
            r['rec_frame'] = chosen.get('rec_frame')
            r['rec_slurry'] = chosen.get('rec_slurry')
            r['frame_bow_rec'] = chosen.get('frame_bow_rec')
            r['slurry_bow_rec'] = chosen.get('slurry_bow_rec')
            r['waf'] = chosen.get('waf', r.get('waf'))
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
  .profile{padding:20px 24px;min-width:0;}
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
  .quad-grid{display:grid;grid-template-columns:1fr;gap:0;}
  .quad-grid .qcell{padding:16px 20px;border-right:none;border-bottom:1px solid var(--panel-line);}
  .quad-grid .qcell:last-child{border-bottom:none;}
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
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<script>
if(typeof Plotly==='undefined'){document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"><\/script>');}
</script>
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

// ─── Plotly 차트 큐 ───
// 차트 함수는 <div>를 리턴하고 그릴 정보를 큐에 쌓음. 카드 렌더 후 flush.
let __chartSeq=0;
const __chartQueue=[];
function __newChartDiv(h){
  const id='plt_'+(__chartSeq++);
  __chartQueue.push({id, h});
  return `<div id="${id}" class="chart-plotly" style="width:100%;height:${h}px"></div>`;
}
// traces+layout을 큐에 넣고 div 리턴 (Plotly 그리기용)
function __newChartDivFull(traces,layout,h){
  const id='plt_'+(__chartSeq++);
  __chartQueue.push({id, traces, layout});
  return `<div id="${id}" class="chart-plotly" style="width:100%;height:${(h||230)}px"></div>`;
}
const __PT_COLORS={'13.3Hr':'#d9772d','18.5Hr':'#0f5c8c'};
const __PT_FALLBACK=['#0f5c8c','#d9772d','#2e7d32','#8e24aa','#c62828'];
function __ptColor(pt,idx){ return __PT_COLORS[pt]||__PT_FALLBACK[idx%__PT_FALLBACK.length]; }
const __PLOT_CONFIG={displayModeBar:false,responsive:true};
// 공통 레이아웃
function __layout(h,extra){
  return Object.assign({
    margin:{l:40,r:8,t:8,b:70},height:h,
    plot_bgcolor:'#fff',paper_bgcolor:'#fff',
    showlegend:false,hovermode:'closest',
    xaxis:{showgrid:false,zeroline:false},
    yaxis:{showgrid:true,gridcolor:'#eef1f4',zeroline:false},
  },extra||{});
}


function lineChart(values,color,soft,opts){
  opts=opts||{};
  if(!values||!values.length) return '<div class="cap">데이터 없음</div>';
  const trace={
    x:PCTS, y:values, customdata:PCTS,
    mode:'lines+markers', type:'scatter',
    line:{color:color,width:2.2}, marker:{color:color,size:4},
    fill:'tozeroy', fillcolor:soft?(soft+'99'):'rgba(0,0,0,0.05)',
    hovertemplate:'%{x}pct: <b>%{y}</b><extra></extra>',
  };
  const yax={showgrid:true,gridcolor:'#eef1f4',zeroline:false,tickfont:{size:12.5}};
  if(opts.fixLo!=null && opts.fixHi!=null){
    yax.range=[opts.fixLo,opts.fixHi];
    if(opts.major) yax.dtick=opts.major;
    // 고정 스케일(0 미포함)에서 tozeroy fill은 범위 밖을 채우므로 제거
    delete trace.fill; delete trace.fillcolor;
  }
  const layout=__layout(180,{
    margin:{l:36,r:8,t:8,b:28},
    xaxis:{dtick:20,tickfont:{size:12.5,family:'JetBrains Mono'},showgrid:false,zeroline:false,title:{text:'pct',font:{size:12.5}}},
    yaxis:yax,
  });
  return __newChartDivFull([trace],layout,180);
}

// X-Factor 프로파일 (Plotly) — 각 lot의 0~100pct를 가로로 이어붙임, 4층 x축
function horizonChart(blocks,color,recProf,opts){
  opts=opts||{};
  // 각 lot을 x축에 이어붙임. lot li의 pct는 x = lotIndex + pct/100 (0~1 폭)
  const traces=[];
  let lotIndex=0;
  const wireStart=[];    // wire별 시작 lotIndex
  const lotMeta=[];      // 각 lot의 {wi, blk, life, pt, date, wire, xCenter}
  let lastLot=null;
  blocks.forEach((blk,wi)=>{
    wireStart.push(lotIndex);
    const lots=blk.lots||[];
    lots.forEach((prof,li)=>{
      const x0=lotIndex;
      const xs=prof.map((_,i)=>x0 + (PCTS[i]/100)*0.86 + 0.07); // lot 폭 안에서 pct
      const pt=(blk.pts&&blk.pts[li])||'';
      const c=pt?__ptColor(pt,0):color;
      const bk=(blk.blks&&blk.blks[li])||'';
      const lf=(blk.lifetimes&&blk.lifetimes[li])||'';
      traces.push({
        x:xs, y:prof, mode:'lines+markers', type:'scatter',
        line:{color:c,width:1.3}, marker:{color:c,size:3.5}, opacity:0.7, showlegend:false,
        customdata:PCTS.map(pc=>[(blk.dates&&blk.dates[li])||blk.date||'',blk.wire||'',lf,bk,pt,pc]),
        hovertemplate:'%{customdata[4]}<br>날짜 %{customdata[0]}<br>wire %{customdata[1]}<br>WG LT %{customdata[2]}<br>blk %{customdata[3]}<br>%{customdata[5]}pct: <b>%{y}</b><extra></extra>',
      });
      lotMeta.push({wi, blk:bk, life:lf, date:((blk.dates&&blk.dates[li])||blk.date||''), xCenter:x0+0.5});
      lastLot={x0};
      lotIndex++;
    });
  });
  if(!traces.length) return '<div class="cap">데이터 없음</div>';

  // 추천선 (최신 lot 구간 위, 빨강)
  if(recProf && recProf.length && lastLot){
    const xs=recProf.map((_,i)=>lastLot.x0 + (PCTS[i]/100)*0.86 + 0.07);
    traces.push({
      x:xs, y:recProf, mode:'lines+markers', type:'scatter', name:'추천',
      line:{color:'#d92d20',width:2.2}, marker:{color:'#d92d20',size:3}, showlegend:true,
      hovertemplate:'추천 %{customdata}pct: <b>%{y}</b><extra></extra>', customdata:PCTS,
    });
  }

  // 스케일
  let yrange=null,dtick=null;
  if(opts.fixLo!=null && opts.fixHi!=null){ yrange=[opts.fixLo,opts.fixHi]; dtick=opts.major||null; }

  // x축 tick: 점(lot)마다 blk/life
  const tickvals=lotMeta.map(m=>m.xCenter);
  const ticktext=lotMeta.map(m=>`${m.date||''}<br>${m.blk||''}<br>${m.life?('WG LT '+m.life):''}`);

  // 구간 브래킷 + wire/날짜
  const shapes=[], annos=[];
  const WIRE_BR_Y=-0.78, WIRE_TX_Y=-0.90, DATE_BR_Y=-0.52, DATE_TX_Y=-0.62;
  function bracket(x0,x1,yb){
    shapes.push({type:'line',xref:'x',yref:'paper',x0:x0,x1:x0,y0:yb+0.03,y1:yb,line:{color:'#b8bfc6',width:1}});
    shapes.push({type:'line',xref:'x',yref:'paper',x0:x0,x1:x1,y0:yb,y1:yb,line:{color:'#b8bfc6',width:1}});
    shapes.push({type:'line',xref:'x',yref:'paper',x0:x1,x1:x1,y0:yb,y1:yb+0.03,line:{color:'#b8bfc6',width:1}});
  }
  blocks.forEach((blk,wi)=>{
    const start=wireStart[wi];
    const cnt=(blk.lots||[]).length;
    if(!cnt) return;
    const end=start+cnt-1;
    const mid=(start+end)/2+0.5;
    bracket(start+0.05, end+0.95, WIRE_BR_Y);
    const w=(blk.wire||'').toString();
    const wShort=w;  // 전체 표시
    annos.push({x:mid,y:WIRE_TX_Y,xref:'x',yref:'paper',text:wShort,showarrow:false,font:{size:12.5,family:'JetBrains Mono',color:'#2d3a46'}});
  });
  // 스펙/target (Bow류 아니면 없음)
  if(opts.specLo!=null){
    shapes.push({type:'rect',xref:'paper',x0:0,x1:1,y0:opts.specLo,y1:opts.specHi,fillcolor:'#000',opacity:0.04,line:{width:0}});
  }
  if(opts.target!=null){
    shapes.push({type:'line',xref:'paper',x0:0,x1:1,y0:opts.target,y1:opts.target,line:{color:'#1a1a1a',width:1,dash:'dash'}});
  }

  const yax={showgrid:true,gridcolor:'#eef1f4',zeroline:false,tickfont:{size:12.5}};
  if(yrange){yax.range=yrange;} if(dtick){yax.dtick=dtick;}
  const layout=__layout(370,{
    margin:{l:44,r:8,t:8,b:200},
    xaxis:{tickvals:tickvals,ticktext:ticktext,tickfont:{size:12.5,family:'JetBrains Mono'},
           showgrid:false,zeroline:false,range:[-0.1,lotIndex+0.1]},
    yaxis:yax, shapes:shapes, annotations:annos,
    showlegend:(recProf&&recProf.length)?true:false,
    legend:{orientation:'h',x:0,y:1.10,font:{size:12.5}},
  });
  return __newChartDivFull(traces,layout,370);
}

// wire>lot 트렌드 (Plotly) — 각 lot=점, 4중첩 x축, 고정스케일, target/스펙
function lotTrendChart(blocks,color,opts){
  opts=opts||{};
  // blocks: [{wire, date, vals:[...], lifetimes?:[...], blks?:[...], pts?:[...]}]
  // 각 lot을 순서대로 평탄화
  const xs=[], ys=[], labels=[], colors=[], cd=[];
  let idx=0;
  const wireStart=[];  // wire 경계 위치
  blocks.forEach((blk,wi)=>{
    wireStart.push(idx);
    const vals=blk.vals||[];
    vals.forEach((v,li)=>{
      xs.push(idx);
      ys.push(v);
      const lt=(blk.lifetimes&&blk.lifetimes[li]!=null)?blk.lifetimes[li]:'';
      const bk=(blk.blks&&blk.blks[li]!=null)?blk.blks[li]:'';
      const pt=(blk.pts&&blk.pts[li]!=null)?blk.pts[li]:'';
      cd.push([(blk.dates&&blk.dates[li])||blk.date||'', blk.wire||'', lt, bk, pt]);
      colors.push(pt?__ptColor(pt,0):color);
      idx++;
    });
  });
  if(!xs.length) return '<div class="cap">데이터 없음</div>';

  // ── x축 4층 구조 ──
  // 점마다: blk / life (tick 라벨 2줄)
  // 구간 브래킷: wire id (annotation)
  // 구간 브래킷: 날짜 (annotation)
  const tickvals=xs.slice(), ticktext=[];
  xs.forEach((x,i)=>{
    const dt=cd[i][0]||'';                                   // 날짜 YYMMDD HH
    const bk=cd[i][3]||'';                                    // blk
    const lt=cd[i][2]!=null&&cd[i][2]!==''?('WG LT '+cd[i][2]):'';  // WG LT
    ticktext.push(`${dt}<br>${bk}<br>${lt}`);   // 점마다 날짜/blk/WG LT
  });

  // 스케일
  let yrange=null, dtick=null;
  if(opts.fixLo!=null && opts.fixHi!=null){ yrange=[opts.fixLo,opts.fixHi]; dtick=opts.major||null; }

  // 선도 가공시간 색으로, 이어지게: 연속 구간을 색별 선분으로 그림
  const traces=[];
  // 1) 연결선 — 인접 두 점마다 선분, 색은 뒤 점(도착)의 가공시간
  for(let i=0;i<xs.length-1;i++){
    const pt=cd[i+1][4]||'_';
    const c=(pt&&pt!=='_')?__ptColor(pt,0):color;
    traces.push({
      x:[xs[i],xs[i+1]], y:[ys[i],ys[i+1]], mode:'lines', type:'scatter',
      line:{color:c,width:1.8}, showlegend:false, hoverinfo:'skip',
    });
  }
  // 2) 가공시간별 마커 (색 구분, legend용)
  const byPt={};
  xs.forEach((x,i)=>{
    const pt=cd[i][4]||'_';
    if(!byPt[pt]) byPt[pt]={x:[],y:[],cd:[]};
    byPt[pt].x.push(x); byPt[pt].y.push(ys[i]); byPt[pt].cd.push(cd[i]);
  });
  Object.keys(byPt).forEach(pt=>{
    const g=byPt[pt];
    const c=(pt&&pt!=='_')?__ptColor(pt,0):color;
    traces.push({
      x:g.x, y:g.y, customdata:g.cd, mode:'markers', type:'scatter',
      name:(pt&&pt!=='_')?pt:'', marker:{color:c,size:7},
      hovertemplate:'%{customdata[4]}<br>날짜 %{customdata[0]}<br>wire %{customdata[1]}<br>WG LT %{customdata[2]}<br>blk %{customdata[3]}<br>값 <b>%{y}</b><extra></extra>',
    });
  });

  const shapes=[];
  // 스펙 밴드 + target
  if(opts.specLo!=null){
    shapes.push({type:'rect',xref:'paper',x0:0,x1:1,y0:opts.specLo,y1:opts.specHi,
                 fillcolor:'#000',opacity:0.04,line:{width:0}});
  }
  if(opts.target!=null){
    shapes.push({type:'line',xref:'paper',x0:0,x1:1,y0:opts.target,y1:opts.target,
                 line:{color:'#1a1a1a',width:1,dash:'dash'}});
  }

  // ── 구간 브래킷 + 라벨 (annotation/shape, x는 data 좌표, y는 paper 아래쪽) ──
  const annos=[];
  // 브래킷 y 위치 (plot 아래 paper 좌표: 음수)
  const WIRE_BR_Y=-0.78, WIRE_TX_Y=-0.90, DATE_BR_Y=-0.52, DATE_TX_Y=-0.62;
  function bracket(x0,x1,yb){
    // ㄴ자 브래킷 (shape, xref data, yref paper)
    shapes.push({type:'line',xref:'x',yref:'paper',x0:x0,x1:x0,y0:yb+0.03,y1:yb,line:{color:'#b8bfc6',width:1}});
    shapes.push({type:'line',xref:'x',yref:'paper',x0:x0,x1:x1,y0:yb,y1:yb,line:{color:'#b8bfc6',width:1}});
    shapes.push({type:'line',xref:'x',yref:'paper',x0:x1,x1:x1,y0:yb,y1:yb+0.03,line:{color:'#b8bfc6',width:1}});
  }
  blocks.forEach((blk,wi)=>{
    const start=wireStart[wi];
    const cnt=(blk.vals||[]).length;
    const end=start+cnt-1;
    const mid=(start+end)/2;
    // wire 브래킷 + id
    bracket(start-0.35, end+0.35, WIRE_BR_Y);
    const w=(blk.wire||'').toString();
    const wShort=w;  // 전체 표시
    annos.push({x:mid,y:WIRE_TX_Y,xref:'x',yref:'paper',text:wShort,showarrow:false,
                font:{size:12.5,family:'JetBrains Mono',color:'#2d3a46'}});
  });

  const yax={showgrid:true,gridcolor:'#eef1f4',zeroline:false,tickfont:{size:12.5}};
  if(yrange){yax.range=yrange;} if(dtick){yax.dtick=dtick;}
  const layout=__layout(370,{
    margin:{l:44,r:8,t:8,b:200},   // 4층 라벨 공간
    xaxis:{tickvals:tickvals,ticktext:ticktext,tickfont:{size:12.5,family:'JetBrains Mono'},
           showgrid:false,zeroline:false,range:[-0.6,xs[xs.length-1]+0.6]},
    yaxis:yax, shapes:shapes, annotations:annos,
    showlegend:Object.keys(byPt).length>1,
    legend:{orientation:'h',x:0,y:1.10,font:{size:12.5}},
  });
  return __newChartDivFull(traces,layout,370);
}

function barSlotChart(blocks,color,unit){
  // 각 lot=막대 하나, 4층 x축(점마다 blk/life, 구간 wire/날짜)
  const xs=[], ys=[], colors=[], cd=[], texts=[];
  let idx=0; const wireStart=[];
  blocks.forEach((blk,wi)=>{
    wireStart.push(idx);
    (blk.vals||[]).forEach((v,li)=>{
      xs.push(idx); ys.push(v);
      const pt=(blk.pts&&blk.pts[li])||'';
      colors.push(pt?__ptColor(pt,0):color);
      const bk=(blk.blks&&blk.blks[li])||'', lf=(blk.lifetimes&&blk.lifetimes[li])||'';
      cd.push([(blk.dates&&blk.dates[li])||blk.date||'',blk.wire||'',lf,bk,pt]);
      texts.push(v!=null?((Math.abs(v)>=100)?v.toFixed(0):v.toFixed(1)):'');
      idx++;
    });
  });
  if(!xs.length) return '<div class="cap">데이터 없음</div>';

  const trace={
    x:xs, y:ys, type:'bar', marker:{color:colors}, customdata:cd,
    text:texts, textposition:'outside', textfont:{size:12.5,family:'JetBrains Mono'},
    hovertemplate:'%{customdata[4]}<br>날짜 %{customdata[0]}<br>wire %{customdata[1]}<br>WG LT %{customdata[2]}<br>blk %{customdata[3]}<br>값 <b>%{y}</b><extra></extra>',
  };

  // x축 tick: 점마다 날짜/blk/WG LT
  const tickvals=xs.slice(), ticktext=cd.map(c=>`${c[0]||''}<br>${c[3]||''}<br>${c[2]?('WG LT '+c[2]):''}`);
  // 구간 브래킷
  const shapes=[], annos=[];
  const WIRE_BR_Y=-0.78, WIRE_TX_Y=-0.90, DATE_BR_Y=-0.52, DATE_TX_Y=-0.62;
  function bracket(x0,x1,yb){
    shapes.push({type:'line',xref:'x',yref:'paper',x0:x0,x1:x0,y0:yb+0.03,y1:yb,line:{color:'#b8bfc6',width:1}});
    shapes.push({type:'line',xref:'x',yref:'paper',x0:x0,x1:x1,y0:yb,y1:yb,line:{color:'#b8bfc6',width:1}});
    shapes.push({type:'line',xref:'x',yref:'paper',x0:x1,x1:x1,y0:yb,y1:yb+0.03,line:{color:'#b8bfc6',width:1}});
  }
  blocks.forEach((blk,wi)=>{
    const start=wireStart[wi], cnt=(blk.vals||[]).length;
    if(!cnt) return;
    const end=start+cnt-1, mid=(start+end)/2;
    bracket(start-0.4,end+0.4,WIRE_BR_Y);
    const w=(blk.wire||'').toString(), wShort=w;
    annos.push({x:mid,y:WIRE_TX_Y,xref:'x',yref:'paper',text:wShort,showarrow:false,font:{size:12.5,family:'JetBrains Mono',color:'#2d3a46'}});
  });

  const layout=__layout(360,{
    margin:{l:44,r:8,t:14,b:200},
    xaxis:{tickvals:tickvals,ticktext:ticktext,tickfont:{size:12.5,family:'JetBrains Mono'},
           showgrid:false,zeroline:false,range:[-0.7,xs[xs.length-1]+0.7]},
    yaxis:{showgrid:true,gridcolor:'#eef1f4',zeroline:false,tickfont:{size:12.5}},
    shapes:shapes, annotations:annos, showlegend:false,
  });
  return __newChartDivFull([trace],layout,360);
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
    const prof=(key)=>B.map(b=>({wire:b.wire, date:b.date, dates:b.dates, lifetimes:b.lifetimes, blks:b.blks, pts:b.pts, lots:(b[key]||[]).map(x=>x.prof)}))
                        .filter(b=>b.lots.length);
    const scal=(key)=>B.map(b=>({wire:b.wire, date:b.date, dates:b.dates, lifetimes:b.lifetimes, blks:b.blks, pts:b.pts, vals:(b[key]||[]).map(x=>x.val)}))
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
  const bow = (d.bow!=null?d.bow:TARGET);
  const plo=(bow-RANGE).toFixed(2), phi=(bow+RANGE).toFixed(2);
  const ptLabel = d.chosen_pt ? `<span class="pt-badge">최근 가공시간 ${d.chosen_pt}</span>` : '';
  const recFrame = d.rec_frame, recSlurry = d.rec_slurry;
  const hasRec = recFrame && recFrame.length;
  return `<section class="eqp" id="eqp-${d.eqp}">
    <div class="eqp-head"><span class="eqp-name">${d.eqp}</span>${ptLabel}<div class="grow"></div>
      <span class="badge badge-waf">WAF ${d.waf||0}개</span>
      <button class="home-btn" onclick="document.getElementById('summary-top').scrollIntoView({behavior:'smooth',block:'start'})">↑ 요약으로</button></div>

    <div class="pt-block-full">
      ${hasRec?`
      <div class="sec-label sec-rec"><span class="tag">RECOMMEND</span> 추천 · 미래 lot (역산) ${d.chosen_pt?('· '+d.chosen_pt):''}</div>
      <div class="bow-band"><span class="lbl">예상 BOW</span><span class="bow-val">${bow.toFixed(2)}</span>
        <span class="bow-range">${plo} ~ ${phi} µm</span>
        <span class="bow-target">Target <b>${TARGET}</b> · 최근 wire <code>${d.wire||''}</code></span></div>
      <div class="profiles">
        <div class="profile"><h3><span class="sw" style="background:var(--frame)"></span>① Frame Temp 추천${d.frame_bow_rec!=null?`<span class="rec-bow">추천 사용시 예상 BOW <b>${d.frame_bow_rec.toFixed(3)}</b></span>`:''}</h3>
          <div class="unit">rec_set_frame_temp · °C · 0→100pct</div>${lineChart(recFrame,'#0f5c8c','#cfe2ef',{fixLo:27.5,fixHi:31,major:0.5})}${tbl(recFrame)}</div>
        <div class="profile"><h3><span class="sw" style="background:var(--slurry)"></span>② Slurry Temp 추천${d.slurry_bow_rec!=null?`<span class="rec-bow">추천 사용시 예상 BOW <b>${d.slurry_bow_rec.toFixed(3)}</b></span>`:''}</h3>
          <div class="unit">rec_set_slurry_temp · °C · 0→100pct</div>${lineChart(recSlurry,'#b8531f','#f0d9c9',{fixLo:19,fixHi:30,major:1})}${tbl(recSlurry)}</div>
      </div>`:''}
      ${buildActualHTML(d.actual, recFrame, recSlurry)}
    </div>
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

    // 장비당 1행 (마지막 가공시간 추천 기준)
    const bow = (d.bow!=null?d.bow:TARGET);
    const lo=(bow-RANGE).toFixed(2), hi=(bow+RANGE).toFixed(2);
    // 추천 판정: Frame 50,60pct 구간 평균 delta (통합 실측 최신 lot 기준)
    let recVerdict='—', recStatus='none', recDiff=null;
    let actFrameAvg=null;
    if(d.actual && d.actual.blocks && d.actual.blocks.length){
      const lastBlk=d.actual.blocks[d.actual.blocks.length-1];
      const frameLots=lastBlk.frame||[];
      if(frameLots.length) actFrameAvg=avg5060(frameLots[frameLots.length-1].prof);
    }
    const recAvg=avg5060(d.rec_frame);
    if(actFrameAvg!=null && recAvg!=null){
      recDiff=Math.abs(recAvg-actFrameAvg);
      if(recDiff>=0.3){recVerdict='변경 요망'; recStatus='out';}
      else{recVerdict='트렌드 유지'; recStatus='ok';}
    }
    rows.push({eqp:d.eqp, bow, lo, hi, waf:d.waf||0,
               lastBow, status, statusTxt, recVerdict, recStatus, recDiff});
  });
  const body = rows.map(r=>`
    <tr class="sum-row sum-${r.status}" onclick="document.getElementById('eqp-${r.eqp}').scrollIntoView({behavior:'smooth',block:'start'})">
      <td class="sum-eqp">${r.eqp}</td>
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
        <th>장비</th><th>예상 BOW</th><th>예상 범위</th><th>WAF</th>
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

// ─── Plotly 차트 큐 flush ───
(function(){
  function drawAll(){
    if(typeof Plotly==='undefined'){
      document.getElementById('cards').insertAdjacentHTML('afterbegin',
        '<div style="background:#fdecea;color:#c62828;padding:14px;border-radius:8px;margin-bottom:14px;font-weight:700">⚠ Plotly를 불러오지 못했습니다. 인터넷/사내망 CDN을 확인하세요.</div>');
      return;
    }
    __chartQueue.forEach(item=>{
      const el=document.getElementById(item.id);
      if(!el) return;
      try{
        if(item.traces){ Plotly.newPlot(item.id, item.traces, item.layout, __PLOT_CONFIG); }
      }catch(e){ console.error('차트 오류', item.id, e); }
    });
    // 그리드 셀 폭에 맞게 재조정
    setTimeout(()=>{
      __chartQueue.forEach(item=>{
        const el=document.getElementById(item.id);
        if(el && el.data){ try{ Plotly.Plots.resize(el); }catch(e){} }
      });
    }, 60);
    // 창 크기 변경 시 재조정
    window.addEventListener('resize', ()=>{
      __chartQueue.forEach(item=>{
        const el=document.getElementById(item.id);
        if(el && el.data){ try{ Plotly.Plots.resize(el); }catch(e){} }
      });
    });
  }
  // Plotly 로드 대기 (CDN fallback 고려)
  if(typeof Plotly!=='undefined'){ drawAll(); }
  else { let tries=0; const t=setInterval(()=>{ if(typeof Plotly!=='undefined'||tries++>40){clearInterval(t); drawAll();} },100); }
})();

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
