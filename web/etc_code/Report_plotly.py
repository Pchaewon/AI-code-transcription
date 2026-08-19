# -*- coding: utf-8 -*-
"""
report_plotly.py — Plotly 리포트 생성 진입점
사용: python report_plotly.py [recommend.csv] [field_store.csv] [out.html]
"""
import sys
from report_data import load_actuals, load_recommend, merge
from report_render import build_html


def main():
    args = sys.argv[1:]
    rec_csv = args[0] if len(args) > 0 else './recommend_future.csv'
    store_csv = args[1] if len(args) > 1 else './data/field_store.csv'
    out_html = args[2] if len(args) > 2 else './reports/recipe.html'

    print(f"[리포트] 추천: {rec_csv}")
    print(f"[리포트] 실측: {store_csv}")

    recs = load_recommend(rec_csv)
    acts = load_actuals(store_csv)
    records = merge(recs, acts)

    if not records:
        print("❌ 데이터 없음 — 리포트 생성 안 함")
        return

    build_html(records, out_html)
    n_act = sum(1 for r in records if r['actual'])
    n_rec = sum(1 for r in records if r['recommend'])
    print(f"✅ 리포트 생성: {out_html} ({len(records)}개 장비, 추천 {n_rec}, 실측 {n_act})")


if __name__ == '__main__':
    main()
