
h = open('build_recommend_report.py', encoding='utf-8').read()

checks = {
    'process_time 분리 (pts)': "'pts'" in h or 'pts.append' in h,
    '5가지-50,60pct 판정': 'avg5060' in h,
    '5가지-고정스케일 Bow': 'fixLo:-4' in h,
    '5가지-home버튼': 'home-btn' in h,
    '5가지-막대레이블': 'lblTxt' in h,
    'X-Factor process_time 분리': 'buildActualHTML' in h,
    '추천 예상 BOW': 'bow_with_recipe' in h or 'frame_bow_rec' in h,
    'x축 날짜': 'widLabel' in h or 'wdate' in h,
}
for name, found in checks.items():
    print(f"{'O' if found else 'X'}  {name}")

