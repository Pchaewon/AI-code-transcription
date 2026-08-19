# -*- coding: utf-8 -*-
"""
report_render.py — Plotly 동적 리포트 렌더링
─────────────────────────────────────────────
· 실측: 가공시간별 색 구분, 한 트렌드, hover 정보
· x축 4단계: 시간(mm/dd hh) → wire id → wire lifetime → blk_id
· Bow/Warp 4행 배치, 좌우 여백 최소화
· 추천: 가장 최근 끝난 가공시간 1개
"""
import json

TARGET_BOW = 1.25
SPEC_LO = 1.0
SPEC_HI = 1.5
PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 가공시간별 색상
PT_COLORS = {
    '13.3Hr': '#0f5c8c',   # 파랑
    '18.5Hr': '#d9772d',   # 주황
}
PT_COLOR_FALLBACK = ['#0f5c8c', '#d9772d', '#2e7d32', '#8e24aa', '#c62828']

# 고정 스케일 (요청 3번 유지)
SCALES = {
    'bow':   {'lo': -4, 'hi': 5,  'dtick': 1},
    'warp':  {'lo': 0,  'hi': 20, 'dtick': 4},
    'frame': {'lo': 27.5, 'hi': 31, 'dtick': 0.5},
    'slurry':{'lo': 19, 'hi': 30, 'dtick': 1},
    'guide': {'lo': -20, 'hi': 20, 'dtick': 5},
}


def pt_color_map(points):
    """등장하는 가공시간에 색 배정."""
    seen = []
    for p in points:
        pt = p.get('process_time', '')
        if pt and pt not in seen:
            seen.append(pt)
    cmap = {}
    fi = 0
    for pt in seen:
        if pt in PT_COLORS:
            cmap[pt] = PT_COLORS[pt]
        else:
            cmap[pt] = PT_COLOR_FALLBACK[fi % len(PT_COLOR_FALLBACK)]
            fi += 1
    return cmap


def build_html(records, out_path):
    data_json = json.dumps(records, ensure_ascii=False)
    html = TEMPLATE.replace('__DATA__', data_json) \
                   .replace('__PCTS__', json.dumps(PCTS)) \
                   .replace('__TARGET__', str(TARGET_BOW)) \
                   .replace('__SPECLO__', str(SPEC_LO)) \
                   .replace('__SPECHI__', str(SPEC_HI)) \
                   .replace('__SCALES__', json.dumps(SCALES)) \
                   .replace('__PTCOLORS__', json.dumps(PT_COLORS)) \
                   .replace('__NEQP__', str(len(records)))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return out_path


TEMPLATE = r'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>Wire Saw APC 추천 리포트</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<script>
// CDN 로드 실패 시 대체 CDN 시도
if(typeof Plotly==='undefined'){
  document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"><\/script>');
}
</script>
<style>
  :root{--ink:#1a2028;--faint:#7a8896;--line:#e3e8ec;--paper:#fff;--bg:#f4f6f8;
        --frame:#0f5c8c;--slurry:#2563a8;--rec:#d92d20;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:'Segoe UI','Malgun Gothic',sans-serif;font-size:13px;}
  .wrap{max-width:1600px;margin:0 auto;padding:20px;}
  h1{font-size:22px;margin:0 0 4px;}
  .sub{color:var(--faint);font-size:12px;margin-bottom:16px;}
  .summary{background:var(--paper);border:1px solid var(--line);border-radius:10px;
           padding:16px 18px;margin-bottom:20px;}
  .summary table{width:100%;border-collapse:collapse;font-size:12px;}
  .summary th{text-align:left;padding:8px 10px;border-bottom:2px solid var(--line);color:var(--faint);font-weight:600;}
  .summary td{padding:8px 10px;border-bottom:1px solid var(--line);}
  .sum-row{cursor:pointer;}
  .sum-row:hover{background:#f0f4f8;}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;
         font-family:'JetBrains Mono',monospace;}
  .b-ok{background:#e6f4ea;color:#1e7e34;}
  .b-out{background:#fdecea;color:#c62828;}
  .b-warn{background:#fff4e5;color:#e07800;}
  .b-none{background:#eef0f2;color:#888;}
  .eqp{background:var(--paper);border:1px solid var(--line);border-radius:12px;
       margin-bottom:24px;overflow:hidden;}
  .eqp-head{display:flex;align-items:center;gap:14px;padding:16px 20px;
            border-bottom:1px solid var(--line);background:linear-gradient(180deg,#fbfcfd,#fff);}
  .eqp-name{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;}
  .grow{flex:1;}
  .home-btn{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;color:#2d3a46;
            background:#eef0f2;border:1px solid var(--line);border-radius:6px;padding:5px 12px;cursor:pointer;}
  .home-btn:hover{background:#dde2e6;}
  .pt-badge{display:inline-block;background:#2d3a46;color:#fff;font-family:'JetBrains Mono',monospace;
            font-size:11px;font-weight:700;padding:3px 10px;border-radius:5px;}
  .sec{padding:14px 18px;}
  .sec-label{font-size:12px;font-weight:700;color:var(--faint);text-transform:uppercase;
             letter-spacing:.5px;margin:8px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line);}
  .rec-band{display:flex;gap:18px;align-items:center;padding:12px 16px;background:#f0f6fb;
            border-radius:8px;margin-bottom:12px;flex-wrap:wrap;}
  .rec-band .big{font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--frame);}
  .rec-lbl{font-size:11px;color:var(--faint);}
  .rec-bow{margin-left:8px;font-size:11px;color:var(--frame);font-family:'JetBrains Mono',monospace;}
  .plot{width:100%;margin:0 0 6px;}
  .plot-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .legend-note{font-size:11px;color:var(--faint);margin:4px 0 10px;}
  .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 3px 0 10px;vertical-align:middle;}
</style></head>
<body><div class="wrap">
  <h1>Wire Saw APC 추천 리포트</h1>
  <div class="sub">가공시간 통합 트렌드 · 최근 가공시간 추천 · 동적 그래프(hover)</div>
  <div id="summary" class="summary"></div>
  <div id="cards"></div>
</div>
<script>
const DATA=__DATA__;
const PCTS=__PCTS__;
const TARGET=__TARGET__, SPEC_LO=__SPECLO__, SPEC_HI=__SPECHI__;
const SCALES=__SCALES__;
const PT_COLORS=__PTCOLORS__;
const PT_FALLBACK=['#0f5c8c','#d9772d','#2e7d32','#8e24aa','#c62828'];

// 가공시간 색 배정
function ptColorMap(points){
  const seen=[]; points.forEach(p=>{const t=p.process_time; if(t&&!seen.includes(t))seen.push(t);});
  const m={}; let fi=0;
  seen.forEach(t=>{ if(PT_COLORS[t]) m[t]=PT_COLORS[t]; else m[t]=PT_FALLBACK[(fi++)%PT_FALLBACK.length]; });
  return m;
}

// x축 4단계 tick 라벨: 시간 / wire / lifetime / blk
function multiTick(points){
  return points.map(p=>{
    const d=p.date||'', w=p.wire||'', lf=(p.lifetime!==''&&p.lifetime!=null)?p.lifetime:'', b=p.blk||'';
    return `${d}<br>${w}<br>${lf}<br>${b}`;
  });
}

// 단일값 계열(bow/warp 등) → Plotly 트레이스 (가공시간별 색)
function scalarTraces(points, field, cmap){
  const byPt={};
  points.forEach((p,i)=>{
    const t=p.process_time||'';
    if(!byPt[t]) byPt[t]={x:[],y:[],cd:[]};
    byPt[t].x.push(i);
    byPt[t].y.push(p[field]);
    byPt[t].cd.push([p.date,p.wire,p.lifetime,p.blk,p.process_time]);
  });
  const traces=[];
  Object.keys(byPt).forEach(t=>{
    const g=byPt[t];
    traces.push({
      x:g.x, y:g.y, customdata:g.cd,
      mode:'markers+lines', type:'scatter', name:t||'—',
      marker:{color:cmap[t]||'#888',size:7},
      line:{color:cmap[t]||'#888',width:1.5},
      connectgaps:false,
      hovertemplate:'<b>%{customdata[4]}</b><br>날짜 %{customdata[0]}<br>wire %{customdata[1]}<br>lifetime %{customdata[2]}<br>blk %{customdata[3]}<br>값 <b>%{y}</b><extra></extra>',
    });
  });
  return traces;
}

// 공통 레이아웃 (여백 최소화 + x축 4단계 tick)
function baseLayout(points, scaleKey, title, extraShapes){
  const tickvals=points.map((_,i)=>i);
  const ticktext=multiTick(points);
  const sc=SCALES[scaleKey];
  const layout={
    title:{text:title,font:{size:13},x:0.01,xanchor:'left'},
    margin:{l:38,r:6,t:26,b:64},   // 여백 최소화
    height:230,
    xaxis:{tickvals:tickvals,ticktext:ticktext,tickfont:{size:8,family:'JetBrains Mono'},
           showgrid:false,zeroline:false,range:[-0.5,points.length-0.5]},
    yaxis:{showgrid:true,gridcolor:'#eef1f4',zeroline:false},
    showlegend:true,
    legend:{orientation:'h',x:0,y:1.14,font:{size:10}},
    plot_bgcolor:'#fff',paper_bgcolor:'#fff',
    shapes:extraShapes||[],
    hovermode:'closest',
  };
  if(sc){ layout.yaxis.range=[sc.lo,sc.hi]; layout.yaxis.dtick=sc.dtick; }
  return layout;
}

const CONFIG={displayModeBar:false,responsive:true};

// Bow/Warp 트렌드 (고정 스케일 + target/spec)
function drawBowWarp(divId, points, field, scaleKey, title){
  const cmap=ptColorMap(points);
  const traces=scalarTraces(points, field, cmap);
  const shapes=[];
  if(scaleKey==='bow'){
    // 스펙 밴드 + target
    shapes.push({type:'rect',xref:'paper',x0:0,x1:1,yref:'y',y0:SPEC_LO,y1:SPEC_HI,
                 fillcolor:'#000',opacity:0.04,line:{width:0}});
    shapes.push({type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:TARGET,y1:TARGET,
                 line:{color:'#1a1a1a',width:1,dash:'dash'}});
  }
  const layout=baseLayout(points, scaleKey, title, shapes);
  Plotly.newPlot(divId, traces, layout, CONFIG);
}

// 프로파일 계열(frame/slurry/guide) — 각 lot의 0~100pct를 하나의 선으로, 가공시간색
function drawProfile(divId, points, field, scaleKey, title, recProf){
  const cmap=ptColorMap(points);
  const traces=[];
  points.forEach((p,i)=>{
    const prof=p[field]; if(!prof) return;
    const t=p.process_time||'';
    traces.push({
      x:PCTS, y:prof,
      mode:'lines', type:'scatter', name:t||'—',
      line:{color:cmap[t]||'#888',width:1},opacity:0.55,
      showlegend:false,
      customdata:PCTS.map(pc=>[p.date,p.wire,p.lifetime,p.blk,t,pc]),
      hovertemplate:'<b>%{customdata[4]}</b><br>날짜 %{customdata[0]}<br>wire %{customdata[1]}<br>lifetime %{customdata[2]}<br>blk %{customdata[3]}<br>%{customdata[5]}pct: <b>%{y}</b><extra></extra>',
    });
  });
  // 추천선 (빨강)
  if(recProf && recProf.length){
    traces.push({x:PCTS,y:recProf,mode:'lines+markers',type:'scatter',name:'추천',
      line:{color:'#d92d20',width:2.2},marker:{color:'#d92d20',size:4},
      hovertemplate:'추천 %{x}pct: <b>%{y}</b><extra></extra>'});
  }
  const sc=SCALES[scaleKey];
  const layout={
    title:{text:title,font:{size:13},x:0.01,xanchor:'left'},
    margin:{l:38,r:6,t:26,b:32},height:220,
    xaxis:{title:{text:'pct',font:{size:9}},showgrid:false,zeroline:false,dtick:20,tickfont:{size:9}},
    yaxis:{showgrid:true,gridcolor:'#eef1f4',zeroline:false},
    showlegend:false,plot_bgcolor:'#fff',paper_bgcolor:'#fff',hovermode:'closest',
  };
  if(sc){ layout.yaxis.range=[sc.lo,sc.hi]; layout.yaxis.dtick=sc.dtick; }
  Plotly.newPlot(divId, traces, layout, CONFIG);
}

// 단일값 막대 (ingot/wait/warm) — 값 레이블
function drawBar(divId, points, field, title){
  const cmap=ptColorMap(points);
  const x=points.map((_,i)=>i), y=points.map(p=>p[field]);
  const colors=points.map(p=>cmap[p.process_time]||'#888');
  const cd=points.map(p=>[p.date,p.wire,p.lifetime,p.blk,p.process_time]);
  const trace={x:x,y:y,customdata:cd,type:'bar',marker:{color:colors},
    text:y.map(v=>v!=null?(Math.abs(v)>=100?v.toFixed(0):v.toFixed(1)):''),
    textposition:'outside',textfont:{size:8},
    hovertemplate:'<b>%{customdata[4]}</b><br>날짜 %{customdata[0]}<br>wire %{customdata[1]}<br>값 <b>%{y}</b><extra></extra>'};
  const layout=baseLayout(points,null,title,[]);
  layout.height=200;
  Plotly.newPlot(divId, [trace], layout, CONFIG);
}

// 요약 테이블
function summaryTable(records){
  const rows=records.map(d=>{
    const a=d.actual, rec=d.recommend;
    let lastBow=null, lastPt='';
    if(a&&a.points&&a.points.length){
      for(let i=a.points.length-1;i>=0;i--){ if(a.points[i].bow!=null){lastBow=a.points[i].bow;lastPt=a.points[i].process_time;break;} }
    }
    let st='none',stt='실제 없음';
    if(lastBow!=null){
      if(lastBow<SPEC_LO||lastBow>SPEC_HI){st='out';stt='스펙 이탈';}
      else if(lastBow<SPEC_LO+0.1||lastBow>SPEC_HI-0.1){st='warn';stt='주의';}
      else{st='ok';stt='정상';}
    }
    const recBow=rec?rec.bow:null;
    const recPt=rec?rec.process_time:(a?a.last_process_time:'');
    return {eqp:d.eqp,recPt:recPt,recBow:recBow,lastBow:lastBow,st:st,stt:stt,nlots:a?a.n_lots:0};
  });
  const body=rows.map(r=>`
    <tr class="sum-row" onclick="document.getElementById('eqp-${r.eqp}').scrollIntoView({behavior:'smooth',block:'start'})">
      <td style="font-family:'JetBrains Mono',monospace;font-weight:700">${r.eqp}</td>
      <td><span class="pt-badge">${r.recPt||'—'}</span></td>
      <td>${r.recBow!=null?r.recBow.toFixed(2):'—'}</td>
      <td>${r.lastBow!=null?r.lastBow.toFixed(3):'—'}</td>
      <td><span class="badge b-${r.st}">${r.stt}</span></td>
      <td>${r.nlots} lot</td>
    </tr>`).join('');
  return `<div style="font-weight:700;margin-bottom:10px">장비별 요약 <span style="color:var(--faint);font-weight:400;font-size:11px">· 행 클릭 시 상세 이동</span></div>
    <table><thead><tr><th>장비</th><th>최근 가공시간</th><th>추천 예상 BOW</th><th>실제 최근 BOW</th><th>상태</th><th>데이터</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

// 장비 카드
function card(d, idx){
  const a=d.actual, rec=d.recommend;
  const eqp=d.eqp;
  const pts=a?a.points:[];
  const cmap=ptColorMap(pts);
  // 가공시간 범례
  const legendHtml=Object.keys(cmap).map(t=>
    `<span class="swatch" style="background:${cmap[t]}"></span>${t}`).join('');

  let html=`<section class="eqp" id="eqp-${eqp}">
    <div class="eqp-head">
      <span class="eqp-name">${eqp}</span>
      ${rec?`<span class="pt-badge">최근 가공시간 ${rec.process_time}</span>`:''}
      <div class="grow"></div>
      <button class="home-btn" onclick="document.getElementById('summary').scrollIntoView({behavior:'smooth',block:'start'})">↑ 요약으로</button>
    </div>`;

  // 추천 섹션 (최근 가공시간 1개)
  if(rec){
    const lo=(rec.bow-0.25).toFixed(2), hi=(rec.bow+0.25).toFixed(2);
    html+=`<div class="sec">
      <div class="sec-label">추천 · 미래 lot (${rec.process_time})</div>
      <div class="rec-band">
        <div><div class="rec-lbl">예상 BOW</div><span class="big">${rec.bow.toFixed(2)}</span></div>
        <div><div class="rec-lbl">범위</div>${lo} ~ ${hi} µm</div>
        <div><div class="rec-lbl">최근 wire</div><code>${rec.wire}</code></div>
      </div>
      <div class="plot-row">
        <div><div class="sec-label">① Frame Temp 추천${rec.frame_bow_rec!=null?`<span class="rec-bow">추천시 예상 BOW ${rec.frame_bow_rec.toFixed(3)}</span>`:''}</div><div class="plot" id="p-${idx}-recframe"></div></div>
        <div><div class="sec-label">② Slurry Temp 추천${rec.slurry_bow_rec!=null?`<span class="rec-bow">추천시 예상 BOW ${rec.slurry_bow_rec.toFixed(3)}</span>`:''}</div><div class="plot" id="p-${idx}-recslurry"></div></div>
      </div>
    </div>`;
  }

  // 실측 섹션
  if(a && pts.length){
    html+=`<div class="sec">
      <div class="sec-label">실측 추이 · 최근 ${a.n_lots} lot · 가공시간 통합</div>
      <div class="legend-note">가공시간: ${legendHtml}</div>
      <div class="sec-label">Bow Trend (Total / Seed / Mid / Tail)</div>
      <div class="plot" id="p-${idx}-bow-total"></div>
      <div class="plot" id="p-${idx}-bow-seed"></div>
      <div class="plot" id="p-${idx}-bow-mid"></div>
      <div class="plot" id="p-${idx}-bow-tail"></div>
      <div class="sec-label" style="margin-top:16px">Warp Trend (Total / Seed / Mid / Tail)</div>
      <div class="plot" id="p-${idx}-warp-total"></div>
      <div class="plot" id="p-${idx}-warp-seed"></div>
      <div class="plot" id="p-${idx}-warp-mid"></div>
      <div class="plot" id="p-${idx}-warp-tail"></div>
      <div class="sec-label" style="margin-top:16px">X-Factor (실측 프로파일 · 추천 오버레이)</div>
      <div class="plot-row">
        <div class="plot" id="p-${idx}-frame"></div>
        <div class="plot" id="p-${idx}-slurry"></div>
      </div>
      <div class="plot-row">
        <div class="plot" id="p-${idx}-wgl"></div>
        <div class="plot" id="p-${idx}-wgr"></div>
      </div>
      <div class="sec-label" style="margin-top:16px">단일값 인자</div>
      <div class="plot" id="p-${idx}-ingot"></div>
      <div class="plot" id="p-${idx}-wait"></div>
      <div class="plot" id="p-${idx}-warm"></div>
    </div>`;
  } else {
    html+=`<div class="sec"><div style="padding:16px;color:var(--faint)">⚠ 실측 데이터 없음</div></div>`;
  }
  html+=`</section>`;
  return html;
}

// 렌더 실행
document.getElementById('summary').innerHTML=summaryTable(DATA);
document.getElementById('cards').innerHTML=DATA.map((d,i)=>card(d,i)).join('');

// Plotly 로드 확인
if(typeof Plotly==='undefined'){
  document.getElementById('cards').insertAdjacentHTML('afterbegin',
    '<div style="background:#fdecea;color:#c62828;padding:16px;border-radius:8px;margin-bottom:16px;font-weight:700">'+
    '⚠ Plotly 라이브러리를 불러오지 못했습니다. 인터넷 연결 또는 사내망 CDN 차단을 확인하세요.<br>'+
    '<span style="font-weight:400;font-size:12px">그래프 없이 요약/표만 표시됩니다.</span></div>');
} else {
  // 각 카드의 Plotly 그래프 그리기 (개별 try/catch — 하나 실패해도 나머지 진행)
  const safe=(fn)=>{try{fn();}catch(e){console.error('그래프 오류:',e);}};
  DATA.forEach((d,idx)=>{
    const a=d.actual, rec=d.recommend;
    if(rec){
      safe(()=>drawProfile(`p-${idx}-recframe`, a?a.points:[], null, 'frame', 'rec_frame', rec.rec_frame));
      safe(()=>drawProfile(`p-${idx}-recslurry`, a?a.points:[], null, 'slurry', 'rec_slurry', rec.rec_slurry));
    }
    if(a && a.points.length){
      const P=a.points;
      safe(()=>drawBowWarp(`p-${idx}-bow-total`, P, 'bow', 'bow', 'Bow Total'));
      safe(()=>drawBowWarp(`p-${idx}-bow-seed`, P, 'bow_seed', 'bow', 'Bow Seed'));
      safe(()=>drawBowWarp(`p-${idx}-bow-mid`, P, 'bow_mid', 'bow', 'Bow Mid'));
      safe(()=>drawBowWarp(`p-${idx}-bow-tail`, P, 'bow_tail', 'bow', 'Bow Tail'));
      safe(()=>drawBowWarp(`p-${idx}-warp-total`, P, 'warp', 'warp', 'Warp Total'));
      safe(()=>drawBowWarp(`p-${idx}-warp-seed`, P, 'warp_seed', 'warp', 'Warp Seed'));
      safe(()=>drawBowWarp(`p-${idx}-warp-mid`, P, 'warp_mid', 'warp', 'Warp Mid'));
      safe(()=>drawBowWarp(`p-${idx}-warp-tail`, P, 'warp_tail', 'warp', 'Warp Tail'));
      safe(()=>drawProfile(`p-${idx}-frame`, P, 'frame', 'frame', 'Frame Temp (실측)', rec?rec.rec_frame:null));
      safe(()=>drawProfile(`p-${idx}-slurry`, P, 'slurry', 'slurry', 'Slurry Temp (실측)', rec?rec.rec_slurry:null));
      safe(()=>drawProfile(`p-${idx}-wgl`, P, 'wg_l', 'guide', 'Guide Shift L', null));
      safe(()=>drawProfile(`p-${idx}-wgr`, P, 'wg_r', 'guide', 'Guide Shift R', null));
      safe(()=>drawBar(`p-${idx}-ingot`, P, 'ingot', 'Ingot Len'));
      safe(()=>drawBar(`p-${idx}-wait`, P, 'wait', 'Wait Time'));
      safe(()=>drawBar(`p-${idx}-warm`, P, 'warm', 'Warm Up Time'));
    }
  });
}
</script>
</body></html>'''
