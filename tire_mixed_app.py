"""
타이어 혼재 적재 시뮬레이터
규격·개수 N종 입력 → 구역/층별 × 평치/벌집 4-way 최적화 + 3D 비교
"""

import streamlit as st
import math, re, itertools
from itertools import combinations
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ── 상수 ─────────────────────────────────────────────────────────────────────
CONT_W, CONT_L, CONT_H = 2352, 12032, 2698
N_SEG   = 8
PALETTE = ['#1D4ED8','#D97706','#16A34A','#DC2626',
           '#7C3AED','#0891B2','#BE185D','#059669']

# ═════════════════════════════════════════════════════════════════════════════
# 1. 파싱
# ═════════════════════════════════════════════════════════════════════════════
def parse_tire_size(s):
    m = re.search(r'(\d+)[^\d]+(\d+)[^\d]+(\d+)', s)
    if not m: return None, None
    w, a, r = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return round((w*a/100*2)+(r*25.4)), w

# ═════════════════════════════════════════════════════════════════════════════
# 2. 구역 배열 (Zone) ─ 컨테이너 길이 방향으로 규격별 구역 할당
# ═════════════════════════════════════════════════════════════════════════════
def pack_zone_flat(od, h, y0, avail_y, max_n):
    """구역 내 평치: 전체 폭×높이 사용, 길이 방향으로 열 적재"""
    fw = CONT_W // od; fh = CONT_H // h
    if fw==0 or fh==0: return [],0,0
    per_col = fw * fh
    cols_avail = avail_y // od
    cols_use   = min(math.ceil(max_n/per_col), cols_avail)
    if cols_use==0: return [],0,0
    packed = min(max_n, cols_use*per_col)
    r=od/2; pos=[]
    for cy in range(cols_use):
        for lz in range(fh):
            for cx in range(fw):
                pos.append((cx*od+r, y0+cy*od+r, lz*h, h, 'z'))
    pos = pos[:packed]
    len_used = math.ceil(packed/per_col)*od
    return pos, packed, len_used

def pack_zone_honeycomb(od, h, y0, avail_y, max_n):
    """구역 내 벌집: y행 우선 반복 → 길이 방향 최소 구역만 사용"""
    STEP = math.sqrt(3)/2*od
    n = max(1, 1+int((CONT_W-od)/STEP))
    wu = od+(n-1)*STEP
    if wu>CONT_W: n-=1; wu=od+(n-1)*STEP
    if n==0: return [],0,0
    hc = CONT_H // h
    if hc==0: return [],0,0
    # 짝수열(오프셋 없음) / 홀수열(od/2 오프셋) 가능 행 수
    m_e = int(avail_y // od)
    m_o = int((avail_y - od*0.5)/od) if avail_y > od*0.5 else 0
    m_o = max(0, m_o)
    if m_e==0 and m_o==0: return [],0,0
    r = od/2; pos = []
    for row in range(max(m_e, m_o)):
        for lz in range(hc):
            cz = lz * h
            for col in range(n):
                cx = col*STEP + r
                is_odd = (col % 2 == 1)
                oy = od/2 if is_odd else 0.0
                mc = m_o if is_odd else m_e
                if row < mc:
                    pos.append((cx, y0+oy+row*od+r, cz, h, 'z'))
    pos = pos[:max_n]
    if pos:
        max_yc = max(p[1] for p in pos)
        len_used = min(max_yc + od/2 - y0, avail_y)
    else:
        len_used = 0
    return pos, len(pos), len_used

def simulate_zone(specs, method):
    """규격 순서대로 컨테이너 길이를 분할해 적재"""
    fn = pack_zone_flat if method=='flat' else pack_zone_honeycomb
    conts=[]; cur={'zones':[],'y_used':0}
    for ti,(od,h,qty,col,lbl) in enumerate(specs):
        rem=qty
        while rem>0:
            avail = CONT_L - cur['y_used']
            if avail < od:
                conts.append(cur); cur={'zones':[],'y_used':0}; avail=CONT_L
            pos,packed,lu = fn(od,h,cur['y_used'],avail,rem)
            if packed==0:
                conts.append(cur); cur={'zones':[],'y_used':0}; continue
            cur['zones'].append((ti,pos,packed))
            cur['y_used'] += lu; rem -= packed
    if cur['zones']: conts.append(cur)
    return conts or [{'zones':[],'y_used':0}]

# ═════════════════════════════════════════════════════════════════════════════
# 3. 층별 배열 (Layer) ─ 컨테이너 높이 방향으로 규격별 층 할당
# ═════════════════════════════════════════════════════════════════════════════
def pack_layer_flat(od, h, z0, avail_z, max_n):
    """층 내 평치: 전체 바닥면(폭×길이) 사용, 높이 방향으로 층 적재"""
    fw=CONT_W//od; fl=CONT_L//od
    if fw==0 or fl==0: return [],0,0
    per_layer = fw*fl
    layers_use = min(math.ceil(max_n/per_layer), avail_z//h)
    if layers_use==0: return [],0,0
    packed = min(max_n, layers_use*per_layer)
    r=od/2; pos=[]
    for lz in range(layers_use):
        for cx in range(fw):
            for cy in range(fl):
                pos.append((cx*od+r, cy*od+r, z0+lz*h, h,'z'))
    return pos[:packed], packed, layers_use*h

def pack_layer_honeycomb(od, h, z0, avail_z, max_n):
    """층 내 벌집: 전체 바닥면 사용, 높이 방향으로 벌집 층 적재"""
    STEP=math.sqrt(3)/2*od
    n=max(1,1+int((CONT_W-od)/STEP))
    wu=od+(n-1)*STEP
    if wu>CONT_W: n-=1; wu=od+(n-1)*STEP
    if n==0: return [],0,0
    m=CONT_L//od
    # 홀수열은 od/2 오프셋 → 마지막 타이어 엣지 = m*od + od/2
    # calc_honeycomb과 동일한 조건: od/2 + m*od ≤ CONT_L
    m_o = m if (od/2 + m*od) <= CONT_L else max(0, m-1)
    oc=math.ceil(n/2); ec=math.floor(n/2)
    one_layer = oc*m + ec*m_o
    if one_layer==0: return [],0,0
    layers_use = min(math.ceil(max_n/one_layer), avail_z//h)
    if layers_use==0: return [],0,0
    packed = min(max_n, layers_use*one_layer)
    r=od/2; pos=[]
    for lz in range(layers_use):
        cz=z0+lz*h
        for col in range(n):
            cx=col*STEP+r; is_odd=(col%2==1)
            oy=od/2 if is_odd else 0.0
            mc=m_o if is_odd else m
            for row in range(mc):
                pos.append((cx, oy+row*od+r, cz, h,'z'))
    return pos[:packed], packed, layers_use*h

def simulate_layer(specs, method):
    """규격 순서대로 컨테이너 높이를 분할해 적재"""
    fn = pack_layer_flat if method=='flat' else pack_layer_honeycomb
    conts=[]; cur={'zones':[],'z_used':0}
    for ti,(od,h,qty,col,lbl) in enumerate(specs):
        rem=qty
        while rem>0:
            avail = CONT_H - cur['z_used']
            if avail < h:
                conts.append(cur); cur={'zones':[],'z_used':0}; avail=CONT_H
            pos,packed,hu = fn(od,h,cur['z_used'],avail,rem)
            if packed==0:
                conts.append(cur); cur={'zones':[],'z_used':0}; continue
            cur['zones'].append((ti,pos,packed))
            cur['z_used'] += hu; rem -= packed
    if cur['zones']: conts.append(cur)
    return conts or [{'zones':[],'z_used':0}]

# ═════════════════════════════════════════════════════════════════════════════
# 4. 3D 메시
# ═════════════════════════════════════════════════════════════════════════════
def cylinder_batch(items, radius):
    n=N_SEG; theta=np.linspace(0,2*np.pi,n,endpoint=False)
    ct,st_=np.cos(theta),np.sin(theta)
    AX,AY,AZ,AI,AJ,AK=[],[],[],[],[],[]
    off=0
    for (cx,cy,cz,hl,axis) in items:
        if axis=='z':
            xb=radius*ct+cx; yb=radius*st_+cy; zb=np.full(n,cz)
            xt=radius*ct+cx; yt=radius*st_+cy; zt=np.full(n,cz+hl)
            c0=(cx,cy,cz); c1=(cx,cy,cz+hl)
        elif axis=='x':
            xb=np.full(n,cx); yb=radius*ct+cy; zb=radius*st_+cz
            xt=np.full(n,cx+hl); yt=radius*ct+cy; zt=radius*st_+cz
            c0=(cx,cy,cz); c1=(cx+hl,cy,cz)
        else:
            xb=radius*ct+cx; yb=np.full(n,cy); zb=radius*st_+cz
            xt=radius*ct+cx; yt=np.full(n,cy+hl); zt=radius*st_+cz
            c0=(cx,cy,cz); c1=(cx,cy+hl,cz)
        xs=np.concatenate([xb,xt,[c0[0],c1[0]]])
        ys=np.concatenate([yb,yt,[c0[1],c1[1]]])
        zs=np.concatenate([zb,zt,[c0[2],c1[2]]])
        for v in range(n):
            vn=(v+1)%n
            AI+=[off+v,off+vn]; AJ+=[off+vn,off+v+n]; AK+=[off+v+n,off+vn+n]
        for v in range(n): AI.append(off+2*n); AJ.append(off+v); AK.append(off+(v+1)%n)
        for v in range(n): AI.append(off+2*n+1); AJ.append(off+v+n); AK.append(off+(v+1)%n+n)
        AX.extend(xs); AY.extend(ys); AZ.extend(zs); off+=2*n+2
    return np.array(AX),np.array(AY),np.array(AZ),AI,AJ,AK

def container_wireframe():
    corners=np.array(list(itertools.product([0,CONT_W],[0,CONT_L],[0,CONT_H])))
    xw,yw,zw=[],[],[]
    for s,e in combinations(corners,2):
        if np.sum(np.abs(s-e)) in [CONT_W,CONT_L,CONT_H]:
            xw+=[s[0],e[0],None]; yw+=[s[1],e[1],None]; zw+=[s[2],e[2],None]
    return go.Scatter3d(x=xw,y=yw,z=zw,mode='lines',
        line=dict(color='rgba(15,23,42,0.2)',width=1.5),
        name='컨테이너',showlegend=False,hoverinfo='none')

def build_3d(specs, container, ci, total_c, tag):
    fig=go.Figure(); fig.add_trace(container_wireframe())
    for (ti,positions,qty_p) in container['zones']:
        if not positions: continue
        od,h,_,color,lbl = specs[ti]
        x,y,z,i,j,k = cylinder_batch(positions, od/2)
        fig.add_trace(go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,
            color=color,opacity=0.85,
            name=f'{lbl}  {qty_p:,}개',showlegend=True,flatshading=False,
            lighting=dict(ambient=0.45,diffuse=0.85,roughness=0.4,specular=0.4,fresnel=0.15),
            lightposition=dict(x=CONT_W*2,y=CONT_L,z=CONT_H*3)))
    total_in = sum(q for _,_,q in container['zones'])
    fig.update_layout(
        title=dict(
            text=f'<b>{tag}</b>  ·  컨테이너 {ci+1}/{total_c}  ·  {total_in:,}개',
            font=dict(size=12,family='Noto Sans KR, sans-serif',color='#0F172A')),
        scene=dict(
            xaxis=dict(title='W (mm)',range=[0,CONT_W],showbackground=True,
                backgroundcolor='rgba(241,245,249,0.6)',gridcolor='rgba(148,163,184,0.4)',tickfont=dict(size=9)),
            yaxis=dict(title='L (mm)',range=[0,CONT_L],showbackground=True,
                backgroundcolor='rgba(226,232,240,0.4)',gridcolor='rgba(148,163,184,0.4)',tickfont=dict(size=9)),
            zaxis=dict(title='H (mm)',range=[0,CONT_H],showbackground=True,
                backgroundcolor='rgba(248,250,252,0.6)',gridcolor='rgba(148,163,184,0.4)',tickfont=dict(size=9)),
            aspectmode='data',camera=dict(eye=dict(x=1.5,y=-1.9,z=1.1))),
        margin=dict(l=0,r=0,t=46,b=0),height=600,
        legend=dict(x=0.01,y=0.98,bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#E2E8F0',borderwidth=1,font=dict(size=11,family='Noto Sans KR')),
        paper_bgcolor='#FAFAFA')
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# 5. Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="타이어 혼재 적재", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[class*="css"],.stApp{font-family:'Noto Sans KR',-apple-system,sans-serif;background:#F8FAFC}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem 2.5rem 3rem;max-width:1440px}

/* ── 헤더 */
.app-header{display:flex;align-items:baseline;gap:1.5rem;padding-bottom:1.2rem;
  border-bottom:2px solid #0F172A;margin-bottom:1.75rem}
.app-title{font-size:1.3rem;font-weight:700;color:#0F172A;letter-spacing:-0.02em}
.app-meta{font-size:0.76rem;color:#64748B;font-family:'IBM Plex Mono',monospace}

/* ── 입력 섹션 */
.section-label{font-size:0.68rem;font-weight:700;color:#94A3B8;
  text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.75rem}
.color-dot{width:11px;height:11px;border-radius:50%;display:inline-block;margin-top:2px}
.input-row-header{display:flex;gap:0.5rem;margin-bottom:0.3rem;padding-left:36px}
.input-col-label{font-size:0.68rem;font-weight:600;color:#94A3B8;
  text-transform:uppercase;letter-spacing:0.06em}
.input-col-label.spec{width:calc(2/4.9*100%)}
.input-col-label.qty{width:calc(2/4.9*100%)}

/* ── 규격 배지 */
.badges-row{display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem}
.tire-badge{padding:0.45rem 0.9rem;border-radius:6px;font-size:0.78rem;
  font-family:'IBM Plex Mono',monospace;background:#fff;
  border:1px solid #E2E8F0;line-height:1.5;border-left-width:4px}

/* ── 비교 요약 테이블 */
.cmp-wrap{background:#fff;border:1px solid #E2E8F0;border-radius:10px;
  overflow:hidden;margin-bottom:1.75rem}
table.cmp-table{width:100%;border-collapse:collapse;font-size:0.83rem}
table.cmp-table th{background:#F8FAFC;font-size:0.65rem;font-weight:700;
  color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;
  padding:0.65rem 1rem;border-bottom:1px solid #E2E8F0;text-align:left}
table.cmp-table td{padding:0.65rem 1rem;border-bottom:1px solid #F1F5F9;
  color:#334155;vertical-align:middle}
table.cmp-table tr:last-child td{border-bottom:none}
table.cmp-table tr.best td{background:#F0FDF4;font-weight:600;color:#0F172A}
.badge-best{background:#16A34A;color:#fff;font-size:0.6rem;font-weight:700;
  border-radius:4px;padding:2px 6px;vertical-align:middle;margin-left:6px}
.badge-method{display:inline-block;font-size:0.65rem;font-weight:600;
  border-radius:4px;padding:2px 7px;margin-right:4px}
.badge-flat{background:#EFF6FF;color:#1D4ED8}
.badge-honey{background:#FFFBEB;color:#B45309}
.badge-zone{background:#F0FDF4;color:#15803D}
.badge-layer{background:#FDF4FF;color:#7E22CE}
.mono{font-family:'IBM Plex Mono',monospace;font-size:0.78rem}

/* ── 컨테이너 네비 */
.cont-nav{display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;
  background:#fff;border:1px solid #E2E8F0;border-radius:8px;
  padding:0.6rem 1rem;width:fit-content}
.cont-nav-label{font-size:0.75rem;font-weight:600;color:#64748B}
.cont-nav-val{font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
  font-weight:700;color:#0F172A}

/* ── Buttons */
.stButton>button{background:#0F172A;color:#F8FAFC;border:none;border-radius:6px;
  font-size:0.82rem;font-weight:500;padding:0.45rem 1.2rem;
  font-family:'Noto Sans KR',sans-serif}
.stButton>button:hover{background:#1E293B}
[data-testid="stBaseButton-primary"]>button,
div[data-testid="stButton"]>button[kind="primary"]{
  background:#0F172A !important;font-size:0.85rem !important}

/* ── 탭 */
.stTabs [data-baseweb="tab-list"]{gap:0;background:transparent;
  border-bottom:1px solid #E2E8F0}
.stTabs [data-baseweb="tab"]{font-size:0.85rem;font-weight:500;color:#64748B;
  padding:0.7rem 1.4rem;background:transparent;
  border-bottom:2px solid transparent;margin-bottom:-1px;border-radius:0}
.stTabs [aria-selected="true"]{color:#0F172A !important;font-weight:700 !important;
  border-bottom:2px solid #0F172A !important;background:transparent !important}
.stTabs [data-baseweb="tab-highlight"]{display:none !important}
</style>
""", unsafe_allow_html=True)

# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <span class="app-title">타이어 혼재 적재 시뮬레이터</span>
  <span class="app-meta">컨테이너 W {CONT_W} × L {CONT_L} × H {CONT_H} mm</span>
</div>
""", unsafe_allow_html=True)

# ── 세션 상태 초기화 ────────────────────────────────────────────────────────
if 'rows' not in st.session_state:
    st.session_state.rows = [
        {'spec':'205/55R16', 'qty':200},
        {'spec':'195/65R15', 'qty':150},
        {'spec':'225/45R17', 'qty':100},
    ]
if 'results' not in st.session_state:
    st.session_state.results = None

# ── 규격 입력 ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">적재 규격 목록</div>', unsafe_allow_html=True)

# 헤더 레이블
h1, h2, h3, h4 = st.columns([0.28, 2.15, 2.15, 0.35])
with h2: st.markdown('<div class="input-col-label">타이어 규격</div>', unsafe_allow_html=True)
with h3: st.markdown('<div class="input-col-label">수량 (개)</div>', unsafe_allow_html=True)

to_delete = None
for idx in range(len(st.session_state.rows)):
    row  = st.session_state.rows[idx]
    dot  = PALETTE[idx % len(PALETTE)]
    c0, c1, c2, c3 = st.columns([0.28, 2.15, 2.15, 0.35])
    with c0:
        st.markdown(
            f'<div style="padding-top:0.55rem">'
            f'<span class="color-dot" style="background:{dot}"></span></div>',
            unsafe_allow_html=True)
    with c1:
        v = st.text_input("spec", value=row['spec'], key=f"spec_{idx}",
                          label_visibility="collapsed", placeholder="예: 205/55R16")
        st.session_state.rows[idx]['spec'] = v
    with c2:
        q = st.number_input("qty", value=int(row['qty']), min_value=1, max_value=99999,
                             key=f"qty_{idx}", label_visibility="collapsed")
        st.session_state.rows[idx]['qty'] = int(q)
    with c3:
        st.markdown('<div style="padding-top:0.3rem"></div>', unsafe_allow_html=True)
        if len(st.session_state.rows) > 1:
            if st.button("✕", key=f"del_{idx}", help="행 삭제"):
                to_delete = idx

if to_delete is not None:
    st.session_state.rows.pop(to_delete)
    st.session_state.results = None
    st.rerun()

# ── Add / Run 버튼 ────────────────────────────────────────────────────────
st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
ba, br, bp = st.columns([1.6, 1.8, 5])
with ba:
    if st.button("＋ 규격 추가", use_container_width=True):
        st.session_state.rows.append({'spec':'', 'qty':100})
        st.session_state.results = None
        st.rerun()
with br:
    run_clicked = st.button("▶  시뮬레이션 실행", use_container_width=True, type="primary")

# ── 시뮬레이션 실행 ───────────────────────────────────────────────────────
if run_clicked:
    specs, valid = [], True
    for i, row in enumerate(st.session_state.rows):
        s = row['spec'].strip()
        if not s:
            st.error(f"행 {i+1}: 규격을 입력하세요."); valid=False; break
        od, h = parse_tire_size(s)
        if od is None:
            st.error(f"행 {i+1}: '{s}' 인식 불가 — 예: 205/55R16"); valid=False; break
        if od > CONT_W or od > CONT_H or h > CONT_H:
            st.error(f"행 {i+1}: 타이어 크기(OD={od}mm)가 컨테이너를 초과합니다."); valid=False; break
        specs.append((od, h, row['qty'], PALETTE[i%len(PALETTE)], s))
    if valid:
        with st.spinner("4가지 방식 계산 중…"):
            st.session_state.results = {
                'specs': specs,
                'zf':    simulate_zone (specs, 'flat'),
                'zh':    simulate_zone (specs, 'honeycomb'),
                'lf':    simulate_layer(specs, 'flat'),
                'lh':    simulate_layer(specs, 'honeycomb'),
            }

# ── 결과 표시 ─────────────────────────────────────────────────────────────
if not st.session_state.results:
    st.stop()

res   = st.session_state.results
specs = res['specs']
total_qty = sum(qty for _,_,qty,_,_ in specs)

# ── 규격 배지 ─────────────────────────────────────────────────────────────
st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
badges = ""
for od,h,qty,col,lbl in specs:
    badges += (f'<span class="tire-badge" style="border-left-color:{col}">'
               f'{lbl} · OD&nbsp;{od}mm · 폭&nbsp;{h}mm · <b>{qty:,}개</b></span>')
st.markdown(f'<div class="badges-row">{badges}</div>', unsafe_allow_html=True)

# ── 4-way 비교 테이블 ─────────────────────────────────────────────────────
combos = [
    ('구역 배열', '평치', 'zone', 'flat',  res['zf'], 'badge-zone', 'badge-flat'),
    ('구역 배열', '벌집', 'zone', 'honey', res['zh'], 'badge-zone', 'badge-honey'),
    ('층별 배열', '평치', 'layer','flat',  res['lf'], 'badge-layer','badge-flat'),
    ('층별 배열', '벌집', 'layer','honey', res['lh'], 'badge-layer','badge-honey'),
]
min_cont = min(len(c[4]) for c in combos)

rows_html = ""
for strat_lbl, meth_lbl, sk, mk, conts, bcls_s, bcls_m in combos:
    n_c       = len(conts)
    packed    = sum(q for c in conts for _,_,q in c['zones'])
    last_c    = conts[-1]
    last_qty  = sum(q for _,_,q in last_c['zones'])
    last_cap  = sum(q for _,_,q in last_c['zones'])  # approx
    is_best   = n_c == min_cont
    row_cls   = 'class="best"' if is_best else ''
    best_badge= '<span class="badge-best">최적</span>' if is_best else ''
    rows_html += f"""
<tr {row_cls}>
  <td>
    <span class="badge-method {bcls_s}">{strat_lbl}</span>
    <span class="badge-method {bcls_m}">{meth_lbl}</span>
    {best_badge}
  </td>
  <td class="mono" style="font-size:1.1rem;font-weight:700">{n_c}</td>
  <td class="mono">{packed:,}</td>
  <td class="mono">{packed/total_qty*100:.1f}%</td>
  <td class="mono">{last_qty:,}개</td>
</tr>"""

st.markdown(f"""
<div class="cmp-wrap">
<table class="cmp-table">
<thead><tr>
  <th>방식 조합</th>
  <th>컨테이너 수</th>
  <th>총 적재</th>
  <th>수량 달성률</th>
  <th>마지막 컨테이너</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

# ── 3D 시각화 탭 ─────────────────────────────────────────────────────────
tab_flat, tab_honey = st.tabs(["📦  평치 (Flat) 3D", "🔶  벌집 (Honeycomb) 3D"])

def render_3d_tab(key_prefix, conts_zone, conts_layer, method_label):
    """평치 또는 벌집 탭 렌더링"""
    zn = len(conts_zone); ln = len(conts_layer)

    oc1, oc2, oc3 = st.columns([1.8, 2.2, 5])
    with oc1:
        st.markdown('<div style="height:0.1rem"></div>', unsafe_allow_html=True)
        strat = st.radio(
            "배열 전략",
            [f"구역 배열  ({zn}컨)", f"층별 배열  ({ln}컨)"],
            key=f"{key_prefix}_strat",
            help="구역: 길이 방향 분할 | 층별: 높이 방향 분할"
        )
    active_conts = conts_zone if strat.startswith("구역") else conts_layer
    active_label = "구역 배열" if strat.startswith("구역") else "층별 배열"
    total_c = len(active_conts)

    with oc2:
        st.markdown('<div style="height:0.1rem"></div>', unsafe_allow_html=True)
        if total_c > 1:
            ci = st.slider(
                f"컨테이너 선택",
                min_value=1, max_value=total_c, value=1,
                key=f"{key_prefix}_ci",
                format=f"%d / {total_c}"
            ) - 1
        else:
            ci = 0
            st.markdown(
                '<div class="cont-nav">'
                '<span class="cont-nav-label">컨테이너</span>'
                '<span class="cont-nav-val">1 / 1</span></div>',
                unsafe_allow_html=True)

    container = active_conts[ci]
    tires_in  = sum(q for _,_,q in container['zones'])
    tag       = f"{method_label} · {active_label}"

    # 컨테이너 내 규격별 현황
    summary_parts = []
    for ti, pos, qty_p in container['zones']:
        od,h,_,col,lbl = specs[ti]
        summary_parts.append(
            f'<span style="color:{col};font-weight:600">{lbl}</span> {qty_p:,}개')
    st.caption("  ·  ".join(summary_parts) + f"  ·  합계 {tires_in:,}개")

    with st.spinner(f"{tag} 3D 생성 중…"):
        fig = build_3d(specs, container, ci, total_c, tag)
    st.plotly_chart(fig, use_container_width=True)

with tab_flat:
    render_3d_tab("f", res['zf'], res['lf'], "평치")

with tab_honey:
    render_3d_tab("h", res['zh'], res['lh'], "벌집")

# ── 컨테이너별 상세 ───────────────────────────────────────────────────────
st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
with st.expander("전체 컨테이너 상세 보기"):
    combo_detail = [
        ("구역×평치", res['zf']),
        ("구역×벌집", res['zh']),
        ("층별×평치", res['lf']),
        ("층별×벌집", res['lh']),
    ]
    detail_tabs = st.tabs([f"{lbl} ({len(c)}컨)" for lbl,c in combo_detail])
    for tab_d, (lbl_d, conts_d) in zip(detail_tabs, combo_detail):
        with tab_d:
            rows_det=[]
            for i,cont in enumerate(conts_d):
                row_d = {"컨테이너": i+1}
                for ti,_,qty_p in cont['zones']:
                    row_d[specs[ti][4]] = qty_p
                row_d["합계"] = sum(q for _,_,q in cont['zones'])
                rows_det.append(row_d)
            st.dataframe(pd.DataFrame(rows_det).fillna(0).astype(
                {c:int for c in pd.DataFrame(rows_det).columns if c!="컨테이너"}),
                use_container_width=True, hide_index=True)
