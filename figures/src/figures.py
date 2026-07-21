#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import svgkit as K
from svgkit import Fig, C
import data as D

# ============================================================ 01 PIPELINE
def fig_pipeline():
    f = Fig("01_pipeline", 1640, 920,
            "전체 파이프라인 — 하나의 NLI 행렬에서 세 갈래",
            "선생(zero-shot) · 답지(silver label) · 학생(classifier) 이 같은 점수 행렬에서 갈라진다",
            "출처: README §2 · scripts/01·02·03·04·05·06")
    # inputs
    f.box(52, 300, 250, 90, C["grayT"], "문서 29,487", ["라벨 0개"], stroke=C["line"], tsize=21, lsize=17)
    f.box(52, 430, 250, 90, C["grayT"], "531 클래스명 + tree", ["3-level taxonomy"], stroke=C["line"], tsize=20, lsize=17)
    # NLI matrix
    f.arrow(302, 345, 372, 400, C["grayD"]); f.arrow(302, 475, 372, 430, C["grayD"])
    f.box(374, 360, 210, 110, C["navyT"], "NLI 유사도 행렬", ["roberta-large-mnli", "문서당 68.5 클래스"],
          stroke=C["navy"], tcolor=C["navy"], tsize=21, lsize=16.5)
    # branch hub
    hubx = 584
    lanes_y = [210, 415, 640]
    for ly in lanes_y:
        f.path(f"M{hubx},415 C{hubx+40},415 {hubx+20},{ly+40} {hubx+70},{ly+40}", stroke=C["grayD"], w=2.2)
    # Lane A — teacher
    f.box(hubx+70, 168, 250, 96, C["navyT"], "① 선생 · zero-shot", ["점수를 그대로 예측"],
          stroke=C["navy"], tcolor=C["navy"], tsize=21, lsize=17)
    f.arrow(hubx+320, 216, hubx+390, 216, C["navy"])
    f.box(hubx+392, 176, 200, 80, "#fff", "path decoding", ["568 경로 argmax"], stroke=C["navy"], tsize=19, lsize=16)
    f.arrow(hubx+592, 216, hubx+662, 216, C["navy"])
    f.box(hubx+664, 178, 150, 76, C["navyT"], "0.4623", ["Example-F1"], stroke=C["navy"], tcolor=C["navy"], tsize=27, lsize=15)
    # Lane B — answer key
    f.box(hubx+70, 373, 250, 96, C["orangeT"], "② 답지 · silver label", ["core class mining"],
          stroke=C["orange"], tcolor=C["orangeD"], tsize=21, lsize=17)
    f.arrow(hubx+320, 421, hubx+390, 421, C["orange"])
    f.box(hubx+392, 381, 200, 80, "#fff", "이중 필터", ["median + min_conf"], stroke=C["orange"], tsize=19, lsize=16)
    f.arrow(hubx+592, 421, hubx+662, 421, C["orange"])
    f.box(hubx+664, 383, 200, 76, C["orangeT"], "silver 30→49%", ["라벨 단위 precision"], stroke=C["orange"],
          tcolor=C["orangeD"], tsize=22, lsize=15)
    # Lane C — student
    f.box(hubx+70, 600, 250, 96, C["tealT"], "③ 학생 · classifier", ["BERT [CLS] + GCN"],
          stroke=C["teal"], tcolor=C["teal"], tsize=21, lsize=17)
    f.arrow(hubx+320, 648, hubx+390, 648, C["teal"])
    f.box(hubx+392, 608, 200, 80, "#fff", "path decoding", ["+ self-training"], stroke=C["teal"], tsize=19, lsize=16)
    f.arrow(hubx+592, 648, hubx+662, 648, C["teal"])
    f.box(hubx+664, 610, 150, 76, C["tealT"], "0.5488", ["Example-F1"], stroke=C["teal"], tcolor=C["teal"], tsize=27, lsize=15)
    # answer-key feeds student
    f.path(f"M{hubx+789},459 C{hubx+789},520 {hubx+520},545 {hubx+195},545 L{hubx+195},600",
           stroke=C["orange"], w=2.2, dash="2 7")
    f.text(hubx+300, 536, "답지로 학습", 17, 700, C["orangeD"])
    # student beats teacher note
    f.text(hubx+664, 738, "학생이 선생을 넘는가 = 파이프라인 성패", 17.5, 600, C["sec"])
    return f

# ============================================================ 02 PROBLEM SETUP
def fig_problem():
    f = Fig("02_problem_setup", 1640, 900,
            "문제 설정 — 라벨 0개, 클래스 531개",
            "정답이 항상 root-to-leaf 경로라는 성질이 예측 공간을 2⁵³¹에서 568로 축소한다",
            "출처: results/phase0.json (전수 검증)")
    # stat tiles (left, 2x3)
    tiles = [("531","클래스",C["ink"]),("3","레벨 (6·64·461)",C["ink"]),
             ("568","candidate paths",C["orangeD"]),("29,487","train 문서",C["ink"]),
             ("19,658","test 문서",C["ink"]),("0","사람 라벨",C["red"])]
    tx, ty, tw, th, gx, gy = 52, 210, 250, 118, 22, 22
    for i,(v,l,col) in enumerate(tiles):
        r,cc = divmod(i,3)
        x=tx+cc*(tw+gx); y=ty+r*(th+gy)
        f.rrect(x,y,tw,th,"#fff",12,C["line"])
        f.text(x+24,y+62,v,44,700,col); f.text(x+24,y+96,l,18,500,C["sec"])
    # insight strip
    f.rrect(52, 484, tw*3+gx*2, 96, C["orangeT"], 12, C["orange"])
    f.text(76, 528, "예측 = 2⁵³¹ 부분집합 선택  →  568 경로 중 하나 선택", 25, 700, C["orangeD"])
    f.text(76, 560, "정답 label set 100%가 root-to-leaf 경로 (길이 2: 7.5% · 길이 3: 92.5% · 예외 0건)", 18.5, 500, C["sec"])
    f.text(76, 636, "모델링 전에 데이터 구조를 먼저 검증한 것이 이후 모든 설계의 출발점", 19, 600, C["ink"])
    # mini taxonomy tree (right), one path highlighted
    ox, oy = 1010, 250
    root=(ox+300, oy)
    mids=[(ox+120,oy+150),(ox+300,oy+150),(ox+480,oy+150)]
    leaves=[(ox+60,oy+300),(ox+180,oy+300),(ox+300,oy+300),(ox+420,oy+300),(ox+540,oy+300)]
    # edges
    for m in mids: f.line(root[0],root[1]+30,m[0],m[1]-30,C["grid"],2)
    hi=[(mids[1]),(leaves[2])]
    f.line(mids[0][0],mids[0][1]+30,leaves[0][0],leaves[0][1]-30,C["grid"],2)
    f.line(mids[0][0],mids[0][1]+30,leaves[1][0],leaves[1][1]-30,C["grid"],2)
    f.line(mids[1][0],mids[1][1]+30,leaves[2][0],leaves[2][1]-30,C["orange"],3)
    f.line(mids[2][0],mids[2][1]+30,leaves[3][0],leaves[3][1]-30,C["grid"],2)
    f.line(mids[2][0],mids[2][1]+30,leaves[4][0],leaves[4][1]-30,C["grid"],2)
    f.line(root[0],root[1]+30,mids[1][0],mids[1][1]-30,C["orange"],3)
    def node(p,fill,stroke,lab,txtcol=C["sec"]):
        f.circle(p[0],p[1],30,fill,stroke,2.5); f.text(p[0],p[1]+7,lab,17,700,txtcol,"middle")
    node(root,C["orangeT"],C["orange"],"root",C["orangeD"])
    for i,m in enumerate(mids):
        node(m,(C["orangeT"] if i==1 else "#fff"),(C["orange"] if i==1 else C["grayD"]),"mid",(C["orangeD"] if i==1 else C["muted"]))
    for i,l in enumerate(leaves):
        node(l,(C["orangeT"] if i==2 else "#fff"),(C["orange"] if i==2 else C["grayD"]),"leaf",(C["orangeD"] if i==2 else C["muted"]))
    f.text(ox+300, oy+372, "한 경로 = 하나의 정답 (root→mid→leaf)", 18, 600, C["orangeD"], "middle")
    return f

# ============================================================ 10 PRECISION-COVERAGE
def fig_prec_cov():
    f = Fig("10_precision_coverage", 1300, 860,
            "Precision–Coverage 트레이드오프",
            "min_conf 필터를 스윕하고 무릎점(0.15)을 선택 — 감이 아니라 곡선으로 결정",
            "출처: results/phase2_sweep.json (train GT 감사)")
    ml,mr,mt,mb=120,150,self_top(f)+30,150
    pw=1300-ml-mr; ph=860-mt-mb
    xmin,xmax=40,100      # coverage %
    ymin,ymax=25,55       # precision %
    def X(v): return ml+pw*(v-xmin)/(xmax-xmin)
    def Y(v): return mt+ph*(1-(v-ymin)/(ymax-ymin))
    # grid
    for gx in range(40,101,10):
        f.line(X(gx),mt,X(gx),mt+ph,C["grid"],1); f.text(X(gx),mt+ph+34,f"{gx}",18,500,C["muted"],"middle")
    for gy in range(25,56,5):
        f.line(ml,Y(gy),ml+pw,Y(gy),C["grid"],1); f.text(ml-16,Y(gy)+6,f"{gy}",18,500,C["muted"],"end")
    f.text(ml+pw/2, mt+ph+72, "Coverage — 답지 보유 문서 %", 20, 600, C["sec"], "middle")
    f.text(ml-16, mt-24, "Precision %", 20, 600, C["sec"], "start")
    # curve
    pts=[(X(s["cov"]),Y(s["prec"])) for s in D.SWEEP]
    d="M"+" L".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    f.path(d, stroke=C["orange"], w=3)
    for s,(x,y) in zip(D.SWEEP,pts):
        knee = abs(s["min_conf"]-D.SWEEP_KNEE)<1e-6
        f.circle(x,y,8 if not knee else 11, C["orange"] if not knee else "#fff", C["orange"] if knee else None, 3)
        lbl=f'min_conf {s["min_conf"]:.2f}'
        f.text(x, y-22 if not knee else y-26, f'{s["prec"]:.1f}%', 18, 700, C["orangeD"], "middle")
        f.text(x+ (0 if knee else 0), y+34, lbl, 14.5, 500, C["muted"], "middle")
    # knee callout
    kx,ky=X(78.6),Y(43.16)
    f.line(kx,ky,kx,mt-6,C["navy"],1.4,dash="4 5")
    f.rrect(kx-96,mt-44,192,42,C["navyT"],10,C["navy"])
    f.text(kx,mt-16,"채택: 무릎점 0.15",18,700,C["navy"],"middle")
    f.text(kx+140, ky+6, "precision 43.2% / coverage 79%", 16.5, 600, C["navy"], "start")
    return f

def self_top(f): return f.top

# ============================================================ 12 TRAJECTORY (waterfall)
ROLE_FILL={"base":C["grayD"],"drop":C["charcoal"],"data":C["gray"],"final":C["orange"]}
ROLE_LBL ={"base":C["sec"],"drop":C["charcoal"],"data":C["ink"],"final":C["orangeD"]}
def fig_trajectory():
    f=Fig("12_perf_trajectory",1560,900,
          "성능 궤적 — 각 막대는 하나의 가설·검증",
          "모델·아키텍처 고정, 데이터만 개선 · Test Example-F1 · seed 고정",
          "출처: results/ (phase1·2·3·4) · README §4")
    ml,mr,mt,mb=120,200,f.top+70,180
    pw=1560-ml-mr; ph=900-mt-mb; ymax=0.60
    def Y(v): return mt+ph*(1-v/ymax)
    n=len(D.JOURNEY); slot=pw/n; barw=150
    def BX(i): return ml+slot*i+(slot-barw)/2
    for gv in [0,.15,.30,.45,.60]:
        f.line(ml,Y(gv),ml+pw,Y(gv),C["grid"],1); f.text(ml-16,Y(gv)+6,f"{gv:.2f}",18,500,C["muted"],"end")
    yn=Y(D.NOST); f.line(ml,yn,ml+pw,yn,C["navy"],1.6,dash="7 6")
    f.text(ml+pw+14,yn-6,"논문 NoST",18,700,C["navy"]); f.text(ml+pw+14,yn+18,f"{D.NOST:.4f}",18,600,C["navy"])
    tops=[]
    for i,b in enumerate(D.JOURNEY):
        x=BX(i); y=Y(b["f1"]); h=(mt+ph)-y; tops.append((x+barw/2,y))
        f.rrect(x,y,barw,h,ROLE_FILL[b["role"]],6)
        f.text(x+barw/2,y-16,f'{b["f1"]:.4f}',29,700,ROLE_LBL[b["role"]],"middle")
        l1,l2=b["label"].split("\n")
        f.text(x+barw/2,mt+ph+36,l1,20,700,C["ink"],"middle")
        f.text(x+barw/2,mt+ph+60,l2,17.5,500,C["sec"],"middle")
        if b["sub"]: f.text(x+barw/2,mt+ph+86,b["sub"],16.5,600,C["orangeD"] if b["role"]=="final" else C["muted"],"middle")
    for i in range(1,n):
        b=D.JOURNEY[i]
        if b["delta"] is None: continue
        x0,y0=tops[i-1]; x1,y1=tops[i]; mx=(x0+x1)/2; my=min(y0,y1)-46
        up=b["delta"]>0
        col=C["orangeD"] if b["role"]=="final" else (C["charcoal"] if not up else C["sec"])
        txt=f'{"▲ +" if up else "▼ −"}{abs(b["delta"]):.1f}pt'
        f.rrect(mx-60,my-24,120,40,C["orangeT"] if b["role"]=="final" else "#fff",20,col,1.4)
        f.text(mx,my+2,txt,21,700,col,"middle")
    return f

def _legend(f, x, y, items, size=18):
    for i,(col,dash,lab) in enumerate(items):
        yy=y+i*32
        f.line(x,yy,x+38,yy,col,3.4,dash=dash)
        f.circle(x+19,yy,5.5,col)
        f.text(x+50,yy+6,lab,size,600,C["ink"])

def _polyline(f, pts, color, dash=None, w=3.4, mark=7):
    d="M"+" L".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    f.path(d, stroke=color, w=w, dash=dash)
    for x,y in pts: f.circle(x,y,mark,"#fff",color,3)

def build(FIGURES):
    figs=[fn() for fn in FIGURES]
    css=K.build_font_css()
    base=os.path.dirname(__file__)
    svgd=os.path.join(base,"fig","svg"); htmd=os.path.join(base,"fig","html")
    os.makedirs(svgd,exist_ok=True); os.makedirs(htmd,exist_ok=True)
    for f in figs:
        svg=f.svg(css)
        open(os.path.join(svgd,f.key+".svg"),"w").write(svg)
        open(os.path.join(htmd,f.key+".html"),"w").write(
            f'<!doctype html><html><head><meta charset="utf-8"><style>*{{margin:0;padding:0}}</style></head><body>{svg}</body></html>')
    print("built:", ", ".join(f.key for f in figs))

# ============================================================ 03 EVAL PROTOCOL
def fig_eval_protocol():
    f=Fig("03_eval_protocol",1640,880,
          "평가 프로토콜 — Ground Truth의 역할 분리",
          "‘라벨 0개’ = GT가 없다는 뜻이 아니라, 모델을 만드는 과정이 GT를 볼 수 없다는 뜻",
          "출처: README §3 (GT 역할 분리)")
    # center production flow (no GT inside)
    fy=430
    f.box(60,fy-46,230,92,C["tealT"],"silver label",["(자동 채굴)"],stroke=C["teal"],tcolor=C["teal"],tsize=21,lsize=16)
    f.arrow(292,fy,352,fy,C["grayD"])
    f.box(354,fy-56,470,112,"#FBFAF8","모델 제작 과정",["mining → classifier → self-training"],stroke=C["line"],tsize=22,lsize=18)
    f.text(589,fy+40,"이 안에서 GT를 절대 보지 않는다",15.5,600,C["teal"],"middle")
    f.arrow(826,fy,886,fy,C["grayD"])
    f.box(888,fy-46,210,92,"#fff","예측 경로",["568 중 argmax"],stroke=C["line"],tsize=21,lsize=16)
    f.arrow(1100,fy,1160,fy,C["ink"],2.8)
    f.box(1162,fy-52,180,104,C["navyT"],"채점",["Example-F1"],stroke=C["navy"],tcolor=C["navy"],tsize=23,lsize=16)
    # test GT feeds ONLY scoring
    f.box(1162,700,180,92,"#fff","test GT",["19,658"],stroke=C["navy"],tcolor=C["navy"],tsize=21,lsize=16)
    f.arrow(1252,700,1252,fy+56,C["navy"],2.6)
    f.text(1360,fy+130,"채점 전용",16.5,700,C["navy"]); f.text(1360,fy+156,"파이프라인 미입력",15.5,500,C["sec"])
    # train GT audits only (dashed, never feeds training)
    f.box(354,150,470,92,C["grayT"],"train GT · 29,487",["오프라인 감사 전용"],stroke=C["grayD"],tsize=21,lsize=17)
    f.path("M470,242 C470,320 300,330 175,330 L175,384",stroke=C["muted"],w=2.2,dash="2 7")
    f.path("M700,242 L700,372",stroke=C["muted"],w=2.2,dash="2 7")
    f.text(560,300,"측정 전용 (silver precision 30·43·49%)",16,700,C["sec"])
    f.text(560,326,"학습 신호로는 미사용",15,500,C["muted"])
    # exception note
    f.rrect(60,700,1030,92,C["orangeT"],12,C["orange"])
    f.text(84,742,"유일한 예외 — 행 7의 audit-validation stopping",20,700,C["orangeD"])
    f.text(84,772,"train에서 3,000개 분리해 stopping 신호로 사용 (순수 weak-supervision 벗어남) · §5.4에 공개",17,500,C["sec"])
    return f

# ============================================================ 04 CORE MINING
def fig_core_mining():
    f=Fig("04_core_mining",1640,900,
          "Core Class Mining — 상대 확신도 + 이중 필터",
          "부모·형제보다 뚜렷이 튀는 클래스만 채굴하고, 트리 조상을 붙여 silver label을 만든다",
          "출처: README §2.2 · scripts/03 (논문 Eq.2–3)")
    # formula
    f.rrect(60,210,700,110,C["navyT"],12,C["navy"])
    f.text(90,262,"conf(D, c) = sim(D, c) − max( sim(parent), sim(siblings) )",25,700,C["navy"])
    f.text(90,298,"주변 대비 상대 확신도 — 절대 점수가 아니다",17.5,500,C["sec"])
    # tree with sims
    ox,oy=180,470
    par=(ox+180,oy); c=(ox+60,oy+150); s1=(ox+300,oy+150)
    f.line(par[0],par[1]+34,c[0],c[1]-34,C["orange"],3)
    f.line(par[0],par[1]+34,s1[0],s1[1]-34,C["grid"],2.4)
    def nd(p,val,name,hot):
        col=C["orange"] if hot else C["grayD"]
        f.circle(p[0],p[1],38,C["orangeT"] if hot else "#fff",col,2.6)
        f.text(p[0],p[1]-2,name,16,700,C["orangeD"] if hot else C["muted"],"middle")
        f.text(p[0],p[1]+18,val,15,600,C["sec"],"middle")
        return p
    nd(par,"0.42","parent",False)
    nd(c,"0.71","c ✓",True)
    nd(s1,"0.38","sibling",False)
    f.text(ox+180,oy+250,"conf = 0.71 − max(0.42, 0.38) = +0.29",19,700,C["orangeD"],"middle")
    f.text(ox+180,oy+280,"c가 주변보다 뚜렷이 튐 → core 후보",16.5,500,C["sec"],"middle")
    # double filter -> silver
    fx=820
    f.arrow(fx-40,470,fx+20,470,C["grayD"])
    f.box(fx+30,410,250,120,"#fff","이중 필터",["① 클래스별 median","② min_conf 절대 하한"],stroke=C["orange"],tcolor=C["orangeD"],tsize=21,lsize=17)
    f.arrow(fx+290,470,fx+350,470,C["grayD"])
    f.box(fx+360,410,260,120,C["orangeT"],"core + 트리 조상",["= silver label","(부모는 유일 → 자동)"],stroke=C["orange"],tcolor=C["orangeD"],tsize=21,lsize=16.5)
    # note
    f.rrect(fx+30,600,590,150,C["grayT"],12,C["line"])
    f.text(fx+56,646,"미채굴 = 오답 증거가 아니다",19,700,C["ink"])
    f.text(fx+56,684,"→ core의 자식 클래스는 negative에서 제외",17,500,C["sec"])
    f.text(fx+56,718,"→ 필터 미통과 문서는 답지 없이 학습에서 제외 (§5.2 trade-off)",16,500,C["sec"])
    return f

# ============================================================ 05 CLASSIFIER ARCH
def fig_classifier_arch():
    f=Fig("05_classifier_arch",1520,900,
          "분류기 구조 — 문서 인코더 · 클래스 인코더 · bilinear 매칭",
          "taxonomy 이웃 정보를 GCN으로 주입해 tail class가 부모의 신호를 빌린다",
          "출처: README §2.3 · src/classifier.py (논문 Eq.7–8)")
    # doc tower (left)
    dx=140
    f.box(dx,230,300,80,C["grayT"],"문서 (max 512 tok)",[],stroke=C["line"],tsize=21)
    f.arrow(dx+150,310,dx+150,356,C["grayD"])
    f.box(dx,358,300,86,C["tealT"],"bert-base-uncased",["[CLS] 표현"],stroke=C["teal"],tcolor=C["teal"],tsize=21,lsize=16.5)
    f.arrow(dx+150,444,dx+150,490,C["grayD"])
    f.box(dx,492,300,72,"#fff","문서 벡터",[],stroke=C["line"],tsize=20)
    # class tower (right)
    cx=1080
    f.box(cx,230,300,80,C["grayT"],"taxonomy 그래프",["531 클래스"],stroke=C["line"],tsize=20,lsize=16)
    f.arrow(cx+150,310,cx+150,356,C["grayD"])
    f.box(cx,358,300,86,C["tealT"],"GCN",["ego-mean · residual"],stroke=C["teal"],tcolor=C["teal"],tsize=21,lsize=16.5)
    f.arrow(cx+150,444,cx+150,490,C["grayD"])
    f.box(cx,492,300,72,"#fff","클래스 벡터 ×531",[],stroke=C["line"],tsize=20)
    f.text(cx+150,600,"centering·normalize·learnable",15.5,600,C["orangeD"],"middle")
    f.text(cx+150,624,"(anisotropy 대응 · §6)",14.5,500,C["muted"],"middle")
    # bilinear center
    bx=760
    f.arrow(dx+300,528,bx-96,540,C["grayD"]); f.arrow(cx,528,bx+96,540,C["grayD"])
    f.circle(bx,540,74,C["orangeT"],C["orange"],2.6)
    f.text(bx,532,"bilinear",22,700,C["orangeD"],"middle"); f.text(bx,562,"d·W·c",17,600,C["sec"],"middle")
    f.arrow(bx,614,bx,660,C["grayD"])
    f.box(bx-150,662,300,72,C["orangeT"],"531 클래스 점수",[],stroke=C["orange"],tcolor=C["orangeD"],tsize=21)
    # loss / label note
    f.rrect(140,770,1240,90,C["grayT"],12,C["line"])
    f.text(168,808,"Loss = per-document sum BCE",19,700,C["ink"])
    f.text(168,840,"positive = cores ∪ 부모 · core의 자식은 negative 제외 · mean 정규화 시 collapse (§6)",16.5,500,C["sec"])
    return f

# ============================================================ 06 PATH DECODING
def fig_path_decoding():
    f=Fig("06_path_decoding",1600,900,
          "Path Decoding — 출력 구조를 아는 것만으로 얻는 이득",
          "531개 무제약 점수 대신 568개 candidate path를 집계해 argmax 경로 하나를 제출",
          "출처: results/phase1_zeroshot.json · README §2.5")
    # flow
    f.box(70,250,250,150,"#fff","531 클래스 점수",["(무제약)"],stroke=C["line"],tsize=21,lsize=16)
    f.arrow(322,325,382,325,C["grayD"])
    f.box(384,250,280,150,C["navyT"],"568 경로 점수 집계",["각 root-to-leaf","경로 점수"],stroke=C["navy"],tcolor=C["navy"],tsize=21,lsize=16.5)
    f.arrow(666,325,726,325,C["grayD"])
    # tree argmax
    ox,oy=830,280
    root=(ox+230,oy); mids=[(ox+90,oy+120),(ox+230,oy+120),(ox+370,oy+120)]
    leaves=[(ox+40,oy+240),(ox+150,oy+240),(ox+230,oy+240),(ox+330,oy+240),(ox+430,oy+240)]
    for m in mids: f.line(root[0],root[1]+26,m[0],m[1]-26,C["grid"],2)
    for i,l in enumerate(leaves):
        pm=mids[min(i//2,2)]
        f.line(pm[0],pm[1]+26,l[0],l[1]-26,C["grid"],2)
    f.line(root[0],root[1]+26,mids[1][0],mids[1][1]-26,C["orange"],3.4)
    f.line(mids[1][0],mids[1][1]+26,leaves[2][0],leaves[2][1]-26,C["orange"],3.4)
    for p,hot in [(root,1)]+[(m,i==1) for i,m in enumerate(mids)]+[(l,i==2) for i,l in enumerate(leaves)]:
        f.circle(p[0],p[1],24,C["orangeT"] if hot else "#fff",C["orange"] if hot else C["grayD"],2.4)
    f.text(ox+230,oy+300,"argmax 경로 = 제출",18,700,C["orangeD"],"middle")
    # decoder comparison bars
    by=560; bx0=90; bw=900; labs=[("top-3 (무제약)","top3",C["grayD"]),("hier-greedy","hier_greedy",C["gray"]),("path (len3)","path_len3",C["orange"]),("path (auto) 채택","path_auto",C["orange"])]
    vmax=0.50
    f.text(bx0,by-22,"디코더별 Example-F1 — 구조 제약만으로 +19pt",19,700,C["ink"])
    for i,(nm,k,col) in enumerate(labs):
        v=D.DECODERS[k]; yy=by+i*66
        f.text(bx0,yy+30,nm,17.5,600,C["ink"])
        f.rrect(bx0+300,yy+8,bw*v/vmax,40,col if "path" in k else C["gray"],6)
        f.text(bx0+300+bw*v/vmax+14,yy+34,f"{v:.4f}",19,700,C["orangeD"] if "path" in k else C["sec"])
    return f

# ============================================================ 07 METRIC MASKING
def fig_metric_masking():
    f=Fig("07_metric_masking",1520,860,
          "지표가 데이터 문제를 가리고 있었다",
          "같은 데이터, 평가 단위만 바꿨다 — 문서 단위 71%가 라벨 단위 30.3%를 숨겼다",
          "출처: results/phase2_sweep.json · README §5.1")
    # two metric cards + reversal
    f.rrect(70,250,360,220,"#fff",14,C["line"])
    f.text(250,332,"71%",70,700,C["muted"],"middle")
    f.text(250,392,"문서 단위 Hit Rate",22,600,C["sec"],"middle")
    f.text(250,432,"‘정답 1개 이상 포함’",17,500,C["muted"],"middle")
    f.arrow(452,360,560,360,C["ink"],3); f.text(506,338,"재측정",17,700,C["ink"],"middle")
    f.rrect(582,250,360,220,C["orangeT"],14,C["orange"])
    f.text(762,332,"30.3%",70,700,C["orangeD"],"middle")
    f.text(762,392,"라벨 단위 Precision",22,600,C["ink"],"middle")
    f.text(762,432,"‘생성 라벨 중 정답 비율’",17,500,C["sec"],"middle")
    # per-doc breakdown stacked bar
    bx,by,bw=70,560,872
    corr=1.1/3.8; 
    f.text(bx,by-18,"문서당 평균 3.8개 라벨을 생성",20,700,C["ink"])
    f.rrect(bx,by+6,bw*corr,64,C["orange"],6)
    f.rrect(bx+bw*corr+3,by+6,bw*(1-corr)-3,64,C["gray"],6)
    f.text(bx+bw*corr/2,by+46,"1.1 정답",19,700,"#fff","middle")
    f.text(bx+bw*corr+ (bw*(1-corr))/2,by+46,"2.7 오답",20,700,C["ink"],"middle")
    f.text(bx,by+110,"학습은 라벨 단위로 이루어진다 → 데이터 품질도 라벨 단위로 평가해야 한다.",18.5,600,C["sec"])
    f.text(bx,by+140,"지표를 잘못 고르면 데이터 문제가 지표상 존재하지 않는다.",18.5,600,C["orangeD"])
    return f

# ============================================================ 08 OVERFITTING
def fig_overfitting():
    f=Fig("08_overfitting",1560,840,
          "과적합 시그니처 — 노이즈 라벨을 외운다",
          "초기 설정은 silver를 외울수록 test가 정체 · 교정 후에는 둘이 함께 상승",
          "출처: results/phase3_log_b32 · phase3_log_512e3")
    ymin,ymax=0.30,0.92
    def panel(px,pw,title,rows):
        mt=f.top+70; mb=180; ph=840-mt-mb
        def X(i): return px+70+ (pw-140)*i/2
        def Y(v): return mt+ph*(1-(v-ymin)/(ymax-ymin))
        f.text(px+pw/2,mt-30,title,20,700,C["ink"],"middle")
        for gv in [0.3,0.5,0.7,0.9]:
            f.line(px+50,Y(gv),px+pw-30,Y(gv),C["grid"],1)
            f.text(px+44,Y(gv)+6,f"{gv:.1f}",15.5,500,C["muted"],"end")
        fit=[(X(i),Y(r["fit"])) for i,r in enumerate(rows)]
        f1=[(X(i),Y(r["f1"])) for i,r in enumerate(rows)]
        _polyline(f,fit,C["charcoal"],dash="6 5",mark=6)
        _polyline(f,f1,C["orange"],mark=6)
        for i,r in enumerate(rows):
            f.text(X(i),840-mb+34,f"ep{r['ep']}",16,600,C["sec"],"middle")
        f.text(fit[-1][0]+6,fit[-1][1]-14,f'{rows[-1]["fit"]:.2f}',16,700,C["charcoal"],"end")
        f.text(f1[-1][0]+14,f1[-1][1]+6,f'{rows[-1]["f1"]:.4f}',16,700,C["orangeD"])
    panel(70,690,"초기 설정 (len 256) — 외우지만 test 정체",D.OVERFIT_B32)
    panel(800,690,"truncation 교정 후 (len 512) — 함께 상승",D.FIXED_512)
    _legend(f,70,760,[(C["charcoal"],"6 5","train 적합도 (silver fit p@1)"),(C["orange"],None,"test Example-F1")])
    f.text(800,772,"train loss는 두 경우 모두 감소 (17.8→9.7 / 13.8→5.6) — 감소가 곧 개선이 아니다",16.5,600,C["sec"])
    return f

# ============================================================ 09 LEVEL ACCURACY
def fig_level_accuracy():
    f=Fig("09_level_accuracy",1240,820,
          "레벨별 정확도 — 오류는 깊은 곳에 집중",
          "zero-shot greedy 예측 정확도 · 계층이 깊어질수록 급감",
          "출처: results/phase1_zeroshot.json")
    rows=[("root",D.LEVEL_ACC["root"],"6 클래스"),("mid",D.LEVEL_ACC["mid"],"64 클래스"),("leaf",D.LEVEL_ACC["leaf"],"461 클래스")]
    ml,mt,mb=150,f.top+50,190; pw=1240-ml-200; ph=820-mt-mb
    def Y(v): return mt+ph*(1-v/70)
    for gv in [0,20,40,60]:
        f.line(ml,Y(gv),ml+pw,Y(gv),C["grid"],1); f.text(ml-16,Y(gv)+6,f"{gv}%",17,500,C["muted"],"end")
    n=len(rows); slot=pw/n; bw=150
    for i,(nm,v,cnt) in enumerate(rows):
        x=ml+slot*i+(slot-bw)/2; y=Y(v); h=(mt+ph)-y
        col=C["orange"] if nm=="leaf" else C["gray"]
        f.rrect(x,y,bw,h,col,6)
        f.text(x+bw/2,y-16,f"{v:.1f}%",30,700,C["orangeD"] if nm=="leaf" else C["ink"],"middle")
        f.text(x+bw/2,mt+ph+38,nm,22,700,C["ink"],"middle")
        f.text(x+bw/2,mt+ph+66,cnt,17,500,C["sec"],"middle")
    f.text(ml+pw+30,mt+40,"class 수가 많고",18,600,C["sec"])
    f.text(ml+pw+30,mt+68,"깊은 leaf일수록",18,600,C["sec"])
    f.text(ml+pw+30,mt+96,"정확도 최저",19,700,C["orangeD"])
    f.text(ml+pw+30,mt+150,"→ 개선 여력이",17,500,C["muted"])
    f.text(ml+pw+30,mt+176,"leaf에 집중",17,500,C["muted"])
    return f

# ============================================================ 11 DECISION LOG
def fig_decision_log():
    f=Fig("11_decision_log",1480,1260,
          "가설 배제 로그 — 원인을 좁혀간 과정",
          "새 기법을 더하기 전에, 측정·데이터·설계를 먼저 검증했다",
          "출처: README §5.1–5.3")
    steps=[
      ("PROBLEM",C["charcoal"],"학습한 분류기 0.3995 < zero-shot 0.4623","학습이 성능을 깎는 역전 → 구조적 결함 신호"),
      ("HYPOTHESIS 1 · 기각",C["red"],"학습 설정 문제인가?","batch·sequence length·decoder 통제 → 효과 모두 1pt 미만"),
      ("HYPOTHESIS 2",C["navy"],"데이터 품질 문제인가?","평가 지표를 라벨 단위로 재정의 → precision 30.3% 확인"),
      ("EXPERIMENT 1",C["teal"],"min_conf 0.15 + truncation 512","silver 43.2% · 과적합 소멸 → +6.2pt (0.4614)"),
      ("EXPERIMENT 2 · 실패",C["red"],"LLM 재검수 1K calls","coverage 3.2% → Example-F1 변화 없음"),
      ("ROOT CAUSE",C["charcoal"],"개입 규모가 부족했다","1K는 노이즈를 줄이기에 부족 — 방법이 아니라 규모"),
      ("RE-EXPERIMENT",C["teal"],"LLM 10K calls 누적","silver 48.8% → +5.7pt (0.5180)"),
      ("RESULT",C["orangeD"],"+ self-training (validation stop)","Example-F1 0.5488 — 논문 NoST(0.5431) 상회"),
    ]
    x=70; w=1340; y=f.top+30; hstep=132
    for i,(tag,col,head,body) in enumerate(steps):
        yy=y+i*hstep
        fill=C["redT"] if col==C["red"] else ("#fff")
        f.rrect(x,yy,w,108,fill,12,col if col in (C["red"],C["orangeD"]) else C["line"], 2 if col in (C["red"],C["orangeD"]) else 1.4)
        f.rrect(x,yy,10,108,col,0)  # status tick handled as rounded left? keep subtle
        f.text(x+34,yy+42,tag,19,700,col)
        f.text(x+34,yy+78,head,21,700,C["ink"])
        f.text(x+540,yy+64,body,18,500,C["sec"])
        if i<len(steps)-1:
            f.arrow(x+w/2,yy+108,x+w/2,yy+hstep,C["grayD"],2.4)
    return f

# ============================================================ 13 COMOVEMENT
def fig_comovement():
    f=Fig("13_precision_f1_comovement",1320,840,
          "Precision–F1 동행 — 병목은 데이터라는 정량 증거",
          "silver precision이 오른 구간에서만 Example-F1이 올랐다 (모델 고정)",
          "출처: README §4·§7 (phase2·3)")
    stages=[s["stage"] for s in D.COMOVE]
    ml,mr,mt,mb=140,180,f.top+50,170; pw=1320-ml-mr; ph=840-mt-mb
    ymin,ymax=0.25,0.55
    def X(i): return ml+ (pw)*i/(len(stages)-1)
    def Y(v): return mt+ph*(1-(v-ymin)/(ymax-ymin))
    for gv in [0.25,0.35,0.45,0.55]:
        f.line(ml,Y(gv),ml+pw,Y(gv),C["grid"],1); f.text(ml-16,Y(gv)+6,f"{gv:.2f}",17,500,C["muted"],"end")
    for i,s in enumerate(stages):
        f.text(X(i),mt+ph+40,s,19,700,C["ink"],"middle")
    prec=[(X(i),Y(s["prec"]/100)) for i,s in enumerate(D.COMOVE)]
    f1=[(X(i),Y(s["f1"])) for i,s in enumerate(D.COMOVE)]
    _polyline(f,prec,C["orange"],mark=8)
    _polyline(f,f1,C["ink"],mark=8)
    for i,s in enumerate(D.COMOVE):
        f.text(prec[i][0],prec[i][1]-20,f'{s["prec"]:.0f}%',18,700,C["orangeD"],"middle")
        f.text(f1[i][0],f1[i][1]+34,f'{s["f1"]:.4f}',18,700,C["ink"],"middle")
    _legend(f,ml,mt-6,[(C["orange"],None,"silver precision (라벨 단위)"),(C["ink"],None,"test Example-F1")])
    f.text(ml+pw,mt+8,"둘 다 상승 · 상관 뚜렷",17,600,C["sec"],"end")
    return f

# ============================================================ 14 LLM BUDGET
def fig_llm_budget():
    f=Fig("14_llm_budget",1440,840,
          "LLM 예산의 임계 규모 — 1K는 무효, 10K는 유효",
          "같은 개입이라도 일정 규모 이상 투입되어야 성능으로 이어진다",
          "출처: README §5.3 (results/phase3_log_llm1k · train_llm9k)")
    def card(x,d,hot):
        w=560
        f.rrect(x,f.top+30,w,300,C["orangeT"] if hot else "#fff",14,C["orange"] if hot else C["line"], 2.4 if hot else 1.4)
        f.text(x+40,f.top+96,f'{d["calls"]} calls',34,700,C["orangeD"] if hot else C["ink"])
        rows=[("코퍼스 커버리지",f'{d["cov"]:.1f}%'),("silver precision",f'+{d["d_prec"]:.1f}pt'),("Example-F1",d["note"])]
        for i,(k,v) in enumerate(rows):
            yy=f.top+150+i*56
            f.text(x+40,yy,k,19,500,C["sec"])
            f.text(x+w-40,yy,v,23,700,(C["orangeD"] if (hot and i==2) else (C["muted"] if (not hot and i==2) else C["ink"])),"end")
            if i<2: f.line(x+40,yy+20,x+w-40,yy+20,C["grid"],1)
    card(70,D.LLM_SCALE["k1"],False)
    card(680,D.LLM_SCALE["k10"],True)
    # delta F1 mini bars
    by=f.top+400; bw=900
    f.text(70,by-6,"Example-F1 개선폭",20,700,C["ink"])
    for i,(lab,v,hot) in enumerate([("1K",0.0,False),("10K",5.7,True)]):
        yy=by+20+i*70
        f.text(70,yy+30,f"{lab} calls",18,600,C["ink"])
        f.rrect(240,yy+8,max(6,bw*v/7.0),42,C["orange"] if hot else C["gray"],6)
        f.text(240+max(6,bw*v/7.0)+14,yy+36,("+%.1fpt"%v) if v else "변화 없음 (0pt)",20,700,C["orangeD"] if hot else C["sec"])
    return f

# ============================================================ 15 SELF-TRAINING STOPPING
def fig_st_stopping():
    f=Fig("15_selftraining_stopping",1540,860,
          "Self-training Stopping — 멈추는 기준이 결과를 가른다",
          "동일 base(0.5180)·동일 알고리즘, stopping만 교체 · validation은 test와 같은 지점에서 최고",
          "출처: results/phase4_st_log.json · phase4_st_val_log.json")
    ml,mr,mt,mb=130,150,f.top+108,170; pw=1540-ml-mr; ph=860-mt-mb
    ymin,ymax=0.49,0.56
    def X(b): return ml+pw*b/800
    def Y(v): return mt+ph*(1-(v-ymin)/(ymax-ymin))
    # legend + top-right note above the plot
    _legend(f,ml,f.top+6,[(C["ink"],None,"고정 실행 test F1 (→ 0.5002)"),(C["orange"],"7 5","validation F1 (→ early stop)")])
    f.text(ml+pw,f.top+14,"test peak 0.5492 미채택 (§5.4)",16.5,600,C["sec"],"end")
    for gv in [0.49,0.51,0.53,0.55]:
        f.line(ml,Y(gv),ml+pw,Y(gv),C["grid"],1); f.text(ml-16,Y(gv)+6,f"{gv:.2f}",17,500,C["muted"],"end")
    for b in [0,200,400,600,800]:
        f.text(X(b),mt+ph+40,f"{b}",17,500,C["sec"],"middle")
    f.text(ml+pw/2,mt+ph+72,"self-training batch",19,600,C["sec"],"middle")
    # best-val vertical line
    f.line(X(200),mt,X(200),mt+ph,C["navy"],1.4,dash="4 5")
    f.text(X(200),mt-14,"best val @ 200",16.5,700,C["navy"],"middle")
    fixed=[(X(r["batch"]),Y(r["test"])) for r in D.ST_FIXED]
    val=[(X(r["batch"]),Y(r["val"])) for r in D.ST_VAL]
    _polyline(f,fixed,C["ink"],mark=7)
    _polyline(f,val,C["orange"],dash="7 5",mark=7)
    # adopted star + annotations (to the right of the peak, no stacking)
    ax,ay=X(200),Y(0.5488)
    f.circle(ax,ay,12,C["orange"],C["orangeD"],3)
    f.text(ax+22,ay-6,"채택 0.5488",19,700,C["orangeD"],"start")
    f.text(X(800)-6,Y(0.5002)+30,"800까지 = 0.5002",17,700,C["ink"],"end")
    f.text(X(800)-6,Y(0.5002)+54,"base(0.518)보다 낮음",15,500,C["sec"],"end")
    return f

# ============================================================ 16 ABLATION TABLE
def fig_ablation_table():
    f=Fig("16_ablation_table",1560,900,
          "누적 개선표 — 각 행은 하나의 가설·검증",
          "test Example-F1 · 데이터 축 개선만으로 모델 축 개선 없이 성능을 견인",
          "출처: README §4")
    rows=[
      ("0","random / majority path","—","0.063 / 0.120","—",False),
      ("1","zero-shot NLI + path (선생)","—","0.4623","—",False),
      ("2","+ classifier (초기 silver30·len256)","30.3%","0.3995","−6.3",False),
      ("3","+ silver 정제 (min_conf0.15)+len512","43.2%","0.4614","+6.2",False),
      ("4","+ LLM refine 1K (과제 예산)","43.8%","0.4571","±0",False),
      ("5","+ LLM refine 10K 누적","48.8%","0.5180","+5.7",False),
      ("6","+ self-training (행3·800 고정)","—","0.4729","+1.2",False),
      ("7","+ self-training (행5·audit-val)","—","0.5488","+3.1",True),
    ]
    x=70; w=1420; y=f.top+20
    cols=[x+26,x+120,x+760,x+980,x+1230]; anch=["start","start","middle","middle","middle"]
    heads=["#","단계","silver prec.","Example-F1","Δ"]
    f.rrect(x,y,w,54,C["ink"],8)
    for c,h,a in zip(cols,heads,anch): f.text(c,y+35,h,19,700,"#fff",a)
    ry=y+54; rh=68
    for i,(num,stage,prec,f1,dlt,hot) in enumerate(rows):
        yy=ry+i*rh
        f.rrect(x,yy,w,rh,(C["orangeT"] if hot else ("#FFFFFF" if i%2 else "#FAF9F6")),0)
        tc=C["orangeD"] if hot else C["ink"]
        f.text(cols[0],yy+43,num,19,700 if hot else 600,tc,"start")
        f.text(cols[1],yy+43,stage,19,700 if hot else 500,tc,"start")
        f.text(cols[2],yy+43,prec,19,600,C["sec"] if not hot else tc,"middle")
        f.text(cols[3],yy+43,f1,21,700,tc,"middle")
        dc=C["red"] if dlt.startswith("−") else (C["orangeD"] if hot else C["teal"])
        f.text(cols[4],yy+43,dlt,20,700,dc if dlt not in("—","±0") else C["muted"],"middle")
    f.line(x,ry,x+w,ry,C["line"],1)
    fy=ry+len(rows)*rh+18
    f.rrect(x,fy,w,64,C["navyT"],10)
    f.text(x+26,fy+40,"논문 참조: Hier-0Shot 0.4742 · TaxoClass-NoST 0.5431 · full(완전지도) 0.5934   |   논문 세팅 준수 최고 = 행 5 (0.5180)",17.5,600,C["navy"])
    return f

# ============================================================ 17 REPRO NOTES
def fig_repro_notes():
    f=Fig("17_repro_notes",1640,900,
          "재현 노트 — 논문과 구현 사이에서 확인한 것들",
          "‘학습은 돌아가되 결과가 조용히 왜곡되는’ 유형 — 수치 불일치 시 모델 탓 전에 논문–구현 diff를 감사",
          "출처: README §6")
    import textwrap
    x=70; w=1500; y=f.top+16
    c0,c1,c2=x+24,x+430,x+1080
    f.rrect(x,y,w,50,C["ink"],8)
    f.text(c0,y+33,"이슈",18,700,"#fff"); f.text(c1,y+33,"확인한 사실",18,700,"#fff"); f.text(c2,y+33,"대응",18,700,"#fff")
    ry=y+50; rh=138
    for i,r in enumerate(D.REPRO):
        yy=ry+i*rh
        f.rrect(x,yy,w,rh,("#FFFFFF" if i%2 else "#FAF9F6"),0)
        f.text(c0,yy+52,r["t"],20,700,C["ink"])
        f.text(c0,yy+84,r["sym"],17,600,C["orangeD"])
        lines=textwrap.wrap(r["find"],34)
        for j,ln in enumerate(lines[:3]):
            f.text(c1,yy+44+j*30,ln,17,500,C["sec"])
        flines=textwrap.wrap(r["fix"],26)
        for j,ln in enumerate(flines[:3]):
            f.text(c2,yy+52+j*30,ln,17.5,600,C["teal"])
        f.line(x,yy,x+w,yy,C["line"],1)
    return f

ALL=[fig_pipeline,fig_problem,fig_eval_protocol,fig_core_mining,fig_classifier_arch,
     fig_path_decoding,fig_metric_masking,fig_overfitting,fig_level_accuracy,fig_prec_cov,
     fig_decision_log,fig_trajectory,fig_comovement,fig_llm_budget,fig_st_stopping,
     fig_ablation_table,fig_repro_notes]

if __name__=="__main__":
    build(ALL)
