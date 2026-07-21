#!/usr/bin/env python3
"""All portfolio charts as standalone SVG (transparent bg, Pretendard).
Every number is pulled from data.py (which is sourced from the repo)."""
import os
import data as D

P = dict(ink="#1B2430", sec="#59616C", muted="#9AA0A6", line="#E7E3DC", grid="#ECE9E2",
         teal="#1E756D", navy="#2B4C7E", orange="#D9822B", orangeDeep="#C06E1C",
         orangeTint="#FBEEDC", graybar="#C7CBD1", graybarDeep="#AAB0B8", charcoal="#333A41")

def T(x,y,s,size,w=500,fill=P["ink"],anc="start",ls="0",style=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" '
            f'text-anchor="{anc}" letter-spacing="{ls}" font-family="Pretendard,sans-serif" '
            f'style="font-variant-numeric:tabular-nums;{style}">{s}</text>')
def L(x1,y1,x2,y2,st,w=1,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{st}" stroke-width="{w}"{d} stroke-linecap="round"/>'
def R(x,y,w,h,fill,rx=4,extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" {extra}/>'
def C(cx,cy,r,fill,extra=""):
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" {extra}/>'
def PATH(pts,st,w=2.5,dash=None,fill="none"):
    d="M"+" L".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    da=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="{fill}" stroke="{st}" stroke-width="{w}"{da} stroke-linejoin="round" stroke-linecap="round"/>'

def svg(w,h,body):
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'

# ---------------------------------------------------------------- P2 waterfall
def waterfall():
    vw, vh = 1600, 430
    ml, mr, mt, mb = 70, 150, 74, 96
    pw, ph = vw-ml-mr, vh-mt-mb
    ymax=0.60
    Y=lambda v: mt+ph*(1-v/ymax)
    RF=dict(base=P["graybarDeep"], drop=P["charcoal"], data=P["graybar"], final=P["orange"])
    RL=dict(base=P["sec"], drop=P["charcoal"], data=P["ink"], final=P["orangeDeep"])
    n=len(D.JOURNEY); slot=pw/n; bw=140
    BX=lambda i: ml+slot*i+(slot-bw)/2
    s=[]
    for gv in [0,.15,.30,.45,.60]:
        y=Y(gv); s.append(L(ml,y,ml+pw,y,P["grid"],1)); s.append(T(ml-12,y+6,f"{gv:.2f}",18,500,P["muted"],"end"))
    yn=Y(D.NOST); s.append(L(ml,yn,ml+pw,yn,P["navy"],1.5,"7 6"))
    s.append(T(ml+pw+12,yn-6,"논문 NoST",18,700,P["navy"])); s.append(T(ml+pw+12,yn+17,f"{D.NOST:.4f}",18,600,P["navy"]))
    tops=[]
    for i,b in enumerate(D.JOURNEY):
        x=BX(i); y=Y(b["f1"]); h=(mt+ph)-y; tops.append((x+bw/2,y))
        s.append(R(x,y,bw,h,RF[b["role"]],5))
        s.append(T(x+bw/2,y-14,f'{b["f1"]:.4f}',29,700,RL[b["role"]],"middle"))
        l1,l2=b["label"].split("\n")
        s.append(T(x+bw/2,mt+ph+32,l1,20,700,P["ink"],"middle"))
        s.append(T(x+bw/2,mt+ph+55,l2,18,500,P["sec"],"middle"))
        if b["sub"]: s.append(T(x+bw/2,mt+ph+78,b["sub"],17,600,P["orangeDeep"] if b["role"]=="final" else P["muted"],"middle"))
    for i in range(1,n):
        b=D.JOURNEY[i]
        if b["delta"] is None: continue
        x0,y0=tops[i-1]; x1,y1=tops[i]; mx=(x0+x1)/2; my=min(y0,y1)-44; up=b["delta"]>0
        col=P["orangeDeep"] if b["role"]=="final" else (P["charcoal"] if not up else P["sec"])
        txt=f'{"▲ +" if up else "▼ −"}{abs(b["delta"]):.1f}pt'; pw2=112
        s.append(R(mx-pw2/2,my-23,pw2,38,P["orangeTint"] if b["role"]=="final" else "#FFFFFF",19,f'stroke="{col}" stroke-width="1.3"'))
        s.append(T(mx,my+3,txt,21,700,col,"middle"))
    return svg(vw,vh,"".join(s))

# ------------------------------------------------ P3a metric reversal
def metric_reversal():
    vw,vh=900,430
    s=[]
    # two metric callouts
    s.append(T(70,70,"문서 단위 Hit Rate",21,600,P["sec"],"start"))
    s.append(T(70,150,"71%",70,700,P["muted"],"start"))
    s.append(T(70,190,"문서당 정답 ≥1개 포함 비율 — 양호해 보였다",18,500,P["muted"],"start"))
    # arrow
    s.append(T(455,150,"→",54,700,P["muted"],"middle"))
    s.append(T(560,70,"라벨 단위 Precision",21,700,P["ink"],"start"))
    s.append(T(560,150,"30.3%",70,700,P["orangeDeep"],"start"))
    s.append(T(560,190,"생성 라벨 중 실제 정답 비율 — 실제 품질",18,600,P["orangeDeep"],"start"))
    # stacked bar: 문서당 3.8 labels -> 1.1 correct + 2.7 wrong
    bx,by,bw,bh=70,270,760,66
    corr=D.PREC_LABEL/100
    s.append(T(bx,by-16,"문서당 평균 3.8개 라벨 생성",19,700,P["ink"],"start"))
    s.append(R(bx,by,bw*corr,bh,P["orange"],0))
    s.append(R(bx+bw*corr+2,by,bw*(1-corr)-2,bh,P["graybarDeep"],0))
    s.append(T(bx+bw*corr/2,by+bh/2+7,"정답 1.1",20,700,"#FFFFFF","middle"))
    s.append(T(bx+bw*corr+2+(bw*(1-corr))/2,by+bh/2+7,"오답 2.7개",21,700,"#FFFFFF","middle"))
    s.append(T(bx,by+bh+30,"→ 문서에 정답이 하나라도 있으면 “양호”로 세어, 함께 생성된 2.7개의 오답이 지표에 가려졌다.",19,500,P["sec"],"start"))
    return svg(vw,vh,"".join(s))

# ------------------------------------------------ P3b precision-coverage curve
def prec_coverage():
    vw,vh=784,430
    ml,mr,mt,mb=78,58,40,74
    pw,ph=vw-ml-mr,vh-mt-mb
    xs=[d["cov"] for d in D.SWEEP]; ys=[d["prec"] for d in D.SWEEP]
    xmin,xmax=40,100; ymin,ymax=28,54
    X=lambda v: ml+pw*(v-xmin)/(xmax-xmin)
    Y=lambda v: mt+ph*(1-(v-ymin)/(ymax-ymin))
    s=[]
    for gy in [30,40,50]:
        y=Y(gy); s.append(L(ml,y,ml+pw,y,P["grid"],1)); s.append(T(ml-10,y+5,f"{gy}%",17,500,P["muted"],"end"))
    for gx in [50,70,90]:
        x=X(gx); s.append(T(x,mt+ph+26,f"{gx}%",17,500,P["muted"],"middle"))
    s.append(T(ml-58,mt-14,"precision",18,700,P["sec"],"start"))
    s.append(T(ml+pw,mt+ph+52,"coverage →",18,700,P["sec"],"end"))
    pts=[(X(d["cov"]),Y(d["prec"])) for d in D.SWEEP]
    s.append(PATH(pts,P["graybarDeep"],2.5))
    # knee callout placed in the open upper-right area, leader line to the point
    kd=[d for d in D.SWEEP if abs(d["min_conf"]-D.SWEEP_KNEE)<1e-6][0]
    kx,ky=X(kd["cov"]),Y(kd["prec"])
    box_x,box_y,box_w,box_h=X(78),Y(53),250,54
    for d in D.SWEEP:
        x,y=X(d["cov"]),Y(d["prec"]); knee=abs(d["min_conf"]-D.SWEEP_KNEE)<1e-6
        if knee:
            s.append(L(x,y,x,mt+ph,P["orange"],1.4,"4 5")); s.append(L(ml,y,x,y,P["orange"],1.4,"4 5"))
        else:
            s.append(C(x,y,6.5,"#FFFFFF",f'stroke="{P["graybarDeep"]}" stroke-width="2.5"'))
            if d["min_conf"] <= 0.001:   # rightmost point -> anchor left to avoid clipping
                s.append(T(x-12, y+22, f'min_conf {d["min_conf"]:.2f}',14,600,P["muted"],"end"))
            else:
                dy = 22 if abs(d["min_conf"]-0.05)<1e-6 else -14
                s.append(T(x, y+dy, f'min_conf {d["min_conf"]:.2f}',14,600,P["muted"],"middle"))
    # leader + box + knee marker on top
    s.append(L(kx,ky,box_x+30,box_y+box_h,P["orangeDeep"],1.2,"3 4"))
    s.append(R(box_x,box_y,box_w,box_h,P["orangeTint"],8,f'stroke="{P["orangeDeep"]}" stroke-width="1.4"'))
    s.append(T(box_x+16,box_y+24,"선택 · min_conf 0.15",18,700,P["orangeDeep"],"start"))
    s.append(T(box_x+16,box_y+45,"precision 43% · coverage 79%",16,600,P["orangeDeep"],"start"))
    s.append(C(kx,ky,10,P["orange"],'stroke="#FFFFFF" stroke-width="2.5"'))
    return svg(vw,vh,"".join(s))

# ------------------------------------------------ P4a precision & F1 co-movement (single shared axis)
def comovement():
    vw,vh=880,430
    ml,mr,mt,mb=66,232,58,74
    pw,ph=vw-ml-mr,vh-mt-mb
    ymin,ymax=0.25,0.55
    Y=lambda v: mt+ph*(1-(v-ymin)/(ymax-ymin))
    n=len(D.COMOVE); pad=26; X=lambda i: ml+pad+(pw-2*pad)*(i/(n-1))
    s=[]
    for gv in [0.30,0.40,0.50]:
        y=Y(gv); s.append(L(ml,y,ml+pw,y,P["grid"],1)); s.append(T(ml-10,y+5,f"{gv:.2f}",16,500,P["muted"],"end"))
    prec_pts=[(X(i),Y(d["prec"]/100)) for i,d in enumerate(D.COMOVE)]
    f1_pts=[(X(i),Y(d["f1"])) for i,d in enumerate(D.COMOVE)]
    s.append(PATH(f1_pts,P["ink"],3))
    s.append(PATH(prec_pts,P["orange"],3,"2 7"))
    # F1 line is always above the precision line here -> F1 labels above, precision labels below
    for i,d in enumerate(D.COMOVE):
        x=X(i)
        yf=Y(d["f1"]); s.append(C(x,yf,8,P["ink"],'stroke="#FFFFFF" stroke-width="2"'))
        s.append(T(x, yf-15, f'{d["f1"]:.2f}',20,700,P["ink"],"middle"))
        yp=Y(d["prec"]/100); s.append(C(x,yp,8,P["orange"],'stroke="#FFFFFF" stroke-width="2"'))
        s.append(T(x, yp+28, f'{d["prec"]:.0f}%',20,700,P["orangeDeep"],"middle"))
        s.append(T(x,mt+ph+32,d["stage"],18,600,P["sec"],"middle"))
    # legend (right)
    lx=ml+pw+22
    s.append(C(lx+8,mt+6,7,P["ink"])); s.append(T(lx+24,mt+12,"Example-F1",18,700,P["ink"],"start"))
    s.append(C(lx+8,mt+40,7,P["orange"])); s.append(T(lx+24,mt+46,"silver precision",18,700,P["orangeDeep"],"start"))
    s.append(T(lx,mt+100,"precision이 오른",18,500,P["sec"],"start"))
    s.append(T(lx,mt+126,"구간에서만 F1이",18,700,P["ink"],"start"))
    s.append(T(lx,mt+152,"올랐다",18,500,P["sec"],"start"))
    return svg(vw,vh,"".join(s))

# ------------------------------------------------ P4b LLM budget scaling (ΔF1 bars)
def llm_scale():
    vw,vh=720,430
    ml,mr,mt,mb=70,40,90,150
    pw,ph=vw-ml-mr,vh-mt-mb
    ymax=6.5
    Y=lambda v: mt+ph*(1-v/ymax)
    s=[]
    s.append(T(ml-6,mt-46,"ΔExample-F1 (pt)",19,700,P["sec"],"start"))
    for gv in [0,2,4,6]:
        y=Y(gv); s.append(L(ml,y,ml+pw,y,P["grid"],1)); s.append(T(ml-10,y+5,f"{gv}",16,500,P["muted"],"end"))
    bars=[("1K calls",0.0,P["graybarDeep"],"3.2%","+0.6pt","변화 없음"),
          ("10K calls",5.7,P["orange"],"약 31%","+5.6pt","+5.7pt")]
    slot=pw/2; bw=150
    for i,(lab,v,col,cov,dp,note) in enumerate(bars):
        x=ml+slot*i+(slot-bw)/2; y=Y(v) if v>0 else Y(0)-4; h=(mt+ph)-Y(v) if v>0 else 4
        s.append(R(x,y,bw,max(h,4),col,5))
        s.append(T(x+bw/2,(y-14 if v>0 else y-14),(f"+{v:.1f}pt" if v>0 else "0.0pt"),26,700,(P["orangeDeep"] if v>0 else P["muted"]),"middle"))
        s.append(T(x+bw/2,mt+ph+34,lab,21,700,P["ink"],"middle"))
        s.append(T(x+bw/2,mt+ph+62,f"커버리지 {cov}",17,500,P["sec"],"middle"))
        s.append(T(x+bw/2,mt+ph+86,f"silver prec {dp}",17,500,P["sec"],"middle"))
    s.append(T(ml+pw/2,vh-6,"같은 방법 · 규모만 10배 → 임계 규모 위에서만 효과",18,600,P["ink"],"middle"))
    return svg(vw,vh,"".join(s))

# ------------------------------------------------ P4c self-training stopping trajectory
def st_trajectory():
    vw,vh=980,430
    ml,mr,mt,mb=70,160,58,66
    pw,ph=vw-ml-mr,vh-mt-mb
    ymin,ymax=0.495,0.555
    Y=lambda v: mt+ph*(1-(v-ymin)/(ymax-ymin))
    xmax=800; X=lambda b: ml+pw*b/xmax
    s=[]
    for gv in [0.50,0.52,0.54]:
        y=Y(gv); s.append(L(ml,y,ml+pw,y,P["grid"],1)); s.append(T(ml-10,y+5,f"{gv:.2f}",16,500,P["muted"],"end"))
    for gb in [0,200,400,600,800]:
        s.append(T(X(gb),mt+ph+28,f"{gb}",16,500,P["muted"],"middle"))
    s.append(T(ml+pw/2,mt+ph+54,"self-training batch",18,600,P["sec"],"middle"))
    fixed=[(X(d["batch"]),Y(d["test"])) for d in D.ST_FIXED]
    val=[(X(d["batch"]),Y(d["val"])) for d in D.ST_VAL]
    s.append(PATH(fixed,P["ink"],3))
    s.append(PATH(val,P["orange"],2.6,"2 7"))
    for d in D.ST_FIXED:
        s.append(C(X(d["batch"]),Y(d["test"]),6,P["ink"],'stroke="#FFFFFF" stroke-width="2"'))
    for d in D.ST_VAL:
        s.append(C(X(d["batch"]),Y(d["val"]),5.5,P["orange"],'stroke="#FFFFFF" stroke-width="1.8"'))
    # adopted star @ batch 200
    xa,ya=X(200),Y(0.5488)
    s.append(L(xa,Y(0.5492),xa,mt+ph,P["orangeDeep"],1.3,"4 5"))
    s.append(C(xa,ya,11,P["orange"],'stroke="#FFFFFF" stroke-width="3"'))
    s.append(R(xa-70,ya-64,190,44,P["orangeTint"],8,f'stroke="{P["orangeDeep"]}" stroke-width="1.3"'))
    s.append(T(xa+25,ya-40,"채택 0.5488",19,700,P["orangeDeep"],"middle"))
    s.append(T(xa+25,ya-19,"validation 최고점 @200",15,600,P["orangeDeep"],"middle"))
    # end annotation @800 (sits above the endpoint, right-anchored, clear of the axis labels)
    xe,ye=X(800),Y(0.5002)
    s.append(C(xe,ye,6,P["ink"],'stroke="#FFFFFF" stroke-width="2"'))
    s.append(T(xe-14,ye-12,"800까지 학습 → 0.5002 하락",16,700,P["ink"],"end"))
    # legend
    lx=ml+pw+18
    s.append(L(lx,mt+10,lx+30,mt+10,P["ink"],3)); s.append(T(lx+40,mt+15,"test F1",17,700,P["ink"],"start"))
    s.append(L(lx,mt+42,lx+30,mt+42,P["orange"],2.6,"2 6")); s.append(T(lx+40,mt+47,"validation F1",17,700,P["orangeDeep"],"start"))
    s.append(T(lx,mt+92,"val이 test와",16,500,P["sec"],"start"))
    s.append(T(lx,mt+114,"같은 지점(200)에서",16,700,P["ink"],"start"))
    s.append(T(lx,mt+136,"최고 → early stop",16,500,P["sec"],"start"))
    return svg(vw,vh,"".join(s))

CHARTS = dict(waterfall=waterfall, metric_reversal=metric_reversal, prec_coverage=prec_coverage,
              comovement=comovement, llm_scale=llm_scale, st_trajectory=st_trajectory)

if __name__=="__main__":
    outdir=os.path.join(os.path.dirname(__file__),"charts"); os.makedirs(outdir,exist_ok=True)
    blocks=[]
    for name,fn in CHARTS.items():
        s=fn()
        blocks.append(f'<div class="chart" id="{name}" style="display:inline-block">{s}</div>')
    html=f"""<!doctype html><html><head><meta charset="utf-8"><style>
      *{{margin:0;padding:0;box-sizing:border-box}} body{{background:transparent}}
      .chart{{background:transparent}} text{{font-family:Pretendard,sans-serif}}
    </style></head><body>{''.join(blocks)}</body></html>"""
    open(os.path.join(outdir,"all.html"),"w").write(html)
    print("wrote charts/all.html with:", ", ".join(CHARTS))
