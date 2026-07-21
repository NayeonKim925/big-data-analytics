#!/usr/bin/env python3
import os
from theme import CSS, page, chrome, _t, _line, _rect, W, H
import data as D

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

ROLE_FILL = dict(base="var(--graybar-deep)", drop="var(--charcoal)",
                 data="var(--graybar)", final="var(--orange)")
ROLE_LBL  = dict(base="var(--sec)", drop="var(--charcoal)",
                 data="var(--ink)", final="var(--orange-deep)")

def hero_waterfall():
    vw, vh = 1712, 400
    ml, mr, mt, mb = 74, 150, 78, 96
    pw, ph = vw-ml-mr, vh-mt-mb
    ymax = 0.60
    def Y(v): return mt + ph*(1 - v/ymax)
    n = len(D.JOURNEY)
    slot = pw/n; barw = 150
    def BX(i): return ml + slot*i + (slot-barw)/2

    s = [f'<svg viewBox="0 0 {vw} {vh}" width="100%" style="display:block">']
    # gridlines
    for gv in [0,0.15,0.30,0.45,0.60]:
        y = Y(gv)
        s.append(_line(ml, y, ml+pw, y, "var(--grid)", 1))
        s.append(_t(ml-14, y+6, f"{gv:.2f}", 19, 500, "var(--muted)", "end"))
    # NoST reference line
    yn = Y(D.NOST)
    s.append(_line(ml, yn, ml+pw, yn, "var(--navy)", 1.5, dash="7 6"))
    s.append(_t(ml+pw+12, yn-6, "논문 NoST", 19, 700, "var(--navy)"))
    s.append(_t(ml+pw+12, yn+18, f"{D.NOST:.4f}", 19, 600, "var(--navy)"))

    tops = []
    for i,b in enumerate(D.JOURNEY):
        x = BX(i); y = Y(b["f1"]); h = (mt+ph)-y
        tops.append((x+barw/2, y))
        s.append(_rect(x, y, barw, h, ROLE_FILL[b["role"]], rx=5))
        # value label
        s.append(_t(x+barw/2, y-16, f'{b["f1"]:.4f}', 30, 700, ROLE_LBL[b["role"]], "middle"))
        # x labels (two lines) + sub
        l1, l2 = b["label"].split("\n")
        s.append(_t(x+barw/2, mt+ph+34, l1, 21, 700, "var(--ink)", "middle"))
        s.append(_t(x+barw/2, mt+ph+58, l2, 19, 500, "var(--sec)", "middle"))
        if b["sub"]:
            s.append(_t(x+barw/2, mt+ph+82, b["sub"], 18, 600,
                        "var(--orange-deep)" if b["role"]=="final" else "var(--muted)", "middle"))
    # delta chips between bars
    for i in range(1,n):
        b = D.JOURNEY[i]
        if b["delta"] is None: continue
        x0,y0 = tops[i-1]; x1,y1 = tops[i]
        mx = (x0+x1)/2; my = min(y0,y1) - 46
        up = b["delta"] > 0
        col = "var(--orange-deep)" if b["role"]=="final" else ("var(--charcoal)" if not up else "var(--sec)")
        txt = f'{"▲ +" if up else "▼ −"}{abs(b["delta"]):.1f}pt'
        pill_w = 118
        s.append(_rect(mx-pill_w/2, my-24, pill_w, 40,
                       "var(--orange-tint)" if b["role"]=="final" else "#FFFFFF",
                       rx=20, extra=f'stroke="{col}" stroke-width="1.3"'))
        s.append(_t(mx, my+2, txt, 22, 700, col, "middle"))
    s.append('</svg>')
    return "".join(s)

def judgment_cards():
    cards = [
        ("1","라벨 구조부터 의심했다","예측 공간 축소","2⁵³¹","568 경로","var(--navy)"),
        ("2","지표부터 다시 측정했다","문서→라벨 단위 precision","71%","30.3%","var(--charcoal)"),
        ("3","LLM은 중요한 문서에만","결정 경계 10K건 재검수","43%","49%","var(--orange-deep)"),
        ("4","Validation으로 멈췄다","self-training stopping","+1.2pt","+3.1pt","var(--orange-deep)"),
    ]
    out = ['<div class="cards">']
    for num,title,sub,a,b,col in cards:
        out.append(f'''<div class="jcard">
          <div class="jnum">{num}</div>
          <div class="jtitle">{title}</div>
          <div class="jsub">{sub}</div>
          <div class="jtrans"><span class="jfrom">{a}</span><span class="jarrow">→</span><span class="jto" style="color:{col}">{b}</span></div>
        </div>''')
    out.append('</div>')
    return "".join(out)

CSS2 = """
.chartcard{ position:absolute; left:104px; top:262px; width:1712px; background:var(--card);
  border:1px solid var(--line); border-radius:14px; padding:16px 26px 6px; }
.charttitle{ font-size:23px; font-weight:700; color:var(--ink); margin-bottom:0; }
.charttitle .muted{ color:var(--muted); font-weight:600; font-size:20px; }
.sectlabel{ position:absolute; left:104px; top:726px; font-size:22px; font-weight:700; color:var(--teal); letter-spacing:.02em; }
.cards{ position:absolute; left:104px; top:762px; width:1712px; display:grid; grid-template-columns:repeat(4,1fr); gap:22px; }
.jcard{ background:var(--card); border:1px solid var(--line); border-top:3px solid var(--teal); border-radius:12px; padding:18px 22px 20px; position:relative; }
.jnum{ position:absolute; right:20px; top:16px; width:32px; height:32px; border-radius:50%; background:var(--teal-tint);
  color:var(--teal); font-weight:700; font-size:18px; display:flex; align-items:center; justify-content:center; }
.jtitle{ font-size:24px; font-weight:700; color:var(--ink); letter-spacing:-.01em; margin-bottom:5px; padding-right:40px; }
.jsub{ font-size:18px; font-weight:500; color:var(--sec); margin-bottom:14px; }
.jtrans{ display:flex; align-items:baseline; gap:12px; font-variant-numeric:tabular-nums; }
.jfrom{ font-size:25px; font-weight:600; color:var(--muted); text-decoration:line-through; text-decoration-color:#CDD1D6; }
.jarrow{ font-size:21px; color:var(--muted); }
.jto{ font-size:31px; font-weight:700; }
.closing{ position:absolute; left:104px; top:944px; font-size:24px; font-weight:600; color:var(--ink); }
.closing b{ color:var(--orange-deep); }
"""

def build_p2():
    inner = chrome("OVERVIEW",
                   "모델이 아니라 <span style='color:var(--orange-deep)'>데이터</span>가 병목이었다",
                   "라벨 0개로 학습한 531-클래스 계층 분류기 — 다섯 지점의 가설·검증 기록",
                   pageno="02", footrule=False)
    inner += f'''<div class="chartcard">
      <div class="charttitle">각 막대 = 하나의 가설을 검증한 결과 <span class="muted">· Test Example-F1 · seed 고정</span></div>
      {hero_waterfall()}
    </div>'''
    inner += '<div class="sectlabel">성능을 바꾼 네 가지 판단</div>'
    inner += judgment_cards()
    inner += '<div class="closing">모델은 그대로. <b>데이터만 고쳤다</b> — 위 다섯 지점이 그 기록이다.</div>'
    html = page(inner, CSS2)
    open(os.path.join(OUT,"slide2.html"),"w").write(html)

# ---------------- Page 1 : cover ----------------
CSS1 = """
.cv-eyebrow{ position:absolute; left:150px; top:300px; font-size:24px; font-weight:700; color:var(--navy); letter-spacing:.16em; }
.cv-title{ position:absolute; left:148px; top:352px; font-size:104px; font-weight:700; line-height:1.04; letter-spacing:-.02em; color:var(--ink); }
.cv-title b{ color:var(--orange-deep); }
.cv-rule{ position:absolute; left:152px; top:610px; width:340px; height:4px; background:var(--navy); }
.cv-sub{ position:absolute; left:150px; top:648px; width:1200px; font-size:30px; font-weight:500; color:var(--sec); line-height:1.5; letter-spacing:-.01em; }
.cv-name{ position:absolute; left:150px; top:800px; font-size:30px; font-weight:700; color:var(--ink); }
.cv-meta{ position:absolute; left:150px; top:844px; font-size:23px; font-weight:500; color:var(--muted); }
.cv-bar{ position:absolute; left:0; right:0; bottom:0; height:56px; background:var(--navy); }
.cv-metrics{ position:absolute; right:150px; top:300px; width:520px; text-align:right; }
.cv-mrow{ display:flex; justify-content:flex-end; align-items:baseline; gap:16px; margin-bottom:6px; }
.cv-mval{ font-size:56px; font-weight:700; color:var(--ink); font-variant-numeric:tabular-nums; }
.cv-mval .up{ color:var(--orange-deep); }
.cv-mlbl{ font-size:21px; font-weight:600; color:var(--muted); }
.cv-mnote{ text-align:right; font-size:20px; color:var(--sec); margin-top:14px; font-weight:500; }
.cv-chip{ position:absolute; right:150px; top:470px; }
"""
def build_p1():
    inner = f'''
    <div class="cv-eyebrow">TAXOCLASS (NAACL 2021) 재현·확장 · AMAZON-531</div>
    <div class="cv-title">모델은 그대로,<br><b>데이터만</b> 고쳤다</div>
    <div class="cv-rule"></div>
    <div class="cv-sub">라벨 0개에서 531-클래스 계층 멀티라벨 분류기를 학습하고,<br>성능 병목이 모델이 아니라 <b style="color:var(--ink)">silver label 품질</b>임을 실험으로 규명한 기록</div>
    <div class="cv-name">Nayeon Kim</div>
    <div class="cv-meta">Weakly-Supervised HMTC · Amazon-531 (train 29,487 / test 19,658) · 개인 100% · GitHub</div>
    <div class="cv-metrics">
      <div class="cv-mrow"><span class="cv-mlbl">Example-F1 (test)</span><span class="cv-mval">0.3995 → <span class="up">0.5488</span></span></div>
      <div class="cv-mnote">zero-shot 대비 +8.7pt · 초기 분류기 대비 +14.9pt<br>논문 TaxoClass-NoST(0.5431) 상회</div>
    </div>
    <div class="cv-bar"></div>
    '''
    html = page(inner, CSS1)
    open(os.path.join(OUT,"slide1.html"),"w").write(html)

if __name__ == "__main__":
    build_p1(); build_p2()
    print("wrote slide1.html, slide2.html")
