#!/usr/bin/env python3
"""Slides 3 & 4 as HTML previews, mirroring the pptx layout exactly (chart PNGs)."""
import os
from theme import CSS, page, chrome

OUT = os.path.join(os.path.dirname(__file__), "out")
CH = "../charts"  # relative to out/
ASPECT = dict(waterfall=3.678, metric_reversal=2.069, prec_coverage=1.802,
              comovement=2.023, llm_scale=1.655, st_trajectory=2.253)

def fit(name, tw, th=None):
    a=ASPECT[name]; w=tw; h=w/a
    if th and h>th: h=th; w=h*a
    return w,h

def img(name, cx, top, tw, th=None):
    w,h=fit(name,tw,th); x=cx-w/2
    return f'<img src="{CH}/{name}.png" style="position:absolute;left:{x:.0f}px;top:{top}px;width:{w:.0f}px;height:{h:.0f}px">'

# ---------------- SLIDE 3 ----------------
CSS3="""
.fcard{ position:absolute; background:var(--card); border:1px solid var(--line); border-radius:14px;
  box-shadow:0 3px 10px rgba(154,149,140,.12); }
.ftitle{ position:absolute; font-size:23px; font-weight:700; color:var(--ink); }
.step{ position:absolute; background:var(--card); border:1px solid var(--line); border-radius:10px; }
.slabel{ position:absolute; font-size:17px; font-weight:700; letter-spacing:.03em; }
.stitle{ position:absolute; font-size:20px; font-weight:700; color:var(--ink); line-height:1.05; }
.sbody{ position:absolute; font-size:16.5px; font-weight:500; color:var(--sec); line-height:1.14; }
.arrow{ position:absolute; font-size:40px; font-weight:700; color:var(--muted); text-align:center; }
.concl{ position:absolute; left:104px; top:900px; font-size:24px; font-weight:600; }
.concl b{ color:var(--orangeDeep); }
"""
def build_p3():
    inner = chrome("PROBLEM SOLVING",
        "잘못 고른 품질 지표가 <span style='color:var(--orangeDeep)'>문제를 가리고</span> 있었다",
        "가설 → 실험 → 실패 → 원인 분석 → 재실험 — 병목을 데이터로 좁혀간 과정",
        pageno="03", footrule=False)
    # figure cards
    inner += f'<div class="fcard" style="left:104px;top:282px;width:928px;height:398px"></div>'
    inner += f'<div class="ftitle" style="left:130px;top:300px">같은 데이터, 평가 지표만 바꿨다</div>'
    inner += img("metric_reversal",568,338,880,332)
    inner += f'<div class="fcard" style="left:1056px;top:282px;width:760px;height:398px"></div>'
    inner += f'<div class="ftitle" style="left:1082px;top:300px">threshold는 감이 아니라 곡선으로 결정했다</div>'
    inner += img("prec_coverage",1436,344,700,320)
    # decision log
    steps=[
        ("PROBLEM","var(--charcoal)","학습한 분류기 0.3995","< zero-shot 0.4623 —<br>학습이 성능을 깎았다"),
        ("가설 1 · 기각","var(--sec)","학습 설정 문제?","batch·seq len·decoder<br>통제 → 모두 1pt 미만"),
        ("가설 2","var(--navy)","데이터 품질 문제?","지표 재정의 → 라벨 단위<br>precision 30.3%"),
        ("원인","var(--charcoal)","지표가 가리고 있었다","문서 단위 71%가<br>과잉 생성을 은폐"),
        ("재실험","var(--orangeDeep)","min_conf 0.15 + len 512","silver 30→43% ·<br>Example-F1 +6.2pt"),
    ]
    n=len(steps); sgap=20; aw=26; sw=(1712-(n-1)*(sgap+aw))/n; sy=716; sh=150
    for i,(lab,col,ti,bo) in enumerate(steps):
        sx=104+i*(sw+sgap+aw)
        inner+=f'<div class="step" style="left:{sx:.0f}px;top:{sy}px;width:{sw:.0f}px;height:{sh}px"></div>'
        inner+=f'<div class="slabel" style="left:{sx+18:.0f}px;top:{sy+16}px;color:{col}">{lab}</div>'
        inner+=f'<div class="stitle" style="left:{sx+18:.0f}px;top:{sy+44}px;width:{sw-34:.0f}px">{ti}</div>'
        inner+=f'<div class="sbody" style="left:{sx+18:.0f}px;top:{sy+92}px;width:{sw-34:.0f}px">{bo}</div>'
        if i<n-1:
            inner+=f'<div class="arrow" style="left:{sx+sw:.0f}px;top:{sy+52}px;width:{aw+sgap}px">›</div>'
    inner+='<div class="concl">평가 지표를 다시 정의하는 순간, <b>보이지 않던 6pt 규모의 문제가 드러났다.</b></div>'
    open(os.path.join(OUT,"slide3.html"),"w").write(page(inner,CSS3))

# ---------------- SLIDE 4 ----------------
CSS4="""
.ctag{ position:absolute; font-size:23px; font-weight:700; color:var(--ink); }
.cpt{ position:absolute; font-size:30px; font-weight:700; color:var(--orangeDeep); text-align:right; }
.ccap{ position:absolute; font-size:19px; font-weight:500; color:var(--ink); line-height:1.14; }
.disc{ position:absolute; left:104px; top:706px; width:1712px; height:296px; background:var(--card);
  border:1px solid var(--line); border-radius:14px; }
.disctitle{ position:absolute; left:136px; top:728px; font-size:22px; font-weight:700; color:var(--teal); }
.discline{ position:absolute; left:136px; width:1650px; font-size:21px; color:var(--sec); line-height:1.3; }
.discline b{ color:var(--ink); } .discline .o{ color:var(--orangeDeep); font-weight:700; }
"""
def build_p4():
    inner = chrome("RESULTS",
        "<span style='color:var(--orangeDeep)'>0.3995 → 0.5488,</span> 각 개입의 기여분을 대조로 분리했다",
        "모든 대조 = 해당 개입만 교체 · 나머지 조건·seed 고정 — Decision Log가 아니라 Contribution Analysis",
        pageno="04", footrule=False)
    cols=[("comovement","귀속 ① 데이터 정제","+6.2pt","silver precision과 F1이 동행 — 병목은 데이터"),
          ("llm_scale","귀속 ② LLM 재검수","+5.7pt","1K은 0pt · 임계 규모 위에서만 효과"),
          ("st_trajectory","귀속 ③ validation stopping","+3.1pt","멈추는 기준이 이득의 절반을 갈랐다")]
    gap=40; cw=(1712-2*gap)/3
    for i,(name,tg,pt,cap) in enumerate(cols):
        cx=104+i*(cw+gap); mid=cx+cw/2
        inner+=f'<div class="ctag" style="left:{cx:.0f}px;top:290px;width:{cw-90:.0f}px">{tg}</div>'
        inner+=f'<div class="cpt" style="left:{cx+cw-176:.0f}px;top:286px;width:176px">{pt}</div>'
        inner+=img(name,mid,356,cw,240)
        inner+=f'<div class="ccap" style="left:{cx:.0f}px;top:612px;width:{cw:.0f}px">{cap}</div>'
    inner+='<div class="disc"></div>'
    inner+='<div class="disctitle">측정 조건 공개</div>'
    inner+='<div class="discline" style="top:772px">· 최종 0.5488의 stopping 신호는 train gold 3,000개를 사용한다(학습 신호로는 미사용). 순수 weak-supervision 세팅 최고치는 <b>0.5180</b>으로 분리 보고.</div>'
    inner+='<div class="discline" style="top:834px">· test 피크 0.5492는 채택하지 않았다 — 결과를 본 뒤 고르면 test 튜닝이 되므로, validation 규칙이 정한 <span class="o">0.5488</span>을 보고.</div>'
    inner+='<div class="discline" style="top:896px">· 참조(논문 보고치): <b>TaxoClass-NoST 0.5431</b> (상회) · full(완전지도) 0.5934 · Hier-0Shot-TC 0.4742</div>'
    open(os.path.join(OUT,"slide4.html"),"w").write(page(inner,CSS4))

if __name__=="__main__":
    build_p3(); build_p4(); print("wrote slide3.html, slide4.html")
