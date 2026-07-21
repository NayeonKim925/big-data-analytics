import PptxGenJS from "pptxgenjs";

// ---- design tokens (hex, no #) ----
const C = {
  page:"F7F6F3", card:"FFFFFF", ink:"1B2430", sec:"59616C", muted:"9AA0A6",
  line:"E7E3DC", teal:"1E756D", tealTint:"E7F0EE", navy:"2B4C7E",
  orange:"D9822B", orangeDeep:"C06E1C", orangeTint:"FBEEDC",
  graybar:"C7CBD1", charcoal:"333A41",
};
const FONT = "Pretendard";
// px(1920x1080) -> inch / pt
const IN = px => +(px/144).toFixed(3);
const PT = px => +(px*0.5).toFixed(1);
const ASPECT = { waterfall:3.678, metric_reversal:2.069, prec_coverage:1.802,
                 comovement:2.023, llm_scale:1.655, st_trajectory:2.253 };

const pptx = new PptxGenJS();
pptx.defineLayoutProps = null;
pptx.layout = "LAYOUT_WIDE"; // 13.333 x 7.5in

function txt(slide, runs, x,y,w,h, o={}) {
  slide.addText(runs, { x:IN(x), y:IN(y), w:IN(w), h:IN(h), margin:0,
    valign:o.valign||"top", align:o.align||"left", fontFace:FONT,
    color:o.color||C.ink, fontSize:o.fontSize||PT(24), bold:o.bold||false,
    charSpacing:o.charSpacing, lineSpacingMultiple:o.lsm, fit:o.fit,
    ...(o.fill?{fill:{color:o.fill}}:{}) });
}
function rect(slide, x,y,w,h, fill, o={}) {
  const type = o.round ? "roundRect" : "rect";
  const opt = { x:IN(x), y:IN(y), w:IN(w), h:IN(h), fill: fill?{color:fill}:{type:"none"} };
  if (o.round) opt.rectRadius = IN(o.round);
  if (o.line) opt.line = { color:o.line, width:o.lineW||1 };
  else opt.line = { type:"none" };
  if (o.shadow) opt.shadow = { type:"outer", color:"9A958C", blur:9, offset:3, angle:90, opacity:0.22 };
  slide.addShape(type, opt);
}
function ellipse(slide, x,y,w,h, fill, o={}) {
  slide.addShape("ellipse", { x:IN(x), y:IN(y), w:IN(w), h:IN(h),
    fill:{color:fill}, line:o.line?{color:o.line,width:o.lineW||1}:{type:"none"} });
}
function lineH(slide, x,y,w, color, wpt=1) {
  slide.addShape("line", { x:IN(x), y:IN(y), w:IN(w), h:0, line:{color, width:wpt} });
}
function chart(slide, name, cx, top, targetW, targetH) {
  const a = ASPECT[name];
  let w = targetW, h = w/a;
  if (targetH && h>targetH) { h=targetH; w=h*a; }
  const x = cx - w/2;
  slide.addImage({ path:"charts/"+name+".png", x:IN(x), y:IN(top), w:IN(w), h:IN(h) });
  return h;
}
function tag(slide, label) {
  const w = label.length*13 + 58, h=52, x=104, y=60;
  rect(slide, x,y,w,h, C.teal);
  txt(slide, [{text:label}], x, y, w, h,
      {color:"FFFFFF", bold:true, fontSize:PT(21), align:"center", valign:"middle", charSpacing:2.4});
  lineH(slide, x+w+30, y+h/2, 1816-(x+w+30), C.line, 1);
}
function pageNo(slide, n){ txt(slide,[{text:n}],1740,996,72,30,{color:C.muted,fontSize:PT(20),align:"right"}); }
function head(slide, runs, sub){
  txt(slide, runs, 104,150,1712,72,{bold:true,fontSize:PT(52),lsm:1.0});
  if(sub) txt(slide,[{text:sub}],106,226,1712,40,{color:C.sec,fontSize:PT(26)});
}

// ============================================================ SLIDE 1 — cover
{
  const s = pptx.addSlide(); s.background = { color: C.page };
  txt(s,[{text:"TAXOCLASS (NAACL 2021) 재현·확장 · AMAZON-531"}],150,296,1300,40,
      {color:C.navy,bold:true,fontSize:PT(24),charSpacing:2.6});
  txt(s,[
    {text:"모델은 그대로,",options:{color:C.ink,breakLine:true}},
    {text:"데이터만 ",options:{color:C.orangeDeep}},
    {text:"고쳤다",options:{color:C.ink}},
  ],148,352,1300,240,{bold:true,fontSize:PT(100),lsm:1.02});
  rect(s,152,616,300,5,C.navy);
  txt(s,[
    {text:"라벨 0개에서 531-클래스 계층 멀티라벨 분류기를 학습하고,",options:{breakLine:true}},
    {text:"성능 병목이 모델이 아니라 ",options:{}},
    {text:"silver label 품질",options:{color:C.ink,bold:true}},
    {text:"임을 실험으로 규명한 기록",options:{}},
  ],150,652,1300,110,{color:C.sec,fontSize:PT(30),lsm:1.42});
  txt(s,[{text:"Nayeon Kim"}],150,806,700,44,{bold:true,fontSize:PT(30)});
  txt(s,[{text:"Weakly-Supervised HMTC · Amazon-531 (train 29,487 / test 19,658) · 개인 100% · GitHub"}],
      150,852,1400,34,{color:C.muted,fontSize:PT(22)});
  // metrics (right)
  txt(s,[{text:"Example-F1 (test)"}],900,300,916,32,{color:C.muted,bold:true,fontSize:PT(22),align:"right"});
  txt(s,[
    {text:"0.3995 ",options:{color:C.ink}},
    {text:"→ ",options:{color:C.muted}},
    {text:"0.5488",options:{color:C.orangeDeep}},
  ],900,338,916,74,{bold:true,fontSize:PT(60),align:"right"});
  txt(s,[
    {text:"zero-shot 대비 +8.7pt · 초기 분류기 대비 +14.9pt",options:{breakLine:true}},
    {text:"논문 TaxoClass-NoST(0.5431) 상회",options:{}},
  ],900,430,916,70,{color:C.sec,fontSize:PT(21),align:"right",lsm:1.35});
  rect(s,0,1024,1920,56,C.navy);
  s.addNotes("표지. Example-F1 0.3995(초기 분류기)→0.5488(최종). zero-shot 0.4623 대비 +8.7pt, 논문 NoST 0.5431 상회. 데이터=Amazon-531(train 29,487/test 19,658), 531 classes, 라벨 0개 설정.");
}

// ============================================================ SLIDE 2 — overview
{
  const s = pptx.addSlide(); s.background = { color: C.page };
  tag(s,"OVERVIEW");
  head(s,[
    {text:"모델이 아니라 ",options:{color:C.ink}},
    {text:"데이터",options:{color:C.orangeDeep}},
    {text:"가 병목이었다",options:{color:C.ink}},
  ],"라벨 0개로 학습한 531-클래스 계층 분류기 — 다섯 지점의 가설·검증 기록");
  // chart card
  rect(s,104,286,1712,452,C.card,{round:14,line:C.line,shadow:true});
  txt(s,[
    {text:"각 막대 = 하나의 가설을 검증한 결과 ",options:{color:C.ink,bold:true}},
    {text:"· Test Example-F1 · seed 고정",options:{color:C.muted,bold:false,fontSize:PT(20)}},
  ],136,306,1600,30,{fontSize:PT(23)});
  chart(s,"waterfall",960,344,1560,384);
  // section label
  txt(s,[{text:"성능을 바꾼 네 가지 판단"}],104,756,700,30,{color:C.teal,bold:true,fontSize:PT(22),charSpacing:0.6});
  // 4 judgment cards
  const cardData=[
    ["1","라벨 구조부터 의심했다","예측 공간 축소","2⁵³¹","568 경로",C.navy],
    ["2","지표부터 다시 측정했다","문서→라벨 단위 precision","71%","30.3%",C.charcoal],
    ["3","LLM은 중요한 문서에만","결정 경계 10K건 재검수","43%","49%",C.orangeDeep],
    ["4","Validation으로 멈췄다","self-training stopping","+1.2pt","+3.1pt",C.orangeDeep],
  ];
  const gap=22, cw=(1712-3*gap)/4, cy=796, ch=150;
  cardData.forEach((d,i)=>{
    const cx=104+i*(cw+gap);
    rect(s,cx,cy,cw,ch,C.card,{round:12,line:C.line});
    ellipse(s,cx+cw-52,cy+16,34,34,C.tealTint);
    txt(s,[{text:d[0]}],cx+cw-52,cy+16,34,34,{color:C.teal,bold:true,fontSize:PT(18),align:"center",valign:"middle"});
    txt(s,[{text:d[1]}],cx+22,cy+18,cw-70,30,{bold:true,fontSize:PT(24)});
    txt(s,[{text:d[2]}],cx+22,cy+54,cw-40,24,{color:C.sec,fontSize:PT(18)});
    txt(s,[
      {text:d[3],options:{strike:true,color:C.muted,fontSize:PT(25)}},
      {text:"  →  ",options:{color:C.muted,fontSize:PT(21)}},
      {text:d[4],options:{bold:true,color:d[5],fontSize:PT(31)}},
    ],cx+22,cy+92,cw-40,44,{});
  });
  // closing
  txt(s,[
    {text:"모델은 그대로. ",options:{color:C.ink,bold:true}},
    {text:"데이터만 고쳤다",options:{color:C.orangeDeep,bold:true}},
    {text:" — 위 다섯 지점이 그 기록이다.",options:{color:C.ink,bold:true}},
  ],104,966,1600,34,{fontSize:PT(24)});
  pageNo(s,"02");
  s.addNotes("개요. 성능 궤적(Test Example-F1): zero-shot 0.4623 → 초기 분류기 0.3995(-6.3pt, silver precision 30%) → 데이터 정제 0.4614(+6.2) → LLM 10K 0.5180(+5.7) → self-training+audit-val 0.5488(+3.1). 논문 NoST 0.5431 상회. 네 가지 판단: 라벨구조 검증(2^531→568경로), 지표 재정의(71%→30.3%), LLM 결정경계 집중(43→49%), validation 기반 stopping(+1.2→+3.1pt).");
}

// ============================================================ SLIDE 3 — problem solving
{
  const s = pptx.addSlide(); s.background = { color: C.page };
  tag(s,"PROBLEM SOLVING");
  head(s,[
    {text:"잘못 고른 품질 지표가 ",options:{color:C.ink}},
    {text:"문제를 가리고",options:{color:C.orangeDeep}},
    {text:" 있었다",options:{color:C.ink}},
  ],"가설 → 실험 → 실패 → 원인 분석 → 재실험 — 병목을 데이터로 좁혀간 과정");
  // two figure cards
  rect(s,104,282,928,398,C.card,{round:14,line:C.line,shadow:true});
  txt(s,[{text:"같은 데이터, 평가 지표만 바꿨다"}],130,300,880,28,{bold:true,fontSize:PT(23)});
  chart(s,"metric_reversal",568,338,880,332);
  rect(s,1056,282,760,398,C.card,{round:14,line:C.line,shadow:true});
  txt(s,[{text:"threshold는 감이 아니라 곡선으로 결정했다"}],1082,300,710,28,{bold:true,fontSize:PT(23)});
  chart(s,"prec_coverage",1436,344,700,320);
  // decision log — 5 native steps
  const steps=[
    ["PROBLEM",C.charcoal,"학습한 분류기 0.3995","< zero-shot 0.4623 — 학습이 성능을 깎았다"],
    ["가설 1 · 기각",C.sec,"학습 설정 문제?","batch·seq len·decoder 통제 → 모두 1pt 미만"],
    ["가설 2",C.navy,"데이터 품질 문제?","지표 재정의 → 라벨 단위 precision 30.3%"],
    ["원인",C.charcoal,"지표가 가리고 있었다","문서 단위 71%가 과잉 생성을 은폐"],
    ["재실험",C.orangeDeep,"min_conf 0.15 + len 512","silver 30→43% · Example-F1 +6.2pt"],
  ];
  const n=steps.length, sgap=20, sw=(1712-(n-1)*(sgap+26))/n, sy=716, sh=150, aw=26;
  steps.forEach((st,i)=>{
    const sx=104+i*(sw+sgap+aw);
    rect(s,sx,sy,sw,sh,C.card,{round:10,line:C.line});
    txt(s,[{text:st[0]}],sx+18,sy+16,sw-30,22,{color:st[1],bold:true,fontSize:PT(17),charSpacing:0.5});
    txt(s,[{text:st[2]}],sx+18,sy+46,sw-32,44,{bold:true,color:C.ink,fontSize:PT(20),lsm:1.05});
    txt(s,[{text:st[3]}],sx+18,sy+94,sw-32,50,{color:C.sec,fontSize:PT(16.5),lsm:1.12});
    if(i<n-1) txt(s,[{text:"›"}],sx+sw+2,sy,aw+sgap,sh,{color:C.muted,bold:true,fontSize:PT(40),align:"center",valign:"middle"});
  });
  txt(s,[
    {text:"평가 지표를 다시 정의하는 순간, ",options:{color:C.ink,bold:true}},
    {text:"보이지 않던 6pt 규모의 문제가 드러났다.",options:{color:C.orangeDeep,bold:true}},
  ],104,900,1600,34,{fontSize:PT(24)});
  pageNo(s,"03");
  s.addNotes("문제 해결. 좌: 초기 품질지표 71%(문서당 정답≥1개)가, 라벨 단위 precision 30.3%(문서당 3.8개 중 2.7개 오답)를 은폐. 우: min_conf sweep의 precision-coverage 곡선에서 무릎점 0.15(precision 43%/coverage 79%) 선택. 로그: 가설1(학습설정) 기각→가설2(데이터)→원인(지표)→재실험(min_conf 0.15+512, +6.2pt).");
}

// ============================================================ SLIDE 4 — results / contribution
{
  const s = pptx.addSlide(); s.background = { color: C.page };
  tag(s,"RESULTS");
  head(s,[
    {text:"0.3995 → 0.5488, ",options:{color:C.orangeDeep}},
    {text:"각 개입의 기여분을 대조로 분리했다",options:{color:C.ink}},
  ],"모든 대조 = 해당 개입만 교체 · 나머지 조건·seed 고정 — Decision Log가 아니라 Contribution Analysis");
  const cols=[
    {name:"comovement", tag:"귀속 ① 데이터 정제", pt:"+6.2pt",
     cap:"silver precision과 F1이 동행 — 병목은 데이터"},
    {name:"llm_scale", tag:"귀속 ② LLM 재검수", pt:"+5.7pt",
     cap:"1K은 0pt · 임계 규모 위에서만 효과"},
    {name:"st_trajectory", tag:"귀속 ③ validation stopping", pt:"+3.1pt",
     cap:"멈추는 기준이 이득의 절반을 갈랐다"},
  ];
  const gap=40, cw=(1712-2*gap)/3, chartH=240, topY=356, capY=612;
  cols.forEach((col,i)=>{
    const cx=104+i*(cw+gap), mid=cx+cw/2;
    txt(s,[{text:col.tag}],cx,290,cw-90,30,{bold:true,fontSize:PT(23)});
    txt(s,[{text:col.pt}],cx+cw-96,286,96,34,{color:C.orangeDeep,bold:true,fontSize:PT(30),align:"right"});
    chart(s,col.name,mid,topY,cw,chartH);
    txt(s,[{text:col.cap}],cx,capY,cw,44,{color:C.ink,fontSize:PT(19),lsm:1.12});
  });
  // disclosure block
  rect(s,104,706,1712,296,C.card,{round:14,line:C.line});
  txt(s,[{text:"측정 조건 공개"}],136,728,700,30,{color:C.teal,bold:true,fontSize:PT(22)});
  txt(s,[
    {text:"· 최종 0.5488의 stopping 신호는 train gold 3,000개를 사용한다(학습 신호로는 미사용). 순수 weak-supervision 세팅 최고치는 ",options:{color:C.sec}},
    {text:"0.5180",options:{color:C.ink,bold:true}},
    {text:"으로 분리 보고.",options:{color:C.sec}},
  ],136,772,1650,60,{fontSize:PT(21),lsm:1.3});
  txt(s,[
    {text:"· test 피크 0.5492는 채택하지 않았다 — 결과를 본 뒤 고르면 test 튜닝이 되므로, validation 규칙이 정한 ",options:{color:C.sec}},
    {text:"0.5488",options:{color:C.orangeDeep,bold:true}},
    {text:"을 보고.",options:{color:C.sec}},
  ],136,834,1650,60,{fontSize:PT(21),lsm:1.3});
  txt(s,[
    {text:"· 참조(논문 보고치): ",options:{color:C.sec}},
    {text:"TaxoClass-NoST 0.5431",options:{color:C.ink,bold:true}},
    {text:" (상회) · full(완전지도) 0.5934 · Hier-0Shot-TC 0.4742",options:{color:C.sec}},
  ],136,896,1650,40,{fontSize:PT(21)});
  pageNo(s,"04");
  s.addNotes("기여 분석. ①데이터 정제 +6.2pt: silver precision(30→43→49%)과 Example-F1(0.40→0.46→0.52)의 동행. ②LLM 규모 +5.7pt: 1K(커버리지 3.2%)=0pt vs 10K(약31%)=+5.7pt, 임계 규모. ③validation stopping +3.1pt: 고정 800배치는 0.5002로 하락, val이 test와 같은 batch200에서 최고 → early stop 0.5488. 측정조건: audit-val은 train gold 3000 사용(순수 WS 최고치 0.5180 별도).");
}

await pptx.writeFile({ fileName: "portfolio.pptx" });
console.log("wrote portfolio.pptx");
