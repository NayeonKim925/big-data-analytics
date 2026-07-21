# Shared design system: CSS tokens + slide chrome + SVG chart helpers.
# Style calibrated to the reference deck (light bg, teal section tags, navy
# eyebrow, clean Pretendard) + the portfolio's own colour rules
# (orange = precision / final-adopted, gray = neutral steps, charcoal = the drop).

W, H = 1920, 1080

CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
:root{
  --page:#F7F6F3; --card:#FFFFFF;
  --ink:#1B2430; --sec:#59616C; --muted:#9AA0A6;
  --line:#E7E3DC; --grid:#ECE9E2;
  --teal:#1E756D; --teal-tint:#E7F0EE;
  --navy:#2B4C7E;
  --orange:#D9822B; --orange-deep:#C06E1C; --orange-tint:#FBEEDC;
  --graybar:#C7CBD1; --graybar-deep:#AAB0B8;
  --charcoal:#333A41;
}
html,body{ background:var(--page); }
.slide{
  width:1920px; height:1080px; position:relative; overflow:hidden;
  background:var(--page); color:var(--ink);
  font-family:Pretendard,'Apple SD Gothic Neo',sans-serif;
  -webkit-font-smoothing:antialiased; text-rendering:geometricPrecision;
}
.tag{ position:absolute; left:104px; top:60px; height:52px; display:flex; align-items:center;
  padding:0 26px; background:var(--teal); color:#fff; font-weight:700; font-size:21px;
  letter-spacing:.14em; }
.toprule{ position:absolute; left:420px; right:104px; top:85px; height:1px; background:var(--line); }
.headline{ position:absolute; left:104px; top:150px; font-size:52px; font-weight:700; letter-spacing:-.01em; line-height:1.08; }
.subtitle{ position:absolute; left:106px; top:224px; font-size:26px; font-weight:500; color:var(--sec); letter-spacing:-.01em; }
.pageno{ position:absolute; right:104px; bottom:44px; font-size:20px; color:var(--muted); font-variant-numeric:tabular-nums; }
.footrule{ position:absolute; left:104px; right:104px; bottom:92px; height:1px; background:var(--line); }
.tnum{ font-variant-numeric:tabular-nums; }
b,strong{ font-weight:700; }
"""

def page(inner, css_extra="", bg="--page"):
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}{css_extra}</style></head>
<body><div class="slide" style="background:var({bg})">{inner}</div></body></html>"""

def chrome(tag, headline, subtitle=None, pageno=None, footrule=True):
    h = f'<div class="tag">{tag}</div><div class="toprule"></div>'
    h += f'<div class="headline">{headline}</div>'
    if subtitle:
        h += f'<div class="subtitle">{subtitle}</div>'
    if footrule:
        h += '<div class="footrule"></div>'
    if pageno:
        h += f'<div class="pageno">{pageno}</div>'
    return h

# ---------- SVG helpers ----------
def _t(x, y, s, size, weight=500, fill="var(--ink)", anchor="start", ls="0", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}" '
            f'font-family="Pretendard,sans-serif" style="font-variant-numeric:tabular-nums" {extra}>{s}</text>')

def _line(x1,y1,x2,y2,stroke,w=1,dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>'

def _rect(x,y,w,h,fill,rx=4,extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" {extra}/>'
