#!/usr/bin/env python3
"""Standalone SVG figure framework — embedded Pretendard subset, shared design
tokens (identical to the portfolio PPT). Each figure is authored once as SVG and
also rendered to high-res PNG."""
import os, io, base64, html
from fontTools import subset
from fontTools.ttLib import TTFont

FONT_DIR = os.path.expanduser("~/.fonts")
WEIGHTS = [("400","Pretendard-Regular.otf"),("500","Pretendard-Medium.otf"),
           ("600","Pretendard-SemiBold.otf"),("700","Pretendard-Bold.otf")]

# ---- design tokens (consistent with the PPT deck) ----
C = dict(
  ink="#1B2430", sec="#59616C", muted="#9AA0A6",
  line="#E7E3DC", grid="#ECE9E2", card="#FFFFFF", page="#F7F6F3",
  teal="#1E756D", tealT="#E7F0EE",
  navy="#2B4C7E", navyT="#EAF0F7",
  orange="#D9822B", orangeD="#C06E1C", orangeT="#FBEEDC",
  gray="#C7CBD1", grayD="#A7ADB5", grayT="#F0EEE9", charcoal="#333A41",
  green="#2E8B57", greenT="#E4F1E9", red="#C0453B", redT="#F7E7E5",
)

_GLYPHS = set()
def _rec(s):
    _GLYPHS.update(str(s))
    return html.escape(str(s), quote=True)

def build_font_css():
    text = "".join(sorted(_GLYPHS)) + "0123456789.,%+-×·—→↓↑▲▼✓✗()[]{}<>=/:'\" "
    blocks = []
    for weight, fn in WEIGHTS:
        path = os.path.join(FONT_DIR, fn)
        font = TTFont(path)
        ss = subset.Subsetter(subset.Options(layout_features="*", notdef_outline=True))
        ss.populate(text=text)
        ss.subset(font)
        buf = io.BytesIO(); font.flavor = "woff2"; font.save(buf)
        b64 = base64.b64encode(buf.getvalue()).decode()
        blocks.append(
          f"@font-face{{font-family:'Pretendard';font-style:normal;font-weight:{weight};"
          f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}")
    return "\n".join(blocks)

class Fig:
    FONT = "'Pretendard','Apple SD Gothic Neo',sans-serif"
    def __init__(self, key, w, h, title=None, subtitle=None, source=None, pad=52, bg="#FFFFFF"):
        self.key=key; self.w=w; self.h=h; self.pad=pad; self.bg=bg
        self.b=[]  # body svg fragments
        self.title=title; self.subtitle=subtitle; self.source=source
        self.top = pad  # content starts here; grows with title block
        if title:
            self.b.append(self._text(pad, pad+34, title, 31, 700, C["ink"]))
            self.top = pad+34
            if subtitle:
                self.b.append(self._text(pad, pad+70, subtitle, 20.5, 500, C["sec"]))
                self.top = pad+70
            self.top += 30

    # ---- primitives ----
    def _text(self, x,y,s,size,weight=500,fill="#1B2430",anchor="start",ls="0",italic=False,op=None):
        st=f' font-style="italic"' if italic else ""
        o=f' opacity="{op}"' if op is not None else ""
        return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
                f'fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}"{st}{o} '
                f'style="font-variant-numeric:tabular-nums">{_rec(s)}</text>')
    def text(self,*a,**k): self.b.append(self._text(*a,**k)); return self
    def tlines(self,x,y,lines,size,weight=500,fill="#1B2430",anchor="start",lh=None,ls="0"):
        lh=lh or size*1.32
        for i,ln in enumerate(lines):
            self.b.append(self._text(x,y+i*lh,ln,size,weight,fill,anchor,ls))
        return self
    def line(self,x1,y1,x2,y2,stroke,w=1.4,dash=None,cap="butt"):
        d=f' stroke-dasharray="{dash}"' if dash else ""
        self.b.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}" stroke-linecap="{cap}"{d}/>')
        return self
    def rrect(self,x,y,w,h,fill,rx=12,stroke=None,sw=1.5,dash=None,op=None):
        s=f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        d=f' stroke-dasharray="{dash}"' if dash else ""
        o=f' opacity="{op}"' if op is not None else ""
        self.b.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{s}{d}{o}/>')
        return self
    def circle(self,cx,cy,r,fill,stroke=None,sw=1.5):
        s=f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.b.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{s}/>')
        return self
    def path(self,d,stroke="none",fill="none",w=1.4,dash=None,cap="round",op=None):
        da=f' stroke-dasharray="{dash}"' if dash else ""
        o=f' opacity="{op}"' if op is not None else ""
        self.b.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" stroke-linecap="{cap}" stroke-linejoin="round"{da}{o}/>')
        return self
    def arrow(self,x1,y1,x2,y2,stroke=None,w=2.4,mid=None):
        stroke=stroke or C["grayD"]
        aid="a"+str(abs(hash((x1,y1,x2,y2,stroke)))%99999)
        self.b.append(f'<defs><marker id="{aid}" markerWidth="9" markerHeight="9" refX="6.5" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="{stroke}"/></marker></defs>')
        self.b.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}" marker-end="url(#{aid})"/>')
        return self

    # ---- composite: labeled box (multi-line centered) ----
    def box(self,x,y,w,h,fill,title=None,lines=None,rx=12,stroke=None,sw=1.5,
            tcolor=None,lcolor=None,tsize=22,lsize=17,dash=None,badge=None,badgecol=None):
        self.rrect(x,y,w,h,fill,rx,stroke,sw,dash)
        cx=x+w/2
        block=[]
        if title: block.append((title,tsize,700,tcolor or C["ink"]))
        for ln in (lines or []): block.append((ln,lsize,500,lcolor or C["sec"]))
        tot=sum(s*1.34 for _,s,_,_ in block)
        cy=y+h/2-tot/2
        for txt,s,wt,col in block:
            cy+=s
            self.b.append(self._text(cx,cy-2,txt,s,wt,col,"middle"))
            cy+=s*0.34
        if badge:
            self.circle(x+22,y+22,15,badgecol or C["tealT"])
            self.b.append(self._text(x+22,y+28,badge,17,700,C["teal"] if not badgecol else "#fff","middle"))
        return self

    def source_note(self):
        if self.source:
            self.b.append(self._text(self.pad,self.h-self.pad+30,self.source,16,500,C["muted"]))

    def svg(self, font_css):
        self.source_note()
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">'
                f'<defs><style>{font_css}\ntext{{font-family:{self.FONT};}}</style></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="{self.bg}"/>'
                + "".join(self.b) + '</svg>')
