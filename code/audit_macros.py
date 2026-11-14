import re, glob, sys, os

MS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "manuscript")
defined = {}
for f in ("numbers_all.tex",):
    for l in open(os.path.join(MS, f)):
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}", l)
        if m:
            defined[m.group(1)] = f

std = set("""section label citep cite textbf textit emph paragraph multicolumn cmidrule toprule midrule bottomrule
begin end frac mathbb mathrm textsc xspace varepsilon le ge in subseteq to rightarrow times lambda rho pi tau beta
gamma sigma Delta mu Phi alpha theta sum prod int log exp max min arg leq geq neq approx sim pm mp cdot ldots dots
cdots quad qquad hline newtheorem usepackage documentclass input bibliography bibliographystyle maketitle keywords
ccsdesc copyrightyear acmYear setcopyright acmConference acmBooktitle acmDOI acmISBN title author affiliation
institution city country caption small label ref eqref left right langle rangle bar hat tilde infty partial nabla
forall exists emptyset setminus cup cap mid vert Vert lVert rVert Big Bigg textrm mathbf mathcal bm boxed widehat
mathbb textwidth linewidth textheight relax newcommand renewcommand def let makeatletter makeatother ifx else fi
hline textsc textnormal hspace vspace rule textbullet ast star circ prime dfrac tfrac sqrt left right
noindent indent par newline linebreak hfill vfill fontsize selectfont bfseries itshape scshape
subsection subsubsection appendix thanks footnote fbox mbox hbox vbox""".split())

used = {}
for f in [os.path.join(MS, "main.tex")] + sorted(glob.glob(os.path.join(MS, "sections", "*.tex"))):
    t = open(f).read()
    for m in re.finditer(r"\\([A-Za-z]+)", t):
        used.setdefault(m.group(1), set()).add(os.path.basename(f))

missing = sorted(n for n in used if n not in defined and n not in std and n[0].isupper())
print("Used but not defined by numbers_all.tex (uppercase heuristic):")
for n in missing:
    print("  ", n, sorted(used[n]))
print("\nDefined but never used:")
for n in sorted(defined):
    if n not in used:
        print("  ", n, "(", defined[n], ")")
