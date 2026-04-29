Proposal folder

Files added:
- proposal.tex : LaTeX source draft for the mini-project proposal.

How to compile to PDF on Windows (PowerShell examples)

Prerequisites:
- Install MikTeX (https://miktex.org) or TeX Live (https://tug.org/texlive/).
- Make sure `pdflatex` (or `latexmk`) is in your PATH.

Simple compile with pdflatex (run 2--3 times to resolve TOC and references):

```powershell
cd proposal
pdflatex proposal.tex
pdflatex proposal.tex
pdflatex proposal.tex
``` 

Using latexmk (recommended if installed):

```powershell
cd proposal
latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" proposal.tex
```

Notes and assumptions
- This `proposal.tex` is a draft using a basic article class. The department requires a specific LaTeX template; if you have that template file, replace the preamble and documentclass accordingly (or share it and I will adapt the source to it).
- Page length should be between 6 and 20 pages. After compiling, open the generated PDF and trim/expand content to meet the limit.

Next steps (recommended):
1. Replace the draft preamble with the official LaTeX template (if available).
2. Review and expand literature review paragraphs and figures; embed any figures in the `proposal/figs/` folder and reference them in the tex.
3. Run the compile commands above; if errors appear, share the .log and I can fix the LaTeX.

If you want, I can:
- Adapt the draft to your department template if you upload it.
- Generate a short two-page executive summary PDF for quick submission.
- Insert figures (flowcharts) generated from repository scripts.
