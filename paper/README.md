# VCSR Paper Draft

This directory contains the first manuscript draft for:

**Verifier-Calibrated Search and Repair for Faithful Text-to-PDDL Generation**

Primary source files:

- `main.tex`: first full LaTeX draft.
- `../references.bib`: shared bibliography.
- `../PAPER_PLAN.md`: paper plan and frozen claim hierarchy.
- `../results/paper/final_vcsr/`: exported paper tables and figure specs.

To compile locally from this directory:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft intentionally treats the final seed `51-55` result as the primary
paper evidence and the Claude-family model benchmark as appendix robustness
evidence.
