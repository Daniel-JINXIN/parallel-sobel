all:
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -f *.log
	rm -f *.aux
	rm -f *.bbl
	rm -f *.blg
	rm -f report.pdf
