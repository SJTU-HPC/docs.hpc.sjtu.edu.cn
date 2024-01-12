THESIS = main
TEMPLATE = elegantbook.cls
# TEX, BIB, TEST dir
TEX_DIR = ./
BIB_DIR = ./

# Option for latexmk
LATEXMK_OPT_BASE = -xelatex -gg -silent
LATEXMK_OPT = $(LATEXMK_OPT_BASE) -f
LATEXMK_OPT_PVC = $(LATEXMK_OPT_BASE) -pvc

all: $(THESIS).pdf cover.pdf

.PHONY : all cleanall pvc view wordcount git zip

$(THESIS).pdf : $(THESIS).tex *.tex $(TEMPLATE) Makefile
	-latexmk $(LATEXMK_OPT) $(THESIS)

cover.pdf : cover.tex $(TEMPLATE) Makefile
	-latexmk $(LATEXMK_OPT) cover

pvc :
	latexmk $(LATEXMK_OPT_PVC) $(THESIS)

validate :
	xelatex -no-pdf -halt-on-error $(THESIS)

view : $(THESIS).pdf
	open $<

wordcount:
	@texcount $(THESIS).tex -inc          | awk '/total/ {getline; print "词数　　:",$$4}'
	@texcount $(THESIS).tex -inc -char    | awk '/total/ {getline; print "字符数　:",$$4}'
	@texcount $(THESIS).tex -inc -ch-only | awk '/total/ {getline; print "中文字数:",$$4}'

clean :
	-@latexmk -c -silent 2> /dev/null
	-@rm -f $(TEX_DIR)/*.aux 2> /dev/null || true

cleanall :
	-@latexmk -C -silent 2> /dev/null
	-@rm -f $(TEX_DIR)/*.aux 2> /dev/null || true

s3 : $(THESIS).pdf
	s3cmd put $< s3://sjtuthesis/README.pdf

git :
	git push --tags github; git push github;
	git push --tags gitlab; git push gitlab;

zip :
	git archive --format zip --output thesis.zip master

small : $(THESIS).pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH -sOutputFile=small.pdf $<
