# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXMULTI   ?= sphinx-multiversion
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# .PHONY: sphinx-apidoc
# sphinx-apidoc:
#	sphinx-apidoc -Mf -o source/modules -M "../MapPy/"
#	rm source/modules/modules.rst

.PHONY: html
html: # sphinx-apidoc # ds_mode
	$(SPHINXMULTI) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

USER_GUIDE = "$(SOURCEDIR)"/contents/user_guide/

# .PHONY: ds_mode
# ds_mode:
# 	pandoc -i "$(USER_GUIDE)"/ds_mode.tex -o "$(USER_GUIDE)"/ds_mode.rst