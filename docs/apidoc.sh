# bash

if ! python -c 'import numpydoc'; then easy_install --user numpydoc; fi
if ! python -c 'import sphinx'; then easy_install --user sphinx; fi

#sphinx-apidoc -H "API Reference" -M -e -f -o . ../sharedmem
# do not use this because we manually use the doc string of sharedmem.sharedmem
# the from * line in __init__.py
