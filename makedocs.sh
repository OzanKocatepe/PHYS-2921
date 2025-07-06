# Moves important files out of sphinx.
mv sphinx/conf.py ..
mv sphinx/index.rst ..
mv sphinx/make.bat ..
mv sphinx/Makefile ..

# Delete all previous sphinx files.
rm -r sphinx/*

# Moves important files back.
mv conf.py sphinx/
mv index.rst sphinx/
mv make.bat sphinx/
mv Makefile sphinx/

# Make documentation.
sphinx-apidoc -o sphinx src/
cd sphinx
make clean
make html

# Move back to root directory.
cd ..

# Remove pre-existing docs files.
rm -r docs/*

# Set up new docs files.
cp -r sphinx/_build/html/* docs/
touch docs/.nojekyll