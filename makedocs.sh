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