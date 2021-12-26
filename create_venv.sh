#!/usr/bin/env bash
   
VENVNAME= FB_RANDOMFOREST

python3 -m venv $VENVNAME .
source $VENVNAME/bin/activate
pip install --upgrade pip

# problems when installing from requirements.txt
pip install ipython
pip install jupyter

jupyter nbextension install --py mayavi --user
jupyter nbextension enable --py mayavi --user

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"