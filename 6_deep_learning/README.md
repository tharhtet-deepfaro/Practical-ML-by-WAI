
## Create anaconda for python
```bash
# create conda environment
conda create --name <environment_name> python=3.11.13

# remove
conda remove -n ENV_NAME --all

where python


```





## Create pipenv
```bash
pipenv install --python <your-python-path or version>

# remove pipenv
pipenv --rm

# check python version
pipenv run python --version

# check pipenv location
pipenv --py

# sync pipenv with conda
pipenv sync

pipenv install opencv-python
pipenv install python-dotenv
pipenv install tensorflow==2.17.0


# dev only for mac os
pipenv install --dev tensorflow-macos tensorflow-metal
pipenv install --dev ipython

# for tessting
pipenv install --dev matplotlib


# for Production run
# Installs only what’s in Pipfile.lock, ignoring dev packages
pipenv --rm # to test remove first
pipenv install --deploy --ignore-pipfile

# for dev
# Everything in Pipfile.lock
# Including [dev-packages]
pipenv install


# test code
pipenv run python 
import matplotlib.pyplot as plt


```