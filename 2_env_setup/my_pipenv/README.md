```bash
# 1. Install pipenv globally
pip install pipenv

# 2. Navigate to your project directory
cd my_pipenv

# 3. Initialize pipenv (creates Pipfile)
pipenv --python 3.9

# 4. Install a package (creates virtual environment if it doesn't exist)
pipenv install requests

# 5. Run a script within the pipenv environment
pipenv run python my_script.py

# 6. Spawn a shell within the pipenv environment
pipenv shell

# 7. Exit the pipenv shell
exit

```