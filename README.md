install poetry: curl -sSL https://install.python-poetry.org | python3 -

create venv: poetry shell

activate venv: source $(poetry env info --path)/bin/activate

install libs: poetry install

Build and install as a library:

poetry build
pip install dist/{wheel_file}.whl



add poetry to path: export  PATH="~/.local/bin:$PATH"

Error "Failed to unlock the collection!":

run -> export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring