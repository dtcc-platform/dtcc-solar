# DTCC Solar

DTCC Solar is [FIXME].

This project is part the
[Digital Twin Platform (DTCC Platform)](https://gitlab.com/dtcc-platform)
developed at the
[Digital Twin Cities Centre](https://dtcc.chalmers.se/)
supported by Sweden’s Innovation Agency Vinnova under Grant No. 2019-421 00041.

## Documentation

* [Introduction](./doc/introduction.md)
* [Installation](./doc/installation.md)
* [Usage](./doc/usage.md)
* [Development](./doc/development.md)
* [Contributing](./doc/contributing.md)

## Authors (in order of appearance)

* FIXME

Part of this code is contributed by ReSpace AB under the MIT License.

## License

DTCC Solar is licensed under the
[MIT license](https://opensource.org/licenses/MIT).

Copyright is held by the individual authors as listed at the top of
each source file.

## Community guidelines

Comments, contributions, and questions are welcome. Please engage with
us through Issues, Pull Requests, and Discussions on our GitHub page.

## Notes

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
