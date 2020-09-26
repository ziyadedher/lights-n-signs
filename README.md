# Lights and Signs Training
## Description 
This repository contains all the Lights and Signs sub-team training code and infrastructure. Most lights and signs development will happen in here and then get deployed to `zeus` once it is ready and tested.

## Structure
The repository is structured in a way that represents the modularization of our system. Our framework resides in `lns`.

### `lns`
This module encapsulates our framework package and shares the same name as it. We use this to create a high-level namespace for our work. Under `lns` we have our different systems that require training and well as a `common` folder.

All the code is well-documented, so this document just serves as a high-level view.

#### `lns.common`
This module contains all code shared between different training systems including dataset structures and other interfaces that should be implemented per training system to streamline the process. For more details, you can check out the code.


## Installation for Development
Here is a sequence of commands to quickly get started with developing the repository.

Download and navigate to the repository:
1. `git clone git@gitlab.com:aUToronto/autonomy/lights-n-signs-training.git`
2. `cd lights-n-signs-training`

Create a virtual environment for development:
3. `python3 -m venv .venv`
4. `source .venv/bin/activate`

Install the package in development mode with linter options:
5. `pip install -U pip`
6. `pip install -e '.[lint]'`

Run `tox` to verify installation, this should pass:
7. `tox`

You're ready to develop the repository.

### Usage
Since you installed the package in development mode using `-e`, any changes you make in your package will be reflected in your Python environment. So the easiest way to test out changes would be to run an interpreter and just run the code through it. For example, if I changed the `Preprocessor` and I want to verify the `preprocess` function still exists, I can do the following in an interactive Python shell in the sourced environment (there should be a `(.venv)` in your command line):
```python
from lns.common.preprocess import Preprocessor
Preprocessor.preprocess()
```


## Development
The repository is quite well-documented and laid out, this is due to a number of factors:
* We use a CI pipeline to enforce certain code quality and consistency regulations, as well as syntactic validity.
* We require all contributions to develop to be merged in using a merge request rather that directly pushed.

NOTE: I would recommend cloning this repository (or any repository in general) using the `git@` link rather than the `https://` one, it makes authentication more secure and quicker in the long run.


### Changes
After making any changes you should always verify that the code is still syntactically valid using by running `tox`. You can also see the Usage section above on how to manually run your code quickly.

Note that many editors have capabilities to do linting and syntax checking on-the-fly. All the linter configurations are in `setup.cfg`, so point your linter to that do load a consistent configuration. For Atom I use `linter-pylama` with an external instance pointed to the `pylama` under `.venv/bin/pylama` and setings under `setup.cfg`. Setting this up can be a little tricky, don't hesitate to ask @ziyadedher.
