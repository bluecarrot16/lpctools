# `lpctools`

Tools for manipulating pixel art sprites and tilesets. 

## Installation

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Clone and install repository

	```
	git clone https://github.com/bluecarrot16/lpctools
	cd lpctools
	conda env create -f environment.yml
	conda activate lpctools
	pip install -e .
	```


Easiest way to install dependencies is with `conda` as above. If you don't want to use conda, you can install the dependencies manually. 

Current dependencies:
- `pillow`
- `numpy`
- `pandas`
- `pytest` (for unit tests)

Future versions will also depend on:
- `pygame`
- `PyTMX`

## Usage

run `lpctools --help` for more detailed usage