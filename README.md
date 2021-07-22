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

run `lpctools --help`, `lpctools COMMAND --help`, `lpctools COMMAND SUBCOMMAND --help`, etc. for more detailed usage

- `lpctools arrange`: organizes animation frames into spritesheets
	+ `lpctools arrange repack`: re-arrange spritesheet
	+ `lpctools arrange distribute`: takes small number of images, arranges them into full set of animations
	+ `lpctools arrange unpack`: takes a spritesheet and slices it up into many individual images
	+ `lpctools arrange pack`: takes many individual images and arranges into a spritesheet
- `lpctools colors`: manipulates palettes, recolors images
	+ `lpctools colors recolor`: re-color image(s) with several palette(s)
	+ `lpctools colors convert-palette`: convert color palettes between different formats
	+ `lpctools colors create-mapping`: create a mapping between several color palettes
	+ `lpctools colors convert-mapping`: convert a mapping between different formats


## Examples

- Construct a full spritesheet of animations for several hairstyles:

	```bash
	lpctools arrange distribute \
	--input tests/arrange_files/hair/hair_plain/ \
	--output tests/arrange_files/shield/hair_plain.png \
	--input tests/arrange_files/hair/hair_page2/ \
	--output tests/arrange_files/shield/hair_page2.png \
	--input tests/arrange_files/hair/hair_shoulderr/ \
	--output tests/arrange_files/shield/hair_shoulderr.png \
	--offsets tests/arrange_files/hair/reference_points_male.png \
	--mask tests/arrange_files/hair/masks_male.png
	```

- Construct a full spritesheet of animations for the crusader shield:

	```bash
	lpctools arrange distribute --input tests/arrange_files/shield/crusader/ --output tests/arrange_files/shield/crusader.png --offsets tests/arrange_files/shield/reference_points_male.png --mask tests/arrange_files/shield/masks_male.png
	```

- Re-arrange images from one spritesheet layout to another:

	```bash
	lpctools arrange repack --input tests/arrange_files/packed-evert.png --from evert --to universal
	```

- Split one spritesheet into several spritesheets, one per animation:

	```bash
	lpctools arrange repack --input tests/arrange_files/packed-evert.png --from evert --to cast thrust walk slash shoot hurt grab push --output-dir tests/arrange_files/repacked
	```

- Recolor two hairstyles to two different palettes:

	```bash
	lpctools colors -v recolor --input tests/recolor_files/hair.png tests/recolor_files/hair2.png --mapping tests/recolor_files/palettes.json
	```

- Recolor a hairstyle to two different palettes, using a palette defined by an image:

	```bash
	lpctools colors -v recolor --input tests/recolor_files/hair.png --mapping tests/recolor_files/map.png
	```

- Recolor an image from one palette to another, with one palette defined by an image and another in a GIMP .gpl file:

	```bash
	lpctools colors -v recolor --input tests/recolor_files/human_head.png --from tests/recolor_files/ivory.png --to tests/recolor_files/ogre.gpl
	```

## Acknowledgements

- Art included for demonstration purposes only, find originals at <https://opengameart.org/content/lpc-collection>. Please do not distribute the included art without credit to the original artists
- Inspired by:
	- joewhite's Universal Hair generator: https://github.com/joewhite/Universal-LPC-spritesheet/tree/universal-hair , GNU GPL 3.0 and CC-BY-SA 3.0	
	- basxto's modular characters https://github.com/basxto/lpc-modular-characters