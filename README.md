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

- Split one spritesheet into several spritesheets, one per animation (will create files `arrange_files/_separated/{cast,walk,thrust,slash,shoot,hurt}.png`):

	```bash
	lpctools arrange separate --input tests/arrange_files/male.png --layout universal --output-dir tests/arrange_files/_separated
	```

	- This can also be accomplished with `repack`; Split one spritesheet into several spritesheets, one per animation:

		```bash
		lpctools arrange repack --input tests/arrange_files/packed-evert.png --from evert --to cast thrust walk slash shoot hurt grab push --output-dir tests/arrange_files/repacked
		```

- Combine multiple spritesheets into one larger layout:
	
	```bash
	lpctools arrange combine --input tests/arrange_files/repacked --layout universal --output tests/arrange_files/_combined.png
	```

	- This can also be accomplished with `repack`; Split one spritesheet into several spritesheets, one per animation:

		```bash
		lpctools arrange repack --input tests/arrange_files/repacked/{cast,thrust,walk,slash,shoot,hurt}.png --from cast thrust walk slash shoot hurt --to universal --output tests/arrange_files/_combined.png
		```


- Recolor two hairstyles to two different palettes:

	```bash
	lpctools colors -v recolor --input tests/recolor_files/hair.png tests/recolor_files/hair_page2.png --mapping tests/recolor_files/palettes.json
	```

- Recolor a hairstyle to two different palettes, using a palette defined by an image:

	```bash
	lpctools colors -v recolor --input tests/recolor_files/hair.png --mapping tests/recolor_files/map.png --palette-names blonde blue
	```

- Recolor an image from one palette to another, with one palette defined by an image and another in a GIMP .gpl file:

	```bash
	lpctools colors -v recolor --input tests/recolor_files/human_head.png --from tests/recolor_files/ivory.png --to tests/recolor_files/ogre.gpl
	```

- Recolor all male "hair" images from the Universal LPC Spritesheet:

	```bash
	$ git clone --shallow https://github.com/sanderfrenken/Universal-LPC-Spritesheet-Character-Generator
	$ time (lpctools colors recolor \
		--input Universal-LPC-Spritesheet-Character-Generator/spritesheets/hair/male/*.png \
		--mapping tests/recolor_files/all-palettes.json)

	real	1m18.583s
	user	1m6.450s
	sys	0m3.001s
	```

## Acknowledgements

- joewhite's Universal Hair generator: https://github.com/joewhite/Universal-LPC-spritesheet/tree/universal-hair , GNU GPL 3.0 and CC-BY-SA 3.0	
	- Example MASK and OFFSET images borrowed and/or adapted from here; example palettes in `tests/recolor_files/all-palettes.json`
	- `distribute` tool heavily inspired by `rake`-based hair build system
- "LPC modular characters" by basxto https://github.com/basxto/lpc-modular-characters
	+ Sample palettes (ivory.gpl, ogre.gpl)
	+ Inspiration for `recolor` and `distribute` tool
- Art included for demonstration purposes only, find originals at <https://opengameart.org/content/lpc-collection>. Please do not distribute the included art without credit to the original artists.

|         File         |                    Authors                     |        License(s)       |                                                                                                                           URL(s)                                                                                                                           |
|----------------------|------------------------------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| hair_page2.png       | Johannes Sjölund (wulax)                       | CC-BY-SA 3.0, GPL 3.0   | https://opengameart.org/content/lpc-medieval-fantasy-character-sprites                                                                                                                                                                                     |
| hair_plain.png       | Manuel Riecke, Joe White                       | CC-BY-SA 3.0, GPL 3.0   | https://opengameart.org/content/liberated-pixel-cup-lpc-base-assets-sprites-map-tiles		https://opengameart.org/content/ponytail-and-plain-hairstyles                                                                                                       |
| human_head.png       | basxto                                         | CC-BY-SA 3.0, GPL 3.0   | https://github.com/basxto/lpc-modular-characters                                                                                                                                                                                                           |
| packed-universal.png | RedShrike, wulax                               | CC-BY-SA 3.0, GPL 3.0   | https://opengameart.org/content/liberated-pixel-cup-lpc-base-assets-sprites-map-tiles	https://opengameart.org/content/lpc-medieval-fantasy-character-sprites                                                                                               |
| packed-evert.png     | RedShrike, wulax, daneeklu, BenCreating, Evert | -                       | -                                                                                                                                                                                                                                                          |
| grab.png             | Redshrike, daneeklu, BenCreating               | CC-BY-SA 3.0, GPL 3.0   | https://opengameart.org/sites/default/files/forum-attachments/cleaned-grab.png https://opengameart.org/content/lpc-farming-tilesets-magic-animations-and-ui-elements https://opengameart.org/content/liberated-pixel-cup-lpc-base-assets-sprites-map-tiles |
| walk_push.png        | Redshrike, daneeklu, BenCreating, Evert        |                         | https://opengameart.org/sites/default/files/forum-attachments/walk_push.png                                                                                                                                                                                |
| crusader.png         | bluecarrot16                                   | OGA-BY 3.0+, CC-BY 3.0+ | https://opengameart.org/content/lpc-shields                                                                                                                                                                                                                |
| spartan.png          | bluecarrot16                                   | OGA-BY 3.0+, CC-BY 3.0+ | https://opengameart.org/content/lpc-shields                                                                                                                                                                                                                |
