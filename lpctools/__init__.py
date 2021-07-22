import sys
from .utils import *
import textwrap

def wrap_fill(text, width=79, **kwargs):
	return textwrap.fill(text, width=width, **kwargs)

_leading_whitespace_re = re.compile('(^[ \t]*)(?:[^ \t\n])')
def dedent(s):
	lines = s.splitlines()
	indents = _leading_whitespace_re.findall(lines[0])
	indent = indents[0]
	return "\n".join(line.removeprefix(indent) for line in lines)

# # https://mike.depalatis.net/blog/simplifying-argparse.html
# def subcommand(args=[], parent=subparsers):
#     def decorator(func):
#         parser = parent.add_parser(func.__name__, description=func.__doc__)
#         for arg in args:
#             parser.add_argument(*arg[0], **arg[1])
#         parser.set_defaults(func=func)
#     return decorator

# def argumenwt(*name_or_flags, **kwargs):
#     return ([*name_or_flags], kwargs)


# if __name__ == "__main__":
#     args = cli.parse_args()
#     if args.subcommand is None:
#         cli.print_help()
#     else:
#         args.func(args)


# @subcommand([argument("-d", help="Debug mode", action="store_true")])
# def test(args):
#     print(args)


# @subcommand([argument("-f", "--filename", help="A thing with a filename")])
# def filename(args):
#     print(args.filename)


# @subcommand([argument("name", help="Name")])
# def name(args):
#     print(args.name)

def main(argv=None):

	def main_colors(argv, ns=None):

		parser = argparse.ArgumentParser(description='Utilities for recoloring images', prog='lpctools colors')
		subparsers = parser.add_subparsers(dest='command', title='subcommands', required=True, 
			description='Use %(prog)s SUBCOMMAND --help for more detailed help.')

		parser.add_argument('--verbose', '-v', action='count', dest='verbose', default=0)

		# RECOLORING SUBCOMMANDS
		# ----------------------
		# RECOLOR subcommand
		parser_recolor = subparsers.add_parser('recolor', help='Recolor an image')	
		parser_recolor.add_argument('--input', dest='input', action='extend', nargs='+',
							help='input filename(s)')

		parser_recolor.add_argument('--output', dest='output', action='store', nargs='+', #action=ExtendActionOverwriteDefault, nargs='+',
							default=['%i/%p.%e'],
							help='output filename pattern')
		# parser_recolor.add_argument('--output-dir', dest='output_dir')
		parser_recolor.add_argument('--from', dest='source')
		parser_recolor.add_argument('--to',  dest='target', action='extend', nargs='+')

		parser_recolor.add_argument('--mapping', dest='mapping')
		parser_recolor.add_argument('--palettes', dest='palettes', action='extend', nargs='+')
		parser_recolor.add_argument('--output-mapping-image', dest='mapping_output')

		# convertpalette subcommand
		parser_palette = subparsers.add_parser('convert-palette', help='Convert a color palette between formats')	
		parser_palette.add_argument('--input')
		parser_palette.add_argument('--output')

		# convertmapping subcommand
		parser_convertpalette = subparsers.add_parser('convert-mapping', help='Convert a color mapping between formats')	
		parser_convertpalette.add_argument('--input')
		parser_convertpalette.add_argument('--output')

		# createmapping subcommand
		parser_mapping = subparsers.add_parser('create-mapping', help='Construct a color mapping from palette(s)')	
		# parser_mapping.add_argument('--source')
		# parser_mapping.add_argument('--target', action='extend', nargs='+')
		parser_mapping.add_argument('--from', dest='source')
		parser_mapping.add_argument('--to',  dest='target',  action='extend', nargs='+')
		parser_mapping.add_argument('--output')

		args = parser.parse_args(argv, ns)

		from .recolor import main_recolor, main_convertpalette, main_convertmapping, main_colormap
		sub_commands = {
			'recolor': main_recolor,
			'convert-palette': main_convertpalette,
			'convert-mapping': main_convertmapping,
			'create-mapping': main_colormap
		}

		sub_commands[args.command](args)


	def main_arrange(argv, ns=None):
		# ARRANGE SUBCOMMANDS
		# -------------------
		
		from .arrange import layouts, distribute_layers, IMAGE_FRAME_PATTERN, main_pack, main_unpack, main_repack, main_distribute

		parser = argparse.ArgumentParser(description='Utilities for arranging and combining images', prog='lpctools arrange')
		subparsers = parser.add_subparsers(dest='command', title='subcommands', required=True, 
			description='Use %(prog)s SUBCOMMAND --help for more detailed help.')

		parser.add_argument('--verbose', '-v', action='count', dest='verbose', default=0, help='Print diagnostic messages. Repeat for higher verbosity.')


		pattern_help = dedent("""\
			Patterns are specified using these characters: 
			- %n = name of the animation (e.g. cast, thrust, shoot)'
			- %d = direction of the animation (e.g. n = north, s = south, etc.)
			- %f = the frame number
			""")

		layouts_help = ("Available layouts:\n" +
			"\n".join(f"- {layout_name}" for layout_name in layouts))


		parser_pack = subparsers.add_parser('pack', 
			help='Packs images, one per animation frame, into a spritesheet',
			formatter_class=argparse.RawTextHelpFormatter,
			epilog=dedent(f"""\
			{pattern_help}

			{layouts_help}
			""")
			)
		parser_pack.add_argument('--input', action='extend',nargs='+', required=True, help='Images to pack, one per frame')
		parser_pack.add_argument('--output', required=True, help='Packed image')
		parser_pack.add_argument('--layout', default='universal', help='Name of the layout')
		parser_pack.add_argument('--pattern',default=IMAGE_FRAME_PATTERN, 
				help="How are images named? See details below (default: '%(default)s)'")


		parser_unpack = subparsers.add_parser('unpack', help='Unpacks a spritesheet into images, one for each animation frame',
			formatter_class=argparse.RawTextHelpFormatter,
			epilog=dedent(f"""\
			{pattern_help}

			Patterns can contain subdirectories, which will be created, e.g. %n/%d-%f.png
			will create `thrust/n-1.png`, `thrust/n-2.png`, etc.

			If --output-dir is given as well, images and subdirectories will be created within.
			OUTPUT_DIR will be created


			{layouts_help}
			""")
			)
		parser_unpack.add_argument('--input',required=True, help='Packed image')
		parser_unpack.add_argument('--pattern',default=IMAGE_FRAME_PATTERN, 
			help="How should frame images be named? See details below (default: '%(default)s)'")
		parser_unpack.add_argument('--output-dir',dest='output_dir', default='.', 
			help='Directory where the frame images should be placed')
		parser_unpack.add_argument('--layout', default='universal')


		parser_repack = subparsers.add_parser('repack', help='Re-packs animation frames from one spritesheet layout to other(s)',
			formatter_class=argparse.RawTextHelpFormatter,
			epilog=dedent(f"""\
			If multiple input images are provided, then multiple FROM layouts must be provided, one per image. All frames will be merged 
			together and used to pack the resulting TO layout(s). If the same frame appears in multiple images, the last occurrance will
			be used. 

			Will create one image per TO layout.

			{layouts_help}
			""")
			)
		parser_repack.add_argument('--from', default=['universal'], help='Layout(s) of the original spritesheet images')
		parser_repack.add_argument('--to', default=['cast','thrust','walk','slash','shoot','hurt'], nargs='+', help='New layout(s) to create')


		layers_help = '\n'.join(wrap_fill(f"- {layer_name} : {layer['help']}", width=79) for layer_name, layer in distribute_layers.items() )

		mask_colors_help = '\n'.join(
			f"- {', '.join(c for c in layer['mask_colors'])}-colored pixels in the mask image will be removed from the {layer_name} layer" for layer_name, layer in distribute_layers.items())

		parser_distribute = subparsers.add_parser('distribute', 
			help='Distributes images across many animations',
			description=wrap_fill(
				'Takes a small set of objects and distributes them across all animation '
				'frames in a layout. Useful for objects that are similar for many '
				'animation frames (e.g. hair, hats, shields). Can optionally offset '
				'and/or create cutout masks for certain frames.'),
			formatter_class=argparse.RawTextHelpFormatter,
			epilog=dedent(f"""\
			ALGORITHM
			- INPUT images are collected as below
			- for each group of INPUT images
			  - INPUT image names are used to determine which animation frame(s) and layer
			    each image belongs to
			  - for each layer:
			    - INPUT images are arranged according to the OFFSET image
			    - the MASK image is used to cutout certain pixels from each frame
			    - animation frames are packed into a LAYOUT
			  - layer images are composited together
			  - the assembled image is written to the corresponding OUTPUT file

			INPUT IMAGES
			Each --input flag defines an INPUT image group. All images in the group
			will be used to make one OUTPUT image. 

			1) --input IMAGE1 IMAGE2 ... --output OUTPUT
				Will load all provided images and use to create one OUTPUT image
			2) --input DIRECTORY1 --output OUTPUT1 --input DIRECTORY2 --output OUTPUT2
				Will load all images in DIRECTORY1 and use to create OUTPUT1,
				then load all images in DIRECTORY2 and use to create OUTPUT2, etc.
			3) --input IMAGE_A1 IMAGE_A2 ... --output OUTPUT_A --input IMAGE_B1 IMAGE_B2 --output OUTPUT_B
				Will load IMAGE_A1, IMAGE_A2, etc. and use to create OUTPUT_A
				then load IMAGE_B1, IMAGE_B2, etc. and use to create OUTPUT_B.

			The number of --input flags should be the same as the number of --output flags. 
			INPUT image groups correspond to OUTPUT images by order.


			IMAGE NAMING
			The parser uses image names to determine for which frame(s) and in which 
			layer(s) the image should be used. Images should be named: 

			`[PREFIX-]D[-FRAMES].png`, where:

			* `D` is the direction the character is facing: `n`, `s`, `e`, or `w`.
			* `PREFIX-` is an optional layer name. If specified, this is either `bg-` or `behindbody-`.
			* `-FRAMES` is optional; if given, it is a dash-separated list specifying 
			  one or more animations or animation frames for which the image should be used. 

			Examples:
			- n.png will be used for all north-facing frames
			- s-shoot.png will be used for all south-facing frames in the `shoot` animation
			- s-shoot-hurt1.png will be used for all south-facing frames in the `shoot` 
			  animation, and for frame 1 of the south-facing `hurt` animation. 

			More specific images will take precedence over less specific ones (e.g. 
			s-shoot1.png > s-shoot.png > s.png)
			

			LAYERS
			The engine generates three layers:

			{layers_help}

			Each of these layers is masked (see below), then the three layers are 
			composited together in order. These cannot be modified from the command line 
			interface, but can be modified using the Python API.  


			OFFSETS
			An OFFSETS image can be provided, which should have the same LAYOUT as the 
			OUTPUT image(s). For each frame in the OFFSETS image, the input frame image 
			will be placed such that the middle pixel lines up with the first (top-most
			and left-most) black pixel in the OFFSETS frame. Only one OFFSETS image can 
			be given. 


			MASKS
			A MASK image can be provided, which should have the same LAYOUT as the OUTPUT 
			image(s). The MASK image is used to remove or cutout certain pixels from each 
			layer of input image(s). Only one MASK image can be given---the colors 
			determine which pixels are masked in each layer: 

			{mask_colors_help}

			LAYOUTS
			{layouts_help}
			""")
			)
		parser_distribute.add_argument('--input', required=True, action='append', nargs='+', 
			help='Input image(s) or directories')
		parser_distribute.add_argument('--output', required=True, nargs='+',
			help='Path where the complete spritesheet(s) should be placed')
		parser_distribute.add_argument('--layout', default='universal')
		parser_distribute.add_argument('--offsets', '--offset', required=True, 
			help='Path to image specifying the x/y coordinate for each frame')
		parser_distribute.add_argument('--masks', '--mask', required=True, 
			help='Path to image specifying the cutouts/masks for each layer')


		args = parser.parse_args(argv, ns)		
		sub_commands = {
			'unpack':main_unpack,
			'repack':main_repack,
			'pack':main_pack,
			'distribute':main_distribute
		}

		sub_commands[args.command](args)


	import argparse

	commands = {
		'colors': main_colors,
		'arrange': main_arrange
	}

	parser = argparse.ArgumentParser(description='Utilities for manipulating pixel art', 
		add_help=False, 
		epilog='Use %(prog)s SUBCOMMAND -h for more detailed help.'
		# ,exit_on_error=False
		)
	parser.add_argument('-h','--help', action='store_const', const=True)
	parser.add_argument('command', choices = list(commands.keys()) + ['help'], nargs='?', default='help', help='Subcommand to execute')

	# try:
		# ns, argv_rest = parser.parse_known_args(argv)
	# except argparse.ArgumentError as err:
		# parser.print_help()

	ns, argv_rest = parser.parse_known_args(argv)

	if ns.command == 'help':
		parser.print_help()
		sys.exit(1)
	if ns.command != 'help' and ns.help:
		argv_rest.append('--help')

	commands[ns.command](argv_rest, ns)
