from .utils import *


# # https://mike.depalatis.net/blog/simplifying-argparse.html
# def subcommand(args=[], parent=subparsers):
#     def decorator(func):
#         parser = parent.add_parser(func.__name__, description=func.__doc__)
#         for arg in args:
#             parser.add_argument(*arg[0], **arg[1])
#         parser.set_defaults(func=func)
#     return decorator

# def argument(*name_or_flags, **kwargs):
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

def main(arguments=None):
	
	import argparse


	parser = argparse.ArgumentParser(description='Utilities for recoloring images')
	subparsers = parser.add_subparsers(dest='command', title='subcommands', description='Use lpctools SUBCOMMAND --help for more detailed help.')
	parser.add_argument('--verbose', '-v', action='count', dest='verbose', default=0)



	# RECOLORING SUBCOMMANDS
	# ----------------------
	# RECOLOR subcommand
	parser_recolor = subparsers.add_parser('recolor', description='Recolor an image')	
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
	parser_palette = subparsers.add_parser('convertpalette', description='Convert a color palette between formats')	
	parser_palette.add_argument('--input')
	parser_palette.add_argument('--output')

	# convertmapping subcommand
	parser_convertpalette = subparsers.add_parser('convertmapping', description='Convert a color mapping between formats')	
	parser_convertpalette.add_argument('--input')
	parser_convertpalette.add_argument('--output')


	# createmapping subcommand
	parser_mapping = subparsers.add_parser('createmapping', description='Construct a color mapping from palette(s)')	
	# parser_mapping.add_argument('--source')
	# parser_mapping.add_argument('--target', action='extend', nargs='+')
	parser_mapping.add_argument('--from', dest='source')
	parser_mapping.add_argument('--to',  dest='target',  action='extend', nargs='+')
	parser_mapping.add_argument('--output')

	args = parser.parse_args(arguments)


	if args.command == 'recolor':
		from .recolor import main_recolor
		main_recolor(args)

	elif args.command == 'convertpalette':
		from .recolor import main_convertpalette
		main_convertpalette(args)

	elif args.command == 'convertmapping':
		from .recolor import main_convertmapping
		main_convertmapping(args)

	elif args.command == 'createmapping':
		from .recolor import main_colormap
		main_colormap(args)









