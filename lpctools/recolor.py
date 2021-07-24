import os
import collections.abc
import colorsys
import json
import numpy as np

from PIL import Image
from PIL.ImageColor import getrgb
from PIL.ImageFilter import Color3DLUT

from .utils import *

def rgb2hex(r, g, b):
	return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def rgba2hex(r, g, b, a):
	return '#{:02x}{:02x}{:02x}{:02x}'.format(r, g, b, a)

def getrgba(c):
	rgb = getrgb(c)
	if len(rgb) == 3:
		return Color(*(rgb + (255,)))
	else: return Color(*rgb)

RGBATuple = collections.namedtuple('RGBATuple', ['r','g','b','a'])
class Color(RGBATuple):
	def __new__(cls, *args, **kwargs):
		if len(args) == 1:
			arg = args[0]
			if isinstance(arg, collections.abc.Iterable) and not isinstance(arg, str): 
				if not isinstance(arg, tuple): 
					arg = tuple(arg)

				if len(arg) == 3:
					rgba = arg + (255,)
				elif len(arg) == 4:
					rgba = arg
				elif len(arg) != 4:
					raise Exception(f'Invalid Color: {args}, {kwargs}')

			else: rgba = getrgba(args[0])
			return super().__new__(cls, *rgba, **kwargs)
		elif len(args) == 4:
			return super().__new__(cls, *args)
		elif len(args) == 3:
			return super().__new__(cls, *args, 255)
		elif len(args) == 0 and len(kwargs) > 0:
			return super().__new__(cls, {'r':0,'g':0,'b':0, 'a': 255, **kwargs})
		else: 
			raise Exception(f'Invalid Color: {args}, {kwargs}')

	def __repr__(self):
		return f"Color('{self.to_hex()}')"

	def to_hex(self):
		return rgba2hex(*self)

	def to_hsv(self):
		return colorsys.rgb_to_hsv(self.r, self.g, self.b) + (self.a,)

	def to_array(self):
		return np.asarray(self, dtype='uint8')

	def to_gpl(self):
		return f'{self.r: >3} {self.g: >3} {self.b: >3} Untitled'

	@staticmethod
	def squish_if_transparent(t):
		if len(t) == 4 and t[3] == 0:
			return (255,255,255,0)
		return tuple(t)

class ImagePalette():
	def __init__(self, colors=[], name=''):
		# self._colors = [getrgba(c) for c in colors]
		self._colors = [Color(c) for c in colors]
		self._colorset = set(self._colors)
		if not name and hasattr(colors, 'name'):
			self.name = colors.name
		else:
			self.name = name

	def __iter__(self):
		yield from self._colors

	def __len__(self):
		return len(self._colors)

	def __contains__(self, rgb):
		return rgb in self._colorset

	def __getitem__(self, i):
		return self._colors[i]

	def __repr__(self):
		r = 'ImagePalette([' + ",".join( f"'{c.to_hex()}'" for c in self._colors ) + ']'
		if self.name != '':
			r += f",name={self.name}"
		r += ")"
		return r



	def to_hex(self):
		return [rgb2hex(*x) for x in self._colors]

	def to_hsv(self):
		return [x.to_hsv() for x in self._colors]


	def reorder(self, ordering):
		return ImagePalette(np.array(self._colors)[ordering], name=self.name)

	def argsort(self, param='auto'):
		"""
		param: channel or list of channels to use for sorting; if list is given, 
		channels will be sorted lexicographically. 'auto' = ['alpha','value','saturation','hue']
		"""


		if param == 'auto':
			channel_order = ['alpha','value','saturation','hue']
		else: channel_order = listify(param)

		if not all(x in ['hue','saturation','value','alpha','auto'] for x in channel_order):
			raise Exception(f"Unknown sort key(s) {channel_order}. ")

		# k channels x N colors
		hsvs = np.array(self.to_hsv(), dtype=int).T
		channels = ['hue','saturation','value','alpha']

		# re-arrange channels in desired order
		# confusingly, np.lexsort sorts by the last row, then second-to-last, 
		# and so on, so we need to provide the channels in reverse order
		data = hsvs[ [channels.index(channel) for channel in reversed(channel_order)], : ]

		sort_order = np.lexsort(data)
		return sort_order

		# hsvs = self.to_hsv()
		# key = {
		# 	'hue':lambda h,s,v,a: h,
		# 	'saturation':lambda h,s,v,a: s,
		# 	'value':lambda h,s,v,a: v
		# }[param]

		# keyed = np.fromiter((key(*c) for c in self), count=len(self), dtype=int)
		# sort_order = np.argsort(keyed)
		# return sort_order
		# else:
			# raise Exception(f"Unknown sort key {param}")
			# return ImagePalette([colorsys.hsv_to_rgb(*x) for x in hsvs_sorted])	

	def sort(self, param='value'):
		return self.reorder(self.argsort(param))

	def sort_hue(self):
		# hsvs = self.to_hsv()
		# hsvs_sorted = sorted(hsvs, key=lambda h,s,v: h)
		# return ImagePalette([colorsys.hsv_to_rgb(*x) for x in hsvs_sorted])
		return self.sort(param='hue')

	def to_png(self, path=None):
		# +1 for the source palette
		img = Image.new('RGBA', size=(len(self), 1))
		pixels = img.load()

		for i, color in enumerate(self):
			pixels[i, 0] = tuple(color)

		if path is not None:
			img.save(path)

		return img

	def to_gpl(self, path=None):
		out = "\n".join([
			"GIMP Palette",
			f"Name: {self.name}",
			f"Columns: {len(self)}",
			"#"] + 
			[c.to_gpl() for c in self]
		)

		if path is not None:
			with open(path,'w') as f:
				f.write(out)
		return out

	def to_json(self, path=None):
		d = [c.to_hex() for c in self]
		if path is not None:
			with open(path,'w') as f:
				json.dump(d, f)


def load_palette_json(path, name=''):
	with open(path, 'r') as f:
		data = json.load(f)
	return ImagePalette(data, name=name)

def load_palette_png(path, name='', squish_transparent=True):
	img = Image.open(path)
	bn, _ = os.path.splitext(os.path.basename(path))

	# for indexed image, get palette directly in order
	if img.mode == 'P':
		colors = img.getpalette()
	else:
		# does not preserve order of colors in image
		# colors = [color for count, color in img.convert('RGBA').getcolors()]
		
		import pandas as pd

		# n_pixels x 4
		img_pixels = np.array(img).reshape((-1, 4))
		# colors = unique_rows(img_pixels)
		
		# treat all transparent pixels as equivalent by mapping all to rgba(255,255,255,0)
		if squish_transparent:
			img_pixels[img_pixels[:,3] == 0] = [255,255,255,0]

		# probably slow but seems like the only sane way to preserve ordering
		all_colors = [tuple(a) for a in img_pixels]
		colors = pd.unique(all_colors)

	return ImagePalette(colors, name=(name or bn))

def load_palette_gpl(path, name=''):
	color_regex = re.compile(r'^\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3}).*$')
	name_regex = re.compile(r'Name:\s+(.+)')
	colors = []
	_name = ''
	with open(path) as f:
		for line in f:
			match = color_regex.match(line)
			if match is not None:
				colors.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
				next

			match = name_regex.match(line)
			if match is not None:
				_name = match.group(1)

	print(colors)

	return ImagePalette(colors, name = name if name else _name)

	# from PIL.GimpPaletteFile import GimpPaletteFile

def load_palette(path, **kwargs):
	basename, ext = os.path.splitext(path)

	palette_loaders = {
		'.gpl': load_palette_gpl,
		'.json': load_palette_json,
		'.png': load_palette_png
	}

	if ext in palette_loaders:
		return palette_loaders[ext](path, **kwargs)
	else:
		raise Exception(f'Do not know how to load a palette from a {ext} file. Possible extensions: {palette_loaders.keys()}')

def save_palette(pal, path, **kwargs):
	basename, ext = os.path.splitext(path)
	palette_savers = {
		'.gpl': lambda path, **kwargs: pal.to_gpl(path, **kwargs),
		'.png': lambda path, **kwargs: pal.to_png(path, **kwargs),
		'.json': lambda path, **kwargs: pal.to_json(path, **kwargs),
	}

	if ext in palette_savers:
		palette_savers[ext](path, **kwargs)
	else:
		raise Exception(f'Do not know how to save a palette to a {ext} file. Possible extensions: {palette_savers.keys()}')


class ImagePaletteMapping(dict):
	def __init__(self, source_palette, dest_palettes):
		source_palette = ImagePalette(source_palette)
		self.source_palette = source_palette

		if isinstance(dest_palettes, dict):
			# self.names = dest_palettes.keys()
			dest_palettes = [ImagePalette(pal, name=name) for name, pal in dest_palettes.items()] #dest_palettes.values()
			# self.names = [pal.name for pal in dest_palettes]
		else: 
			# self.names = range(len(dest_palettes))
			dest_palettes = [ImagePalette(d) for d in dest_palettes]

		self.dest_palettes = dest_palettes

		self.names = [pal.name or i for i,pal in enumerate(dest_palettes)]
		self.n_palettes = len(dest_palettes)

		# s[0]: [ d1[0], d2[0], ... ],
		# s[1]: [ d1[1], d2[1], ... ]

		if not all(len(source_palette) == len(d) for d in dest_palettes):
			raise Exception("Source palette and all target palettes must all have the same number of colors: \n" +
				f"source palette {source_palette.name} = {len(source_palette)} colors: {source_palette} \n" +
				"target palettes: \n" + 
				"\n".join(f"- #{i}, {d.name} = {len(d)} colors: {d}" for i, d in enumerate(dest_palettes))
				)

		# super().__init__(zip(source_palette, dest_palettes))
		# 
		super().__init__( (s, [ d[i] for d in dest_palettes ]) for i,s in enumerate(source_palette) )

	def __repr__(self):
		return f"ImagePaletteMapping({repr(self.source_palette)}, {repr(self.dest_palettes)})"

	def __add__(self, other):
		if len(other) != len(self):
			raise NotImplementedError("Mappings must have the same number of colors to concatenate")
		if set(self.keys()) == set(other.keys()):
			self_reordered = self.reorder_like(other)
			return ImagePaletteMapping(self_reordered.source_palette, 
				self_reordered.dest_palettes + other.dest_palettes)
		raise Exception("Cannot add mappings with different source palettes")

	def reorder_like(self, other):
		# self = ['a','b','c','d']
		# other = ['d','a','b','c']
		# self[[3, 0, 1, 2]] == other

		ordering = [self.index(c) for c in other.source_palette]
		return self.reorder(ordering)

	def reorder(self, ordering):
		source_palette = self.source_palette.reorder(ordering)
		dest_palettes = [d.reorder(ordering) for d in self.dest_palettes]
		return ImagePaletteMapping(source_palette, dest_palettes)

	def sort_colors(self, param='value', verbose=False):
		ordering = self.source_palette.argsort(param)
		return self.reorder(ordering)

	@property
	def palettes(self):
		return [self.source_palette] + self.dest_palettes

	def to_image(self, path=None):
		# +1 for the source palette
		img = Image.new('RGBA', size=(len(self), self.n_palettes+1))
		pixels = img.load()

		for i, (sp, dps) in enumerate(self.items()):
			pixels[i, 0] = sp
			for j, dp in enumerate(dps):
				pixels[i, j+1] = tuple(dp)

		if path is not None:
			img.save(path)

		return img

	def to_json(self, path=None):

		# d = { 'source': [c.to_hex() for c in self.keys()], n: [None]*len(self.keys()) for in in self.names}

		d = { name: [c.to_hex() for c in colors] for name, colors in zip(self.names, zip(*self.values())) }
		d['source'] = [c.to_hex() for c in self.keys()]

		if path is not None:
			with open(path, 'w') as f:
				json.dump(d, f)

		return d

	def to_ndarray(self):
		"""express the mapping as a (len(self) x self.n_palettes+1 x 4) numpy.ndarray
		"""
		# return np.asarray([list(self.keys()), list(self.values())])

		arr = np.empty(shape=(len(self), self.n_palettes+1, 4))

		arr[:,0,:] = np.asarray(list(self.keys()), dtype=float)

		for j, dpcs in enumerate(self.values()):

			# print(np.asarray(list(dpcs)))
			arr[j, 1:, :] = np.asarray(list(dpcs))

		return arr

	def recolor_image(self, img):

		img = img.convert('RGBA')

		# "data" is a numpy array with shape = (height, width, 4) 
		data = np.array(img)   

		# datas is a list of copies of `data`, one per destination palette
		# (*datas) = np.repeat(data[..., np.newaxis], axis=-1)
		datas = [data.copy() for i in range(self.n_palettes)]

		# clone data so when checking for matching colors, we are always referencing the original; 
		# this is in case one color appears in both the source and the destination palette; we 
		# don't want to re-map it twice
		orig = data


		# len(self) x n_palettes x 4
		arr = self.to_ndarray()

		# c1 and c2 represent the source color and the destination color, respectively
		# shape(c1) == shape(c2) == (4,)
		for (c1, *dps) in arr:

			# get the pixels within `orig` where all 4 channels match the values in c1
			# c1 is broadcasted and compared to the last axis of orig, then that 
			# axis is "all"'d
			# shape(targets) = (height, width)
			targets = (orig == c1).all(axis=-1) #(red == c1[0]) & (green == c1[1]) & (blue == c1[2]) & (alpha == c1[3])


			for j, c2 in enumerate(dps):

				# for all matching pixels, re-assign the last axis of `data` to the channel
				# values specified by `c2`. 
				datas[j][targets,:] = c2 

		return [Image.fromarray(data) for data in datas]


def save_palette_mapping(mapping, path, **kwargs):
	basename, ext = os.path.splitext(path)
	mapping_savers = {
		'.png': lambda path, **kwargs: mapping.to_image(path, **kwargs),
		'.json': lambda path, **kwargs: mapping.to_json(path, **kwargs),
	}

	if ext in mapping_savers:
		mapping_savers[ext](path, **kwargs)
	else:
		raise Exception(f'Do not know how to save a mapping to a {ext} file. Possible extensions: {mapping_savers.keys()}')


def load_palette_mapping_json(data, names=None):
	if isinstance(data, str):
		with open(data) as f:
			data = json.load(f)

	source = data['source']
	data.pop('source')
	dests = data

	return ImagePaletteMapping(source, dests)

def load_palette_mapping_png(img, names=None, **kwargs):
	if isinstance(img, str):
		img_path = img
		img = Image.open(img_path).convert('RGBA')

	# (height = n_palettes+1, width = n_colors, 4) 
	data = np.array(img)

	source = data[0,:,:]
	dests = data[1:, :, :]

	if names is not None:
		if len(names) != dests.shape[0]:
			raise Exception(f'Error: {len(names)} names provided for {dests.shape[0]} palettes')
		dests = { name:pal for name, pal in zip(names, dests) }

	return ImagePaletteMapping(source, dests)	


def load_palette_mapping(path, **kwargs):
	basename, ext = os.path.splitext(path)
	mapping_loaders = {
		'.png': load_palette_mapping_png,
		'.json': load_palette_mapping_json
	}

	if ext in mapping_loaders:
		return mapping_loaders[ext](path, **kwargs)
	else:
		raise Exception(f'Do not know how to load a mapping from a {ext} file. Possible extensions: {mapping_loaders.keys()}')



# class ImagePaletteMapping(dict):
# 	def __init__(self, source_palette, dest_palette):
# 		# self.source = source_palette
# 		# self.destination = dest_palette
# 		assert len(source_palette) == len(dest_palette)

# 		# self._mapping = dict(zip(source_palette, dest_palette))
# 		super().__init__(zip(source_palette, dest_palette))

# 	def map(self, rgb):

# 		if rgb in self: 
# 			return self[rgb]
# 		else: return rgb

# 	def to_lut(self):
# 		def generate(r,g,b):
# 			return self.map((round(255*r), round(255*g), round(255*b)))
# 		return Color3DLUT.generate(size=65, callback=generate)

# 	def to_json(self):
# 		pass

# 	def to_image(self):
# 		img = Image.new('RGBA', size=(len(self), 2))
# 		pixels = img.load()

# 		for i, (p1, p2) in enumerate(self.items()):
# 			p1 = p1[0:3]
# 			p2 = p2[0:3]

# 			pixels[i,0] = p1
# 			pixels[i,1] = p2
# 		return img

# 	def to_ndarray(self):
# 		"""express the mapping as a (len(self) x 2 x 4) numpy.ndarray
# 		"""
# 		# return np.asarray([list(self.keys()), list(self.values())])
# 		return np.asarray(list(
# 			zip(self.keys(), self.values())
# 			))

# 	def recolor_image(self, img):

# 		img = img.convert('RGBA')

# 		# "data" is a numpy array with shape = (height, width, 4) 
# 		data = np.array(img)   

# 		# clone data so when checking for matching colors, we are always referencing the original; 
# 		# this is in case one color appears in both the source and the destination palette; we 
# 		# don't want to re-map it twice
# 		orig = data.copy() 


# 		# len(self) x 2 x 4
# 		arr = self.to_ndarray()

# 		# c1 and c2 represent the source color and the destination color, respectively
# 		# shape(c1) == shape(c2) == (4,)
# 		for (c1, c2) in arr:

			
# 			# get the pixels within `orig` where all 4 channels match the values in c1
# 			# c1 is broadcasted and compared to the last axis of orig, then that 
# 			# axis is "all"'d
# 			# shape(targets) = (height, width)
# 			targets = (orig == c1).all(axis=-1) #(red == c1[0]) & (green == c1[1]) & (blue == c1[2]) & (alpha == c1[3])

# 			# for all matching pixels, re-assign the last axis of `data` to the channel
# 			# values specified by `c2`. 
# 			data[targets,:] = c2 

# 		return Image.fromarray(data)


# 	@staticmethod
# 	def from_dict(d):
# 		pal1 = ImagePalette(data.keys())
# 		pal2 = ImagePalette(data.values())
# 		return ImagePaletteMapping(pal1, pal2)


# def load_palette_map_json(data):
# 	if isinstance(data, str):
# 		with open(data) as f:
# 			data = json.load(f)

# 	return ImagePaletteMapping.from_dict(data)


def recolor_map(img, mapping):
	lut = mapping.to_lut()
	print(lut)
	return img.filter(lut)

def recolor_index(img, colormap):
	pass

def coerce_palette(img, palette):
	pass

def audit_palette(img, palette):
	pass


def collapse_recolors(imgs):
	pass






def make_mapping(source_path, target_paths, names=None, verbose=False):
	if verbose: print(f"Source palette: loading from {source_path}")
	source_pal = load_palette(source_path)

	if verbose: print(f"Target palettes: {target_paths}")
	if names is None or len(names) == 0:
		names = [''] * len(target_paths)

	dest_pals = []
	for name, pal_str in zip(names, target_paths):
		if '=' in pal_str:
			name, path = pal_str.split('=')
		else: 
			path = pal_str
		
		if name != '':
			if verbose: print(f"- loading palette '{name}' from {path}")
			dest_pals.append(load_palette(path, name=name))
		else: 
			if verbose: print(f"- loading untitled palette from {path}")
			dest_pals.append(load_palette(path))

	colormap = ImagePaletteMapping(source_pal, dest_pals)
	return colormap


def convert_palette(input, output, sort=None, verbose=False):
	in_pal = load_palette(input)
	if sort is not None:
		if verbose: print(f"Sorting by channel {sort}")
		in_pal = in_pal.sort(sort)
	save_palette(in_pal, output)

def main_convertpalette(args):
	return convert_palette(args.input, args.output)


def convert_mapping(input, output, names=[], sort=None, verbose=True):
	in_map = load_palette_mapping(input, names=names)
	if sort is not None:
		if verbose: print(f"Sorting by channel {sort}")
		in_map = in_map.sort_colors(sort)
	save_palette_mapping(in_map, output)

def main_convertmapping(args):
	return convert_mapping(args.input, args.output, args.names, sort=args.sort, verbose=args.verbose)


def main_create_mapping(args):
	colormap = make_mapping(args.source, args.target, verbose=args.verbose)
	if args.verbose: 
		print(colormap)
		print(f"Saving mapping to {args.output}")

	save_palette_mapping(colormap, args.output)


def concat_mappings(mappings, source=None, targets=None, filter=False, drop=False, sort=None):
	pass

def main_concat_mappings(args):
	pass

# def main_recolor(args):
# 	# mapping = load_palette_map_json(args.mapping)

# 	if args.mapping is not None:
# 		if args.verbose: 
# 			if args.palettes is not None:
# 				print(f"Reading palette mapping: {args.mapping}")
# 			else:
# 				print(f"Reading palette mapping: {args.mapping}; renaming the palettes {args.palettes}")
# 		mapping = load_palette_mapping(args.mapping, names=args.palettes)
# 	elif args.source is not None and args.target is not None:
# 		mapping = make_mapping(args.source, args.target, verbose=args.verbose)
# 	else: 
# 		raise Exception('Must specify the color mapping, using either --mapping or --from and --to.')

# 	if args.mapping_output is not None:
# 		if args.verbose: print(f"Writing image representation of palette mapping to {args.mapping_output}")
# 		mapping_img = mapping.to_image()
# 		mapping_img.save(args.mapping_output)



# 	output_paths = args.output
# 	if len(output_paths) == 1:
# 		output_paths = output_paths * len(args.input)
# 	elif len(output_paths) != len(args.input):
# 		raise Exception("Must give either one --output argument, or the same number of --output as --input arguments (one per image) \n"
# 			f"- Inputs: {args.input} \n"
# 			f"- Outputs: {output_paths} \n")

# 	if args.verbose: print(output_paths)

# 	for input_path, output_path_fmt in zip(args.input, output_paths):
# 		if args.verbose: print(f"Reading input image {input_path}...")
# 		input_path_basename = os.path.basename(input_path)
# 		input_path_basename_sans_ext, _ = os.path.splitext(input_path_basename)
# 		input_path_sans_ext, input_path_ext = os.path.splitext(input_path)
# 		input_path_ext = input_path_ext.lstrip('.')

# 		img = Image.open(input_path)

# 		out_imgs = mapping.recolor_image(img) #recolor_map(img, mapping)

# 		for (out_img, palette_name) in zip(out_imgs, mapping.names):
# 			output_path = format_placeholders(output_path_fmt, {
# 				'%B': input_path_basename,
# 				'%b': input_path_basename_sans_ext,
# 				'%i': input_path_sans_ext, 
# 				'%e': input_path_ext,
# 				'%I': input_path,
# 				'%p': palette_name
# 			})

# 			if args.verbose: print(f"- writing output from palette '{palette_name}' to {output_path}")
# 			mkdirpf(output_path)
# 			out_img.save(output_path)

# 		# [out.save(args.output + name + ".png") for (out, name) in zip(outs, mapping.names)]


def main_recolor(args):
	# mapping = load_palette_map_json(args.mapping)

	mappings = []
	if len(args.mapping) > 0:
		if len(args.palettes) > 0:
			if len(args.palettes) != len(args.mapping):
				raise Exception("If both the --mapping and --palette-names are given, they must be given the same number of times")
			palette_names = args.palettes

			if args.verbose: print(f"Reading palette mapping(s): {args.mapping}; renaming the palettes {args.palettes}")
		else: 
			palette_names = [[]] * len(args.mapping)
			if args.verbose: print(f"Reading palette mapping(s): {args.mapping}")

		for mapping, names in zip(args.mapping, palette_names):
			mappings.append(load_palette_mapping(mapping, names=names))

	if len(args.source) > 0 and len(args.target) > 0:
		if len(mappings) > 0 and args.mode == 'product':
			print("Warning: multiple mappings were specified with both --mapping and --from/--to flags, and you " 
				"have indicated to take the product of palettes in all mappings. The order of the palettes may "
				"not be what you expect, since all mappings specified with --mapping will be evaluated before "
				"any mappings specified with --from/--to. Use option -v to see in what order the mappings are "
				"evaluated. ")
		if len(args.source) != len(args.target):
			raise Exception("If using --from and --to, must provide the same number of --from flags as --to flags.")

		for source, target in zip(args.source, args.target):
			mappings.append(make_mapping(source, target, verbose=args.verbose))

	if len(mappings) == 0:
		raise Exception('Must specify the color mapping, using either --mapping or --from and --to.')

	if args.mapping_output is not None:
		if args.verbose: print(f"Writing image representation of palette mapping to {args.mapping_output}")
		mapping_img = mapping.to_image()
		mapping_img.save(args.mapping_output)
	
	recolor(args.input, mappings, args.output, mode=args.mode, verbose=False)


def recolor(images, mappings, output_paths, mode='sum', verbose=False):
	# mapping = load_palette_map_json(args.mapping)

	if len(output_paths) == 1:
		output_paths = output_paths * len(images)
	elif len(output_paths) != len(images):
		raise Exception("Must give either one --output argument, or the same number of --output as --input arguments (one per image) \n"
			f"- Inputs: {images} \n"
			f"- Outputs: {output_paths} \n")


	for input_path, output_path_fmt in zip(images, output_paths):
		if verbose: print(f"Reading input image {input_path}...")
		input_path_basename = os.path.basename(input_path)
		input_path_basename_sans_ext, _ = os.path.splitext(input_path_basename)
		input_path_sans_ext, input_path_ext = os.path.splitext(input_path)
		input_path_ext = input_path_ext.lstrip('.')

		img = Image.open(input_path)

		def save_img(out_img, palette_name):
			output_path = format_placeholders(output_path_fmt, {
				'%B': input_path_basename,
				'%b': input_path_basename_sans_ext,
				'%i': input_path_sans_ext, 
				'%e': input_path_ext,
				'%I': input_path,
				'%p': palette_name
			})

			if verbose: print(f"- writing output from palette '{palette_name}' to {output_path}")
			mkdirpf(output_path)
			out_img.save(output_path)


		# apply each mapping in series
		if mode == 'sum':
			for mapping in mappings:

				out_imgs = mapping.recolor_image(img) #recolor_map(img, mapping)

				for (out_img, palette_name) in zip(out_imgs, mapping.names):
					save_img(out_img, palette_name)

		# apply all combinations of mappings
		elif mode == 'product':

			# start with a single image (the input image)
			# apply the first mapping to all images; collect a list of output images (one per palette)
			# then apply the next mapping to each of the accumulated output images; continue
			# until no mappings remain

			mapped_imgs = [img]
			palette_paths = ['']

			remaining_mappings = mappings[:]
			while len(remaining_mappings) > 0:

				mapping, *remaining_mappings = remaining_mappings

				if verbose: print(f"Applying mapping {repr(mapping)}")

				mapping_out_imgs = []
				mapping_palette_paths = []

				for img, palette_path in zip(mapped_imgs, palette_paths):
					mapping_out_imgs.extend(mapping.recolor_image(img)) #recolor_map(img, mapping)
					mapping_palette_paths.extend([palette_path + '-' + palette_name for palette_name in mapping.names])
					
				mapped_imgs = mapping_out_imgs
				palette_paths = mapping_palette_paths

				if verbose: print(f" -> {zip(mapped_imgs, palette_paths)}")


			for (out_img, palette_name) in zip(mapped_imgs, palette_paths):
				save_img(out_img, palette_name.lstrip('-'))

		else:
			raise Exception(f"Unsupported mapping combinator {mode}; choose from 'sum' or 'product'")


