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

	def to_hex(self):
		return [rgb2hex(*x) for x in self._colors]

	def to_hsv(self):
		return [colorsys.rgb_to_hsv(*x) for x in self._colors]

	def sort_hue(self):
		hsvs = self.to_hsv()
		hsvs_sorted = sorted(hsvs, key=lambda h,s,v: h)
		return ImagePalette([colorsys.hsv_to_rgb(*x) for x in hsvs_sorted])

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
		data = json.parse(f)
	return ImagePalette(data, name=name)

def load_palette_png(path, name=''):
	img = Image.open(path)
	bn, _ = os.path.splitext(os.path.basename(path))

	# for indexed image, get palette directly in order
	if img.mode == 'P':
		colors = img.getpalette()
	else:
		# does not preserve order of colors in image
		# colors = [color for count, color in img.convert('RGBA').getcolors()]
		
		import pandas as pd

		img_pixels = np.array(img).reshape((-1, 4))
		# colors = unique_rows(img_pixels)
		
		# probably slow but seems like the only sane way to preserve ordering
		colors = pd.unique([tuple(a) for a in img_pixels])

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

		if isinstance(dest_palettes, dict):
			# self.names = dest_palettes.keys()
			dest_palettes = [ImagePalette(pal, name=name) for name, pal in dest_palettes.items()] #dest_palettes.values()
			# self.names = [pal.name for pal in dest_palettes]
		else: 
			# self.names = range(len(dest_palettes))
			dest_palettes = [ImagePalette(d) for d in dest_palettes]

		self.names = [pal.name or i for i,pal in enumerate(dest_palettes)]
		self.n_palettes = len(dest_palettes)

		# s[0]: [ d1[0], d2[0], ... ],
		# s[1]: [ d1[1], d2[1], ... ]

		assert all(len(source_palette) == len(d) for d in dest_palettes)

		# super().__init__(zip(source_palette, dest_palettes))
		# 
		super().__init__( (s, [ d[i] for d in dest_palettes ]) for i,s in enumerate(source_palette) )

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


def load_palette_mapping_json(data):
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
			raise Exception(f'Error: {len(names)} names provided for {dests.shape[0]}')
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
		
		if name != '':
			if verbose: print(f"- loading palette '{name}' from {path}")
			dest_pals.append(load_palette(path, name=name))
		else: 
			if verbose: print(f"- loading untitled palette from {path}")
			dest_pals.append(load_palette(pal_str))

	colormap = ImagePaletteMapping(source_pal, dest_pals)
	return colormap



def main_convertpalette(args):
	in_pal = load_palette(args.input)
	save_palette(in_pal, args.output)

def main_convertmapping(args):
	in_map = load_palette_mapping(args.input)
	save_palette_mapping(in_map, args.output)

def main_colormap(args):
	colormap = make_mapping(args.source, args.target, verbose=args.verbose)
	if args.verbose: 
		print(colormap)
		print(f"Saving mapping to {args.output}")

	save_palette_mapping(colormap, args.output)

def main_recolor(args):
	# mapping = load_palette_map_json(args.mapping)

	if args.mapping is not None:
		if args.verbose: 
			if args.palettes is not None:
				print(f"Reading palette mapping: {args.mapping}")
		mapping = load_palette_mapping(args.mapping, names=args.palettes)
	elif args.source is not None and args.target is not None:
		colormap = make_mapping(args.source, args.target, verbose=args.verbose)
	else: 
		raise Exception('Must specify the color mapping, using either --mapping or --from and --to.')

	if args.mapping_output is not None:
		if args.verbose: print(f"Writing image representation of palette mapping to {args.mapping_output}")
		mapping_img = mapping.to_image()
		mapping_img.save(args.mapping_output)



	output_paths = args.output
	if len(output_paths) == 1:
		output_paths = output_paths * len(args.input)
	elif len(output_paths) != len(args.input):
		raise Exception("Must give either one --output argument, or the same number of --output as --input arguments (one per image) \n"
			f"- Inputs: {args.input} \n"
			f"- Outputs: {output_paths} \n")

	if args.verbose: print(output_paths)

	for input_path, output_path_fmt in zip(args.input, output_paths):
		if args.verbose: print(f"Reading input image {input_path}...")
		input_path_basename = os.path.basename(input_path)
		input_path_basename_sans_ext, _ = os.path.splitext(input_path_basename)
		input_path_sans_ext, input_path_ext = os.path.splitext(input_path)
		input_path_ext = input_path_ext.lstrip('.')

		img = Image.open(input_path)

		out_imgs = mapping.recolor_image(img) #recolor_map(img, mapping)

		for (out_img, palette_name) in zip(out_imgs, mapping.names):
			output_path = format_placeholders(output_path_fmt, {
				'%B': input_path_basename,
				'%b': input_path_basename_sans_ext,
				'%i': input_path_sans_ext, 
				'%e': input_path_ext,
				'%I': input_path,
				'%p': palette_name
			})

			if args.verbose: print(f"- writing output from palette '{palette_name}' to {output_path}")
			mkdirpf(output_path)
			out_img.save(output_path)

		# [out.save(args.output + name + ".png") for (out, name) in zip(outs, mapping.names)]

