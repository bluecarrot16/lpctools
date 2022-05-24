import os
import os.path
import re
import collections
import itertools
import textwrap

def wrap_fill(text, width=79, **kwargs):
	return textwrap.fill(text, width=width, **kwargs)

_leading_whitespace_re = re.compile('(^[ \t]*)(?:[^ \t\n])')
def dedent(s):
	lines = s.splitlines()
	indents = _leading_whitespace_re.findall(lines[0])
	indent = indents[0]
	return "\n".join(line.removeprefix(indent) for line in lines)

# https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def all_equal(iterable):
	g = itertools.groupby(iterable)
	return next(g, True) and not next(g, False)

def without_keys(d, keys): return {x: d[x] for x in d if x not in keys}

def dict_strs(d):
	return { x:str(y) for x, y in d.items() }

def get_pos_dim(poly):
	_xs, _ys = zip(*poly)
	x0 = int(min(_xs))
	y0 = int(min(_ys))
	w = int(max(_xs)) - x0
	h = int(max(_ys)) - y0
	return ((x0, y0), (w, h))

def get_points_in_path(poly):
	pos, dim = get_pos_dim(poly)
	print(pos, dim)
	x0, y0 = pos
	w, h = dim

	points = []
	for _x in range(x0, x0+w):
		for _y in range(y0, y0+h):
			if is_point_in_path(_x, _y, poly): 
				point = ((_x-x0, _x), (_y-y0, _y))
				points.append(point)
	return points

def is_point_in_path(x: int, y: int, poly) -> bool:
		"""Determine if the point is in the path.

		Args:
			x -- The x coordinates of point.
			y -- The y coordinates of point.
			poly -- a list of tuples [(x, y), (x, y), ...]

		Returns:
			True if the point is in the path.
		"""
		num = len(poly)
		j = num - 1
		c = False
		for i in range(num):
				if ((poly[i][1] > y) != (poly[j][1] > y)) and \
								(x < poly[i][0] + (poly[j][0] - poly[i][0]) * (y - poly[i][1]) /
																	(poly[j][1] - poly[i][1])):
						c = not c
				j = i
		return c

def parse_named_paths(paths, default_names=None, names_required=False):
	named_paths = []

	if default_names == True:
		default_names = list(map(str, range(len(paths))))
	elif default_names is not None:
		if len(default_names) != len(paths):
			raise Exception('default_names, if given, must be same length as paths')

	for i, named_path in enumerate(paths):
		if named_path.count('=') == 1:
			name, path = named_path.split('=')
			named_paths.append((name, path))
		elif default_names is not None:
			named_paths.append((default_names[i], named_path))
		elif not names_required:
			named_paths.append((None, path))
		else:
			raise Exception("If more than one image is specified per --input group, images must be named by layer by writing LAYER_NAME=IMAGE_PATH.")
	return named_paths

# def unique_rows(ar):
# 	"""
# 	borrowed from scikit-image
# 	Copyright (C) 2019, the scikit-image team
# 	https://github.com/scikit-image/scikit-image/blob/65f73ee17123b13488c97a17cba661f394f284c3/skimage/util/unique.py
# 	"""
# 	import numpy as np
# 	import pandas as pd

# 	if ar.ndim != 2:
# 		raise ValueError("unique_rows() only makes sense for 2D arrays, "
# 						 "got %dd" % ar.ndim)
# 	# the view in the next line only works if the array is C-contiguous
# 	ar = np.ascontiguousarray(ar)
# 	# np.unique() finds identical items in a raveled array. To make it
# 	# see each row as a single item, we create a view of each row as a
# 	# byte string of length itemsize times number of columns in `ar`
# 	ar_row_view = ar.view('|S%d' % (ar.itemsize * ar.shape[1]))
# 	# _, unique_row_indices = np.unique(ar_row_view, return_index=True)
	
# 	ar_out = ar[unique_row_indices]
# 	return ar_out

def unique_rows(a):
	import numpy as np
	import pandas as pd
	
	# np.unique() is slow, in part because it sorts;
	# pd.unique() is much faster, but only 1D
	# This is inspired by https://github.com/numpy/numpy/issues/11136#issue-325345618
	# It creates a 1D view where each element is a byte-encoding of a row, then uses
	# pd.unique(), and then reconstruct the original type.
	if a.ndim != 2:
		raise ValueError(f'bad array dimension {a.ndim}; should be 2')
	b = np.ascontiguousarray(a).view(
		np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
	)[:, 0]
	return pd.unique(b).view(a.dtype).reshape(-1, a.shape[1])


def mkdirp(*paths):
	fn = os.path.join(*paths)
	os.makedirs(fn, exist_ok=True)
	return fn

def mkdirpf(*paths):
	fn = os.path.join(*paths)
	dn = os.path.dirname(fn)
	if dn != '':
		os.makedirs(dn, exist_ok=True)
	return fn

def format_placeholders(template, placeholders, special='%'):
	"""replaces a set of named placeholders in a template string"""

	for placeholder, value in placeholders.items():
		regex = '(?<!' + re.escape(special) + ')' + re.escape(placeholder)
		template = re.sub(regex, str(value), template)

	template = template.replace(special * 2, special)
	return template

assert format_placeholders('%b-%p.png',{'%b': 'test', '%p': 'palette'}) == 'test-palette.png'
assert format_placeholders('%b-%i.png',{'%b': 'test', '%i': 1}) == 'test-1.png'
assert format_placeholders('%%-%b-%p.png',{'%b': 'test', '%p': 'palette'}) == '%-test-palette.png'
assert format_placeholders('%%b-%p.png',{'%b': 'test', '%p': 'palette'}) == '%b-palette.png'


def pattern_to_regex(pattern, placeholders={}, special='%'):
	placeholder_patterns = collections.defaultdict(lambda: r'.+', placeholders)

	return re.sub( '(?<!' + re.escape(special) + ')' + re.escape(special) + r'(\w)', 
		lambda name: r'(?P<' + name.group(1) + '>' + placeholder_patterns[name.group(1)] + ')',
		pattern)

assert pattern_to_regex('%n-%d-%f') == '(?P<n>.+)-(?P<d>.+)-(?P<f>.+)'
assert pattern_to_regex('%n-%d%f',placeholders={'f':r'\d+','d':r'\D+'}) == '(?P<n>.+)-(?P<d>\\D+)(?P<f>\\d+)'

# import argparse
# class ExtendActionOverwriteDefault(argparse._AppendAction):
# 	def __call__(self, parser, namespace, values, option_string=None):
# 		items = getattr(namespace, self.dest, None)
# 		print("items:")
# 		print(items)

# 		if items == self.default:
# 			setattr(namespace, self.dest, items)
# 		else:
# 			items = argparse._copy_items(items)
# 			items.extend(values)
# 			setattr(namespace, self.dest, items)

def composite_images(images, inplace=True):
	"""composites each image in images on top of one another, in order
	"""
	if len(images) == 0:
		return None
	elif len(images) == 1 :
		return images[0]
	else:
		if inplace:
			base, *layers = images
		else: 
			base = Image.new(images[0].mode, images[0].size, color='#ffffff00')
			layers = images

		for img in layers:
			base.alpha_composite(img)
	return base

def listify(s):
	if not isinstance(s,collections.abc.Iterable) or isinstance(s, str):
		return [s]
	else: return s

def get_color_hue_range(N, s=0.8, l=0.7):
	import colorsys
	HSV_tuples = [(x*1.0/N, s, l) for x in range(N)]
	RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
	return [tuple(int(255*c) for c in t) for t in RGB_tuples]


