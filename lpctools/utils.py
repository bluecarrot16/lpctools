import os
import os.path
import re
import collections

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


def unique_rows(ar):
	"""
	borrowed from scikit-image
	Copyright (C) 2019, the scikit-image team
	https://github.com/scikit-image/scikit-image/blob/65f73ee17123b13488c97a17cba661f394f284c3/skimage/util/unique.py
	"""
	import numpy as np
	import pandas as pd

	if ar.ndim != 2:
		raise ValueError("unique_rows() only makes sense for 2D arrays, "
						 "got %dd" % ar.ndim)
	# the view in the next line only works if the array is C-contiguous
	ar = np.ascontiguousarray(ar)
	# np.unique() finds identical items in a raveled array. To make it
	# see each row as a single item, we create a view of each row as a
	# byte string of length itemsize times number of columns in `ar`
	ar_row_view = ar.view('|S%d' % (ar.itemsize * ar.shape[1]))
	# _, unique_row_indices = np.unique(ar_row_view, return_index=True)
	
	ar_out = ar[unique_row_indices]
	return ar_out


def mkdirp(*paths):
	fn = os.path.join(*paths)
	os.makedirs(fn, exist_ok=True)
	return fn

def mkdirpf(*paths):
	fn = os.path.join(*paths)
	os.makedirs(os.path.dirname(fn), exist_ok=True)
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
