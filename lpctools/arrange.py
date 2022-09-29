import os
import os.path
from glob import glob
from pathlib import Path
import itertools

from PIL import Image
import collections.abc
import numpy as np

# for a sheet:
# 	for each animation:
#		for each animation_layer:
#			assemble frames from images
#			offset each frame according to template
#			mask each frame according to template
#		combine all layers for each frame
#	concatenate animations
# 

from .recolor import Color
from .utils import *


COLOR_TRANSPARENT = Color(255,255,255,0)


class ImageCollection(dict):

	def pick_image():
		pass

	def apply_templates():
		pass

class FrameTemplate():
	"""
	A FrameTemplate is a combination of a mask and offset coordinates
	"""

	# number of frames
	# frame size
	# offset for each frame
	def __init__(self, offset=None, mask=None, frame_size=(64,64)):

		self.frame_size = frame_size

		if offset is None:
			self.offset = (frame_size[0]//2,frame_size[1]//2)
		else: 
			self.offset = offset
			if self.offset[0] > self.frame_size[0] or self.offset[1] > self.frame_size[1]:
				raise Exception(f"Offset {offset} outside the bounds of frame size {frame_size}")

		if mask is None:
			self.mask = Image.new(mode='RGBA',size=self.frame_size,color=COLOR_TRANSPARENT)
		else:
			self.mask = mask
			if mask.size != self.frame_size:
				raise Exception(f"Mask size {mask.size} != frame size {frame_size}")

		if self.offset is None and self.mask is None:
			raise Exception('Must set either `masks`, `offsets`, or both!')

	def apply(self, img, tmp_img=None, transparent_img=None):
		box_whole = (0,0, self.frame_size[0], self.frame_size[1])

		if tmp_img is None: 
			tmp_img = Image.new(mode='RGBA',size=self.frame_size,color=COLOR_TRANSPARENT)
		else:
			tmp_img.paste(COLOR_TRANSPARENT, box=box_whole)
		
		if transparent_img is None: transparent_img = Image.new(mode='RGBA', size=self.frame_size)

		# might be None if `pick_image` didn't find any images for this layer
		if img is not None:

			# offset coordinates are w/r/t the middle of img
			box = (self.offset[0] - img.size[0]//2, 
				   self.offset[1] - img.size[1]//2)			
			tmp_img.paste(img, box=box)


		return Image.composite(transparent_img, tmp_img, self.mask)

	@staticmethod
	def from_images(offset=None, mask=None, mask_colors=['#ffffff'], **kwargs):
		_offset = None
		if offset is not None: 
			bbx = offset.getbbox()

			# if image is completely empty, getbbox() returns None, so use default offset
			if bbx is not None:
				_offset = bbx[:2]

		_mask = None
		if mask is not None:
			if not isinstance(mask_colors,np.ndarray):
				mask_colors = [Color(mask_color).to_array() for mask_color in mask_colors]  #[np.newaxis, np.newaxis, :]
			
			arr = np.logical_or.reduce( [(np.array(mask) == mask_color).all(axis=-1) for mask_color in mask_colors] )
			_mask = Image.fromarray( (arr.astype('uint8') * 255) , mode='L')

		return FrameTemplate(_offset, _mask, **kwargs)


def pick_image(afi, images, verbose=False):
	""" picks the best image corresponding to (animation_name, direction, frame), from a collection of images
	"""

	choices = [afi, (afi[0], afi[1], None), (None, afi[1], afi[2]), (None, afi[1], None), (None, None, None)]
	for c in choices:
		if c in images:
			if verbose: print(f"PICK {afi} --> {c}  '{images[c].filename}'")
			return images[c]

	if verbose: print(f"miss {afi}")

# def get_animation_templates(animations, offsets_image, masks_image, layout, verbose=False, **kwargs):
# 	offsets_images = layout.unpack_images(offsets_image)
# 	masks_images = layout.unpack_images(masks_image)

# 	templates = []
# 	for animation in animations:
# 		templates.append(
# 			AnimationTemplate.from_images(
# 				offsets = [pick_image(afi, offsets_images, verbose=verbose) for afi in animation.frames],
# 				masks = [pick_image(afi, masks_images, verbose=verbose) for afi in animation.frames],
# 				**kwargs
# 			)
# 		)
# 	return templates


def get_frame_templates_from_images(layout, offsets_image=None, masks_image=None, verbose=False, **kwargs):
	""" splits a masks_image and/or offsets_image according to a `layout`, then produces a FrameTemplate for 
	each frame in the mapping, i.e. templates[afi] = FrameTemplate()
	"""

	if offsets_image is None:
		offsets_images = {}
	else: 
		offsets_images = layout.unpack_images(offsets_image)

	if masks_image is None:
		masks_images = {}
	else: 
		masks_images = layout.unpack_images(masks_image)

	templates = {}
	for afi in layout:
		templates[afi] = FrameTemplate.from_images(
			offset = pick_image(afi, offsets_images, verbose=verbose),
			mask = pick_image(afi, masks_images, verbose=verbose),
			frame_size = layout.frame_size,
			**kwargs
		)
	return templates



def distribute_images(images, templates, positions, verbose=False):
	""" for each `afi` in `positions`: , picks a suitable image from `images` and applies `templates[afi]`;
	returns a mapping of { `afi`:image }
	"""

	images_distributed = {}
	for afi in positions:
		img = pick_image(afi, images, verbose=verbose)
		template = templates[afi]

		if template is not None:
			images_distributed[afi] = template.apply(img)
		else: 
			images_distributed[afi] = img

	return images_distributed

def distribute_images_via_layout(images, layout, offsets_image, masks_image, verbose=False):
	"""calculates templates from offsets_image and masks_image using `layout`, then distributes
	`images` and repacks into `layout`
	"""
	templates = get_frame_templates_from_images(
		layout = layout,
		offsets_image = offsets_image, 
		masks_image = masks_image)

	images_distributed = distribute_images(images, templates, positions = layout, verbose=verbose)

	return layout.pack_images(images_distributed)





class Animation(collections.namedtuple('_AnimationTuple',['name','direction','nframes'])):
	@property
	def frames(self):
		for i in range(self.nframes):
			yield AnimationFrameID(self.name, self.direction, i)

	@staticmethod
	def make_directions(name, nframes, directions=['n','w','s','e']):
		return {direction: Animation(name,nframes,direction) for direction in directions}

animations = {
	'cast': Animation.make_directions('cast',7),
	'thrust': Animation.make_directions('thrust',8),
	'walk': Animation.make_directions('walk',9),
	'slash': Animation.make_directions('slash',6),
	'shoot': Animation.make_directions('shoot',13),
	'hurt': Animation.make_directions('hurt',6,['s']),
	'grab': Animation.make_directions('grab',3),
	'push': Animation.make_directions('push',9),
}

animation_synonyms = {
	'spellcast':'cast',
	'magic':'cast',
	'swing':'slash'
}


class AnimationFrameID(collections.namedtuple('_AnimationFrameID',['name','direction','frame'])):
	__slots__ = ()

	def __new__(cls, name=None,direction=None,frame=None):
		if name in animation_synonyms: 
			name = animation_synonyms[name]

		if frame is not None: 
			if isinstance(frame, str) and frame in 'ABCDEF':
				frame = int(frame, 16)
			else: frame = int(frame)
		return super().__new__(cls, name, direction, frame)

	def format(self,pattern):
		return format_placeholders(pattern,{
			'%n':self.name,
			'%d':self.direction,
			'%f':self.frame
		})

	def to_dict(self):
		return dict(name=self.name, direction=self.direction, frame=int(self.frame))

	@staticmethod
	def from_dict(d):

		return AnimationFrameID(
			d['name']      if 'name'      in d else d['n'] if 'n' in d else None,
			d['direction'] if 'direction' in d else d['d'] if 'd' in d else None,
			d['frame']     if 'frame'     in d else d['f'] if 'f' in d else None)


class SpritesheetLayout():
	def __init__(self, animation_positions, size=None, frame_size=(64,64)):
		assert isinstance(animation_positions, collections.abc.Mapping)

		self.inverse_positions = { tuple(pos) : AnimationFrameID(*afi) for pos, afi in animation_positions.items() }
		self.positions = { AnimationFrameID(*afi): tuple(pos) for pos, afi in reversed(animation_positions.items()) }

		assert (len(pos) == 2 for pos in self.positions.values())

		# zip acts like transpose; +1 because if len(x) == i, largest index should be i-1 
		xs, ys = zip(*self.positions.values())
		_size = np.array((max(xs)+1, max(ys)+1))

		# if size is given, check that no elements exceed the provided bounds
		if size is not None:
			size = np.array(size, dtype=int)
			if any(_size > size):
				elems_out_of_range = [afi for afi, pos in self.items() if (np.array(pos) > size).any()]
				raise Exception(f"The following Animation Frames are out of the indicated size of {size}: {elems_out_of_range}")
		else:
			# otherwise, use calculated size
			size = _size
		self.size = tuple(int(s) for s in size)
		self.frame_size = tuple(frame_size)

	def __eq__(self, other):
		return (self.frame_size == other.frame_size) and (self.positions == other.positions)

	def __iter__(self):
		yield from self.positions

	def __len__(self):
		return len(self.positions)

	def items(self):
		return self.positions.items()

	def get_animations(self):
		animations = collections.defaultdict(lambda: collections.defaultdict(int))
		afis = self.inverse_positions.values()

		for (name, direction, frame) in afis:
			animations[name][direction] = max(animations[name][direction], frame)

		# print(animations)
		# return [Animation(name, direction, nframes) for nframes, direction in directions.items() for (name, directions) in animations.items()]
		anims_list = []
		for (name, directions) in animations.items():
			for direction, last_frame in directions.items():
				anims_list.append(Animation(name, direction, last_frame+1))

		return anims_list

	def get_pos(self, afi):
		return self.positions[afi]

	def get_pixel_pos(self, afi):
		pos = self.get_pos(afi)
		return (pos[0]*self.frame_size[0], 
				pos[1]*self.frame_size[1])

	@property
	def pixel_size(self):
		return (self.size[0] * self.frame_size[0],
				self.size[1] * self.frame_size[1])

	def pack_images(self, images, verbose=True):
		assert isinstance(images, collections.abc.Mapping)
		images = { AnimationFrameID(*afi): img for afi, img in images.items() }

		new_img = Image.new('RGBA',self.pixel_size, color=COLOR_TRANSPARENT)

		for pos, afi in self.inverse_positions.items():
			if afi not in images or images[afi] == None and verbose:
				print(f"Warning: missing {afi} for this layout")
			else:
				img = images[afi]
				if img.size != self.frame_size:
					print(f"Warning: image size {img.size} != layout frame size {frame_size}")
					tl = self.get_pixel_pos(afi)
					pos = (
						tl[0] + self.frame_size[0]//2 - img.size[0]//2,
						tl[1] + self.frame_size[1]//2 - img.size[1]//2
					)
				else: pos = self.get_pixel_pos(afi)
				new_img.paste(img, pos)

		# for afi in images.keys():
		# 	if afi not in self.positions:
		# 		raise Exception(f"Do not know how to place animation '{afi}' in this layout")


		# for afi, img in images.items():
		# 	new_img.paste(img, self.get_pixel_pos(afi))

		return new_img

	def unpack_images(self, img, verbose=True):
		if img.size != self.pixel_size:
			if img.size[0] < self.pixel_size[0] or img.size[1] < self.pixel_size[1]:
				raise Exception(f"Image {img.filename} is smaller than layout; Image size: {img.size}, layout size: {self.pixel_size}")
			else: 
				print(f"Warning: image {img.filename} is larger than layout; right- and/or bottom- edge of image will be trimmed. Image {img.filename} size: {img.size}, layout size: {self.pixel_size}")

		output = {}
		for afi, pos in self.positions.items():
			(x, y) = self.get_pixel_pos(afi)
			bbox = (x, y, x+self.frame_size[0], y+self.frame_size[0])
			sub_img = img.crop( bbox )
			setattr(sub_img,'filename', f"{img.filename}#({x},{y})={afi}")
			output[afi] = sub_img

		return output

	def to_array(self):
		out = np.empty(shape=self.size, dtype='object')
		for afi, pos in self.positions.items():
			out[pos] = afi
		return out

	def to_dict(self):
		arr = self.to_array()
		out = []

		for y in range(arr.shape[1]):
			out.append([])
			for x in range(arr.shape[0]):
				c = None if arr[x,y] is None else arr[x,y].to_dict()
				out[y].append(c)
		return dict(
			frame_size=tuple(int(c) for c in self.frame_size), 
			size=tuple(int (c) for c in self.size), rows=out)

	def to_json(self, path):
		import json
		with open(path, 'w') as fp:
			json.dump(self.to_dict(), fp)

	def to_image(self, path=None):
		from PIL import ImageDraw, ImageFont
		anims = self.get_animations()
		colors = dict(zip([a.name for a in anims], get_color_hue_range(len(anims))))

		img = Image.new(mode='RGBA',size=self.pixel_size,color=COLOR_TRANSPARENT)
		draw = ImageDraw.Draw(img)

		for afi, pos in self.positions.items():
			px_pos = self.get_pixel_pos(afi)
			px_box = (px_pos,(px_pos[0]+self.frame_size[0]-1, px_pos[1]+self.frame_size[1]-1))
			color = colors[afi.name]
			text=f"{afi.name}\n{afi.direction}\n#{afi.frame}"
			font = ImageFont.load_default()

			draw.rectangle(px_box, outline=color)
			# bbox = draw.multiline_textsize(text, font=font)
			draw.text((px_pos[0]+2, px_pos[1]+1), text=text, fill=color, font=font, spacing=2)

			# draw.multiline_text(, test=text, fill=color)

		if path is not None:
			img.save(path)
		return img

	@staticmethod
	def from_array(arr, **kwargs):
		out = {}
		for i, row in enumerate(arr):
			for j, afi in enumerate(row):
				if afi is not None:
					# follow image convention, where (x = col, y = row)
					# out[afi] = (j, i) #(i, j)
					out[(j, i)] = afi  #(i, j)
		return SpritesheetLayout(out, **kwargs)

	@staticmethod
	def from_rows(rows, **kwargs):
		"""each row is a list of tuples or dicts or `null`s; 
		each tuple is either (animation, direction, frame : int) or (animation, direction, frames : iterable) : ); 
		equivalently, a dict of {'name', 'direction', 'frame'} or {'name', 'direction', 'frames'}. 
		where ``frames`` is given, the row will then be populated at that position with
			[AnimationFrameID(animation, direction, 0), AnimationFrameID(animation, direction, 1), ... AnimationFrameID(animation, direction, nframes-1)] 

		"""
		
		out_rows = []
		for row in rows:
			if (isinstance(row, tuple) and len(row) == 3) or (row is None):
				row = [row]
			
			out_row = []
			for c in row:
				if c is not None:
					if isinstance(c, dict):
						if 'frames' in c:
							if isintance(c['frames'], 'str'):
								_frame = c['frames']
							else:
								_frame = range(c['frames'])
						else:
							_frame = c.get('frame', None)
						c = (c.get('name',None), c.get('direction',None), _frame)

					assert len(c) == 3, "rows must be a list of lists of (3-length tuples) or None"
					if isinstance(c[2], str):
						c2 = c[2].split(':')
						if len(c) > 1:
							c = (c[0],c[1]) + (range(*c2),)
						else: 
							raise Exception(f"Unknown entry {c}; expected a 3-length tuple of [animation_name, direction, frame(s)]. "
								"Any value can be `null`/`None`, but all must be included. Frame(s) can be a single number, indicating a single frame, "
								"A list of frames, or a string indicating a range of frames. If `frame(s)` is a string, "
								"and must be given as `{start_frame}:{end_frame+1}`, e.g. `'0:5'` = [0, 1, 2, 3, 4].")

					if isinstance(c[2], collections.abc.Iterable):
						out_row.extend([ (c[0], c[1], i) for i in c[2] ])
					else:
						out_row.append(c)
				else: out_row
			out_rows.append(out_row)

		return SpritesheetLayout.from_array(out_rows, **kwargs)


	@staticmethod
	def from_animation(name, nframes, directions=['n','w','s','e'], **kwargs):
		return SpritesheetLayout.from_rows( [(name, direction, range(nframes)) for direction in directions] , **kwargs)

	# def from_animations(animations):
		# return SpritesheetLayout.from_rows( [(name, direction, range(nframes)) for direction in directions] )
		


layouts = {
	'universal': SpritesheetLayout.from_rows([
			('cast'   , 'n' , range(7))  ,
			('cast'   , 'w' , range(7))  ,
			('cast'   , 's' , range(7))  ,
			('cast'   , 'e' , range(7))  ,
			('thrust' , 'n' , range(8))  ,
			('thrust' , 'w' , range(8))  ,
			('thrust' , 's' , range(8))  ,
			('thrust' , 'e' , range(8))  ,
			('walk'   , 'n' , range(9))  ,
			('walk'   , 'w' , range(9))  ,
			('walk'   , 's' , range(9))  ,
			('walk'   , 'e' , range(9))  ,
			('slash'  , 'n' , range(6))  ,
			('slash'  , 'w' , range(6))  ,
			('slash'  , 's' , range(6))  ,
			('slash'  , 'e' , range(6))  ,
			('shoot'  , 'n' , range(13)) ,
			('shoot'  , 'w' , range(13)) ,
			('shoot'  , 's' , range(13)) ,
			('shoot'  , 'e' , range(13)) ,
			('hurt'   , 's' , range(6))
		]),
		'universal-idle': SpritesheetLayout.from_rows([
			[ ('cast'   , 'n' , range(7)) ] ,
			[ ('cast'   , 'w' , range(7)) ] ,
			[ ('cast'   , 's' , range(7)) ] ,
			[ ('cast'   , 'e' , range(7)) ] ,
			[ ('thrust' , 'n' , range(8)) ] ,
			[ ('thrust' , 'w' , range(8)) ] ,
			[ ('thrust' , 's' , range(8)) ] ,
			[ ('thrust' , 'e' , range(8)) ] ,
			[ ('idle'   , 'n' , 0 ), ('walk'   , 'n' , range(8)) ],
			[ ('idle'   , 'w' , 0 ), ('walk'   , 'w' , range(8)) ],
			[ ('idle'   , 's' , 0 ), ('walk'   , 's' , range(8)) ],
			[ ('idle'   , 'e' , 0 ), ('walk'   , 'e' , range(8)) ],
			[ ('slash'  , 'n' , range(6))  ],
			[ ('slash'  , 'w' , range(6))  ],
			[ ('slash'  , 's' , range(6))  ],
			[ ('slash'  , 'e' , range(6))  ],
			[ ('shoot'  , 'n' , range(13)) ],
			[ ('shoot'  , 'w' , range(13)) ],
			[ ('shoot'  , 's' , range(13)) ],
			[ ('shoot'  , 'e' , range(13)) ],
			[ ('hurt'   , 's' , range(6))  ]
		]),
	'evert': SpritesheetLayout.from_rows([
			[('cast'   , 'n' , range(7))],
			[('cast'   , 'w' , range(7))],
			[('cast'   , 's' , range(7))],
			[('cast'   , 'e' , range(7))],

			[('thrust' , 'n' , range(8)), None, None, ('run' , 'n' , range(8))],
			[('thrust' , 'w' , range(8)), None, None, ('run' , 'w' , range(8))],
			[('thrust' , 's' , range(8)), None, None, ('run' , 's' , range(8))],
			[('thrust' , 'e' , range(8)), None, None, ('run' , 'e' , range(8))],
			[('walk'   , 'n' , range(9)), ('carry', 'n', range(9))],
			[('walk'   , 'w' , range(9)), ('carry', 'w', range(9))],
			[('walk'   , 's' , range(9)), ('carry', 's', range(9))],
			[('walk'   , 'e' , range(9)), ('carry', 'e', range(9))],
			[('slash'  , 'n' , range(6)), ('grab'  , 'n' , range(3)), ('push'  , 'n' , range(9))],
			[('slash'  , 'w' , range(6)), ('grab'  , 'w' , range(3)), ('push'  , 'w' , range(9))],
			[('slash'  , 's' , range(6)), ('grab'  , 's' , range(3)), ('push'  , 's' , range(9))],
			[('slash'  , 'e' , range(6)), ('grab'  , 'e' , range(3)), ('push'  , 'e' , range(9))],
			[('shoot'  , 'n' , range(13)), ('jump'  , 'n' , range(5))],
			[('shoot'  , 'w' , range(13)), ('jump'  , 'w' , range(5))],
			[('shoot'  , 's' , range(13)), ('jump'  , 's' , range(5))],
			[('shoot'  , 'e' , range(13)), ('jump'  , 'e' , range(5))],
			[('hurt'   , 's' , range(6))]
		]),
		'basxto': SpritesheetLayout.from_rows([
			('hurt'   , 's' , range(6))  ,
			('walk'   , 'n' , range(9))  ,
			('walk'   , 'w' , range(9))  ,
			('walk'   , 's' , range(9))  ,
			('walk'   , 'e' , range(9))  ,
			('slash'  , 'n' , range(6))  ,
			('slash'  , 'w' , range(6))  ,
			('slash'  , 's' , range(6))  ,
			('slash'  , 'e' , range(6))  ,
			('cast'   , 'n' , range(7))  ,
			('cast'   , 'w' , range(7))  ,
			('cast'   , 's' , range(7))  ,
			('cast'   , 'e' , range(7))  ,
			('thrust' , 'n' , range(8))  ,
			('thrust' , 'w' , range(8))  ,
			('thrust' , 's' , range(8))  ,
			('thrust' , 'e' , range(8))  ,
			('shoot'  , 'n' , range(13)) ,
			('shoot'  , 'w' , range(13)) ,
			('shoot'  , 's' , range(13)) ,
			('shoot'  , 'e' , range(13)) ,
			('gun'  , 'n' , range(9)) ,
			('gun'  , 'w' , range(9)) ,
			('gun'  , 's' , range(9)) ,
			('gun'  , 'e' , range(9)) ,
			('jump'  , 'n' , range(13)) ,
			('jump'  , 'w' , range(13)) ,
			('jump'  , 's' , range(13)) ,
			('jump'  , 'e' , range(13)) ,
			('grab'  , 'n' , range(3)),
			('grab'  , 'w' , range(3)),
			('grab'  , 's' , range(3)),
			('grab'  , 'e' , range(3)),
			('run' , 'n' , range(8)), 
			('run' , 'w' , range(8)), 
			('run' , 's' , range(8)), 
			('run' , 'e' , range(8))
		]),
	'sit': SpritesheetLayout.from_rows([
			[('sit-ground' , 'n' , 0), ('sit-cross' , 'n' , 0), ('sit-chair' , 'n' , 0)],
			[('sit-ground' , 'w' , 0), ('sit-cross' , 'w' , 0), ('sit-chair' , 'w' , 0)],
			[('sit-ground' , 's' , 0), ('sit-cross' , 's' , 0), ('sit-chair' , 's' , 0)],
			[('sit-ground' , 'e' , 0), ('sit-cross' , 'e' , 0), ('sit-chair' , 'e' , 0)],
		]),
	'cast': SpritesheetLayout.from_animation('cast',7),
	'thrust': SpritesheetLayout.from_animation('thrust',8),
	'walk': SpritesheetLayout.from_animation('walk',9),
	'walk-noidle': SpritesheetLayout.from_animation('walk',8),
	'idle': SpritesheetLayout.from_animation('idle',2),
	'idle1': SpritesheetLayout.from_animation('idle',1),
	'idle2': SpritesheetLayout.from_animation('idle',2),
	'idle3': SpritesheetLayout.from_animation('idle',3),
	'slash': SpritesheetLayout.from_animation('slash',6),
	'shoot': SpritesheetLayout.from_animation('shoot',13),
	'hurt': SpritesheetLayout.from_animation('hurt',6,['s']),
	'grab': SpritesheetLayout.from_animation('grab',3),
	'push': SpritesheetLayout.from_animation('push',9),
	'carry': SpritesheetLayout.from_animation('carry',9),
	'jump': SpritesheetLayout.from_animation('jump',5),
	'run': SpritesheetLayout.from_animation('run',8),
	'gun': SpritesheetLayout.from_animation('gun',9),
	'demux': SpritesheetLayout.from_rows([
			[('cast', 's', 0), ('cast', 'w', 0), ('cast', 'n', 0), ('cast', 'e', 0)]  ,
			[('hurt' , 's' , 2), ('hurt' , 's' , 3), ('hurt' , 's' , 4), ('hurt' , 's' , 5) ] 
		], frame_size=(128, 128)),
	'heads': SpritesheetLayout.from_rows([
			[(None,'n',None)],
			[(None,'w',None),('cast','w',1),('cast','w',2)],
			[(None,'s',None),('cast','s',1),('cast','s',2)],
			[(None,'e',None),('cast','e',1),('cast','e',2)],
			[('hurt','s',range(3))],
			[('hurt','s',range(3,6))],
		])
}


def load_layout(layout, **kwargs):
	"""
	Accepts a name for an existing layout or a file path to a JSON layout file and creates a SpritesheetLayout object.

	layout: str or SpritesheetLayout

	"""
	if isinstance(layout, SpritesheetLayout):
		return layout

	if layout in layouts:
		return layouts[layout]
	else:
		path = layout
		basename, ext = os.path.splitext(path)

		layout_loaders = {
			'.json': load_layout_json
		}

		if ext in layout_loaders:
			return layout_loaders[ext](path, **kwargs)

		raise Exception(f"{layout} not a built-in layout or path to a layout folder I know how to open. "
			f"Layout file formats: {layout_loaders.keys()}; built-in layouts: {layouts.keys()}")

def load_layout_json(path):
	import json
	if isinstance(path, str):
		with open(path) as f:
			data = json.load(f)

		return SpritesheetLayout.from_rows(**data)


def save_layout(layout, path, **kwargs):
	basename, ext = os.path.splitext(path)
	layout_savers = {
		'.png': lambda path, **kwargs: layout.to_image(path, **kwargs),
		'.json': lambda path, **kwargs: layout.to_json(path, **kwargs),
	}

	if ext in layout_savers:
		layout_savers[ext](path, **kwargs)
	else:
		raise Exception(f'Do not know how to save a layout to a {ext} file. Possible extensions: {layout_savers.keys()}')



IMAGE_FRAME_PATTERN = '%n-%d-%f.png'


def load_images(image_paths, pattern=IMAGE_FRAME_PATTERN, 
	frame_pattern=re.compile(r'(?P<n>[^\dABCDEF]+)(?P<f>[\dABCDEF]+)?'), 
	sep='-', verbose=False):
	"""
	loads images from a set of directories and produces a dict mapping `AnimationFrameID`s to `PIL.Image`s
	"""
	
	if not isinstance(pattern, re.Pattern):
		regex = re.compile(
			pattern_to_regex(pattern, placeholders={'f':r'\d+','d':r'\D+'})
		)
	else: 
		regex = pattern
		pattern = pattern.pattern
	if verbose: print(f"Searching pattern '{pattern}'")


	# if the pattern contains a path separator, apply the pattern to the full image path
	# otherwise, only apply the basename
	if not os.path.sep in pattern:
		image_names = (os.path.basename(f) for f in image_paths)
	else: image_names = image_paths

	# images = {AnimationFrameID.from_dict(regex.match(name).groupdict()) : Image.open(path) for name, path in zip(image_names, image_paths)}
	images = {}
	for name, path in zip(image_names, image_paths):
		m = regex.match(name)

		if m is None: 
			if verbose: print(f"- skip  {path} which does not fit pattern...")
			continue

		id = m.groupdict()

		# if regex contains named capture group "frames", it means the
		# filename refers to multiple frames, e.g. e-cast1-shoot.png, etc.
		if 'frames' in id and id['frames'] is not None:
			if verbose: print(f"- MULTI {path} --> ...")

			# separate these frames, open an Image and build an AFI for each
			frames = id['frames'].split(sep)
			for frame in frames:
				m = frame_pattern.match(frame)
				if m is None:
					if verbose: print(f"  - skip  {frame}")
				else:
					afi = AnimationFrameID.from_dict({**id, **m.groupdict()})
					images[afi] = Image.open(path)
					if verbose: print(f"  - FOUND {frame} = {path} --> {afi}")
			continue


		afi = AnimationFrameID.from_dict(id)
		images[afi] = Image.open(path)
		if verbose:
			print(f"- FOUND {path} --> {afi}")

	return images

def mirror_images(images, from_direction='e', to_direction='w', orientation='h', verbose=False):
	"""
	Identifies image frames with direction `from_direction` and mirrors them with the given `orientation` 
	to produce image frames in `to_direction`. Generally this is used to mirror east-facing images
	to make west-facing images.
	"""
	new_images = images.copy()

	for afi, image in images.items():
		if afi.direction == from_direction:
			new_afi = AnimationFrameID(afi.name, to_direction, afi.frame)
			if verbose: print(f"{afi} --({orientation})-> {new_afi}")
			new_images[new_afi] = image.transpose(Image.FLIP_LEFT_RIGHT if orientation == 'h' else PIL.Image.Transpose.FLIP_TOP_BOTTOM)
	return new_images

def pack_animations(image_paths, layout, output=None, pattern=IMAGE_FRAME_PATTERN, verbose=False):
	layout = load_layout(layout)

	images = load_images(image_paths, pattern)
	img = layout.pack_images(images)

	if output is not None:
		img.save(output)

	return img

def main_pack(args):
	return pack_animations(args.images, args.layout, args.output, args.pattern)

def unpack_animations(image, layout, pattern=IMAGE_FRAME_PATTERN, output_dir='.', verbose=False):
	img = Image.open(image)
	layout = load_layout(layout)

	images = layout.unpack_images(img, verbose=verbose)

	if pattern is not None:
		mkdirp(output_dir)
		for afi, img in images.items():
			outfile = mkdirpf(output_dir, afi.format(pattern))
			img.save(outfile)

	return images

def main_unpack(args):
	return unpack_animations(args.input, args.layout, args.pattern, args.output_dir, verbose=args.verbose)

def repack_animations(images, from_layouts, to_layouts, output_dir='.', output_pattern=None, mirror=False, verbose=False):
	images = listify(images)

	from_layouts = listify(from_layouts)

	if len(from_layouts) != len(images):
		raise Exception("Must specify same number of source layouts as images. Source layouts: {from_layouts}; images: {images}")

	if verbose: 
		print("Input images: {images}")
		print("Reading from layouts: {from_layouts}")

	unpacked_images = {}
	for image_path, from_layout in zip(images, from_layouts):
		if verbose: print(f"{image_path} -> {from_layout}")
		img = Image.open(image_path)
		from_layout = load_layout(from_layout)
		unpacked_images.update( from_layout.unpack_images(img) )
		if verbose: print(f"= {len(unpacked_images)} images total")

	if mirror:
		orientation = {'e': 'h', 'w':'h', 'n':'v', 's':'v'}[mirror[1]]
		unpacked_images = mirror_images(unpacked_images, 
			from_direction=mirror[0], to_direction=mirror[1], orientation=orientation, verbose=verbose)

	if verbose: print("Writing to layouts: {to_layouts}")

	if output_pattern is None:
		output_pattern = str(Path(output_dir) / "%l.png")

	for layout_name in to_layouts:
		layout = load_layout(layout_name)
		new_img = layout.pack_images(unpacked_images)
		outfile = mkdirpf(format_placeholders(output_pattern, {'%l':layout_name}))
		if verbose: print(f"- Saved {layout_name} -> {outfile}")
		new_img.save(outfile)


def parse_mirror(arg_mirror):
	if arg_mirror:
		if arg_mirror == True or arg_mirror == 'true':
			mirror = ('e','w')
		else:
			mirror = tuple(arg_mirror.split(':'))


def main_repack(args):
	
	return repack_animations(args.input, args.from_layouts, args.to_layouts, 
		output_dir=args.output_dir, output_pattern=args.output_pattern,
		mirror=parse_mirror(args.mirror), verbose=args.verbose)


def separate(images, from_layouts, verbose=False, **kwargs):
	animations = set()
	for from_layout in from_layouts:
		from_layout = load_layout(from_layout)
		animations.update(from_layout.get_animations())
	names = [a.name for a in animations]
	if verbose:
		print(f"Detected animations {names}")

	# to_layouts = [load_layout(name) for name in names]
	to_layouts = names

	return repack_animations(images, from_layouts, to_layouts, verbose=verbose, **kwargs)


def main_separate(args):
	separate(args.input, args.from_layouts, 
		output_dir=args.output_dir, output_pattern=args.output_pattern, 
		mirror=parse_mirror(args.mirror), verbose=args.verbose)

def combine(inputs, layout, output=None, verbose=False):
	unpacked_images = {}

	def guess_layout(img_path):
		basename, ext = os.path.splitext(os.path.basename(img_path))
		basename = basename.lower()
		if basename in layouts:
			return basename
		elif basename in animation_synonyms:
			return animation_synonyms[basename]
		else:
			return None

	def guess_layout_and_load_img(img_path):
		from_layout = guess_layout(img_path)
		if from_layout is not None:
			if verbose:
				print(f"{img_path} -> layout {from_layout}")
			img = Image.open(img_path)
			from_layout = load_layout(from_layout)
			unpacked_images.update( from_layout.unpack_images(img) )

	for p in inputs:
		if os.path.isdir(p):
			for img_path in glob(str(Path(p) / '*.png')):
				guess_layout_and_load_img(img_path)

		else:
			guess_layout_and_load_img(p)


	to_layout = load_layout(layout)
	img = to_layout.pack_images(unpacked_images)

	if output is not None:
		img.save(output)

	return img


def main_combine(args):
	combine(args.input, args.layout, args.output)



MULTI_FRAME_IMAGE_REGEX = r'(?P<d>(?!bg-)(?!behindbody-)[^\-]+)(?:-(?P<frames>.*))?.png'

distribute_layers = {
	'bg':         { 'pattern': re.compile('bg-'+MULTI_FRAME_IMAGE_REGEX),
					# 'pattern': re.compile(r'bg-(?P<d>[^\-]+)(?:-(?P<n>\D+)(?P<f>\d+)?)?.png'),
					'help': ('Images named like bg-DIRECTION[-SUFFIX] (e.g. bg-n.png, bg-s-hurt1.png, bg-s-shoot2-shoot3.png, etc.) '
						"will be behind the character's entire body. This is useful for things like long hair behind the character's body, different amounts of which will be exposed depending on the body movements in different animations. (See e.g. the 'ponytail2' hairstyle.)"	
						),
					'mask_colors':['#808080','#C0C0C0','#ffffff'] },
	'behindbody': { 'pattern': re.compile('behindbody-'+MULTI_FRAME_IMAGE_REGEX),
					# 'pattern': re.compile(r'behindbody-(?P<d>[^\-]+)(?:-(?P<n>\D+)(?P<f>\d+)?)?.png'),
					'help': ('Images named like behindbody-DIRECTION[-SUFFIX] (e.g. behindbody-w.png, behindbody-s-cast1.png, behindbody-s-cast1-walk1.png, etc.) '
						"will be behind the character's torso, but still in front of the far arm (in east- and west-facing poses). This is useful for things like a braid slung over the character's far shoulder, which should be obscured by the character's torso, but should itself obscure the far arm. (See e.g. the 'princess', 'shoulderl' and 'shoulderr' hairstyles.)"
						),
					'mask_colors':['#808080','#C0C0C0'] },
	'main':       { 'pattern': re.compile(MULTI_FRAME_IMAGE_REGEX),
					# 'pattern': re.compile(r'(?P<d>(?!bg-)(?!behindbody-)[^\-]+)(?:-(?P<n>\D+)(?P<f>\d+)?)?.png'), 
					'help': 'Images named like DIRECTION[-SUFFIX], (e.g. e.png, s-shoot2.png, s-hurt2-hurt3.png, etc.) will be in the foreground.',
					'mask_colors':['#ffffff'] },
}

def match_inputs_to_outputs(inputs, outputs, 
	error_msg="Must give either one --output path or an equal number of --output paths to groups of --input images."):
	if isinstance(outputs, str) or outputs is None:
		outputs = [outputs]
	elif not isinstance(outputs, collections.abc.Iterable):
		raise ("outputs must be str, None, or list of same length as image_paths")

	if len(outputs) == 1:
		outputs = outputs * len(inputs)	
	elif len(outputs) != len(inputs):
		raise Exception(error_msg)

	return outputs

def make_frame_templates_per_layer(layout, layers, offsets_image=None, masks_image=None):
	offsets_image = Image.open(offsets_image) if offsets_image is not None else None
	masks_image = Image.open(masks_image) if masks_image is not None else None

	layer_templates = {}

	for layer_name, layer_args in layers.items():

		layer_templates[layer_name] = get_frame_templates_from_images(
			layout = layout,
			offsets_image = Image.open(layer_args['offsets_image']) if 'offsets_image' in layer_args else offsets_image,
			masks_image   = Image.open(layer_args['masks_image']) if 'masks_image' in layer_args else masks_image, 
			mask_colors   = layer_args['mask_colors'] if 'mask_colors' in layer_args else ['#ffffff']
		)
	return layer_templates


def distribute_repack(image_paths, from_layout, to_layout, offsets_image, masks_image, outputs=None, 
	layers=distribute_layers, verbose=False): 

	"""unpacks image from `from_layout`, then distributes it and re-packs to `to_layout`"""

	from_layout = load_layout(from_layout)
	to_layout   = load_layout(to_layout)

	# image_paths: list of dicts; each dict maps layer_name to image path
	image_groups = listify(image_paths)
	if not all(isinstance(ig, collections.abc.Mapping) for ig in image_groups):
		raise Exception("For distribute_repack, every image_group must be a dict of layer_name:image_path")

	if len(outputs) != len(image_paths):
		raise Exception("Must provide same number of --input and --output images")

	# construct a set of frame templates for each layer
	layer_templates = make_frame_templates_per_layer(to_layout, layers, offsets_image, masks_image)

	output_imgs = []
	for image_group_layers, group_output_path in zip(image_groups, outputs):
		if verbose: print(f"BEGIN GROUP '{image_group_layers}'")

		# image_group_layers is a dict of layer_name : image_path
		img_layers = []
		for layer_name, layer_args in layers.items():
			if verbose: print(f"LAYER '{layer_name}'")

			if layer_name in image_group_layers:
				# import pdb; pdb.set_trace()
				images = from_layout.unpack_images(Image.open(image_group_layers[layer_name]))

				# maybe there are no images for this layer; if so, save some loops
				if len(images) > 0: 
					
					images_distributed = distribute_images(images, 
						templates=layer_templates[layer_name], 
						positions=to_layout, 
						verbose=verbose)

					img_layers.append( to_layout.pack_images(images_distributed) )
					continue

			if verbose: print('- found no images')

		img = composite_images(img_layers)

		if group_output_path is not None:
			if verbose: print(f"END GROUP: --> {group_output_path}")
			mkdirpf(group_output_path)
			img.save(group_output_path)
		else:
			if verbose: print(f"END GROUP (no output)")

		output_imgs.append(img)



def distribute(image_paths, offsets_image, masks_image, layout, output=None, 
	layers=distribute_layers, verbose=False):

	layout = load_layout(layout)

	image_groups = []

	if verbose: print(f"image_paths: {image_paths}\n")

	# image_paths: (list of images) | (list of dirs) | (list of lists of images)
	# list of lists
	for image_group in image_paths:

		if not isinstance(image_group, list):
			image_group = [image_group]

		image_paths_are_dirs = [os.path.isdir(p) for p in image_group]

		# list of directories
		if all(image_paths_are_dirs):
			image_group = list(
				itertools.chain.from_iterable( glob(os.path.join(d,'*.png')) for d in image_group )
			)
		
		# mixture of images and directories (prohibited due to ambiguity)
		elif any(image_paths_are_dirs):
			raise NotImplementedError("image_paths must be either a list of lists, a list of directories, or a list of images; cannot mix directories and images.")
		
		# list of images
		image_groups.append(image_group)

	if verbose: print(f"image_groups: {image_groups}\n")

	# output: str | list of str | None
	output = match_inputs_to_outputs(image_groups, output)

	if verbose: print(f"output: {output}\n")	

	# construct a set of frame templates for each layer; each layer needs a different 
	# template since it may use a different mask image and/or color. offsets could 
	# technically be different too
	layer_templates = make_frame_templates_per_layer(layout, layers, offsets_image, masks_image)

	output_imgs = []
	for image_group, group_output in zip(image_groups, output):
		if verbose: 
			print(f"BEGIN GROUP '{image_group}'")

		img_layers = []
		for layer_name, layer_args in layers.items():
			if verbose: print(f"LAYER '{layer_name}'")

			images = load_images(image_group, layer_args['pattern'], verbose=verbose)

			# maybe there are no images for this layer; if so, save some loops
			if len(images) == 0: 
				if verbose: print('- found no images')
				continue

			images_distributed = distribute_images(images, 
				templates=layer_templates[layer_name], 
				positions=layout, 
				verbose=verbose)

			img_layers.append( layout.pack_images(images_distributed) )

		img = composite_images(img_layers)

		if group_output is not None:
			if verbose: print(f"END GROUP: --> {group_output}")
			mkdirpf(group_output)
			img.save(group_output)
		else:
			if verbose: print(f"END GROUP (no output)")

		output_imgs.append(img)


	# img = distribute_images_via_layout(images, layout, 
	# 	offsets_image=Image.open(offsets_image), 
	# 	masks_image=Image.open(masks_image))
	# return img
	return output_imgs

def main_distribute(args):
	distribute(args.input, args.offsets, args.masks, args.layout, args.output, 
		verbose=args.verbose)

def main_distribute_repack(args, default_layer = list(distribute_layers.keys())[-1]):
	image_groups = []
	for image_group in args.input:
		if len(image_group) == 1:
			image_groups.append( { default_layer : image_group[0] } )
		else:
			image_layers = {}
			for named_image_path in image_group:
				if named_image_path.count('=') == 1:
					layer_name, image_path = named_image_path.split('=')
					image_layers[layer_name] = image_path
				else:
					raise Exception("If more than one image is specified per --input group, images must be named by layer by writing LAYER_NAME=IMAGE_PATH.")
			image_groups.append(image_layers)


	distribute_repack(image_groups, args.from_layout, args.to_layout, 
		args.offsets, 
		args.masks, 
		outputs=args.output, 
		verbose=args.verbose)




def convert_layout(input, output, verbose=True):
	layout = load_layout(input)
	save_layout(layout, output)

def main_convert_layout(args):
	return convert_layout(args.input, args.output, verbose=args.verbose)




