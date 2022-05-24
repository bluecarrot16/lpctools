import shlex
import subprocess
from testutils import *

class TestAnimationFrameID():
	def test_afi(self):
		from lpctools.arrange import AnimationFrameID

		afi = AnimationFrameID('cast','n','4')
		assert afi == ('cast','n',4)

class TestLayout():
	def test_repack(self, tmpdir):
		from lpctools.arrange import repack_animations
		
		repack_animations('tests/arrange_files/packed-evert.png', 
			from_layouts='evert', to_layouts=['universal'], output_dir=tmpdir)

		assert filecmp.cmp(tmpdir / 'universal.png', 'tests/arrange_files/packed-universal.png')


		outfile = str(tmpdir / "male-mirrored.png")
		repack_animations('tests/arrange_files/male-w-only.png', 
			from_layouts='universal', to_layouts=['universal'], output_pattern=outfile, mirror=('w','e'))
		assert filecmp.cmp(outfile,'tests/arrange_files/male-mirrored.png')

	def test_unpack(self, tmpdir):
		tmpdir = str(tmpdir)

		import lpctools.arrange

		lpctools.arrange.unpack_animations('tests/arrange_files/packed-universal.png',
			layout='universal',
			output_dir=tmpdir)

		assert_dirs_are_same(tmpdir, 'tests/arrange_files/unpacked')


		lpctools.arrange.unpack_animations('tests/arrange_files/walk_push.png',
			layout='push',
			output_dir=tmpdir)

		lpctools.arrange.unpack_animations('tests/arrange_files/grab.png',
			layout='grab',
			output_dir=tmpdir)

		assert_dirs_are_same(tmpdir, 'tests/arrange_files/unpacked_extended')

	def test_pack(self, tmpdir):

		import lpctools.arrange
		
		outfile = str(tmpdir / 'packed-universal.png')
		lpctools.arrange.pack_animations(
			glob('tests/arrange_files/unpacked/*.png'), 
			layout='universal', 
			output=outfile
			)

		assert filecmp.cmp(outfile, 'tests/arrange_files/packed-universal.png')


		outfile = str(tmpdir / 'packed-evert.png')
		lpctools.arrange.pack_animations(
			glob('tests/arrange_files/unpacked_extended/*.png'), 
			layout='evert', 
			output=outfile
			)

		assert filecmp.cmp(outfile, 'tests/arrange_files/packed-evert.png')

	def test_layout(self):
		import lpctools.arrange as arr
		from lpctools.arrange import AnimationFrameID, Animation
		import numpy as np

		assert arr.layouts['universal'].size == (13, 21)
		assert arr.layouts['universal'].pixel_size == (13*64, 21*64)

		universal_animations = [
			Animation(name='cast', direction='n', nframes=7), 
			Animation(name='cast', direction='w', nframes=7), 
			Animation(name='cast', direction='s', nframes=7), 
			Animation(name='cast', direction='e', nframes=7), 
			Animation(name='thrust', direction='n', nframes=8), 
			Animation(name='thrust', direction='w', nframes=8), 
			Animation(name='thrust', direction='s', nframes=8), 
			Animation(name='thrust', direction='e', nframes=8), 
			Animation(name='walk', direction='n', nframes=9), 
			Animation(name='walk', direction='w', nframes=9), 
			Animation(name='walk', direction='s', nframes=9), 
			Animation(name='walk', direction='e', nframes=9), 
			Animation(name='slash', direction='n', nframes=6), 
			Animation(name='slash', direction='w', nframes=6), 
			Animation(name='slash', direction='s', nframes=6), 
			Animation(name='slash', direction='e', nframes=6), 
			Animation(name='shoot', direction='n', nframes=13), 
			Animation(name='shoot', direction='w', nframes=13), 
			Animation(name='shoot', direction='s', nframes=13), 
			Animation(name='shoot', direction='e', nframes=13), 
			Animation(name='hurt', direction='s', nframes=6)
		]
		assert arr.layouts['universal'].get_animations() == universal_animations

		universal = np.array([
			[AnimationFrameID(name='cast', direction='n', frame=0), AnimationFrameID(name='cast', direction='n', frame=1), AnimationFrameID(name='cast', direction='n', frame=2), AnimationFrameID(name='cast', direction='n', frame=3), AnimationFrameID(name='cast', direction='n', frame=4), AnimationFrameID(name='cast', direction='n', frame=5), AnimationFrameID(name='cast', direction='n', frame=6), None, None, None, None, None, None], 
			[AnimationFrameID(name='cast', direction='w', frame=0), AnimationFrameID(name='cast', direction='w', frame=1), AnimationFrameID(name='cast', direction='w', frame=2), AnimationFrameID(name='cast', direction='w', frame=3), AnimationFrameID(name='cast', direction='w', frame=4), AnimationFrameID(name='cast', direction='w', frame=5), AnimationFrameID(name='cast', direction='w', frame=6), None, None, None, None, None, None], 
			[AnimationFrameID(name='cast', direction='s', frame=0), AnimationFrameID(name='cast', direction='s', frame=1), AnimationFrameID(name='cast', direction='s', frame=2), AnimationFrameID(name='cast', direction='s', frame=3), AnimationFrameID(name='cast', direction='s', frame=4), AnimationFrameID(name='cast', direction='s', frame=5), AnimationFrameID(name='cast', direction='s', frame=6), None, None, None, None, None, None], 
			[AnimationFrameID(name='cast', direction='e', frame=0), AnimationFrameID(name='cast', direction='e', frame=1), AnimationFrameID(name='cast', direction='e', frame=2), AnimationFrameID(name='cast', direction='e', frame=3), AnimationFrameID(name='cast', direction='e', frame=4), AnimationFrameID(name='cast', direction='e', frame=5), AnimationFrameID(name='cast', direction='e', frame=6), None, None, None, None, None, None], 
			[AnimationFrameID(name='thrust', direction='n', frame=0), AnimationFrameID(name='thrust', direction='n', frame=1), AnimationFrameID(name='thrust', direction='n', frame=2), AnimationFrameID(name='thrust', direction='n', frame=3), AnimationFrameID(name='thrust', direction='n', frame=4), AnimationFrameID(name='thrust', direction='n', frame=5), AnimationFrameID(name='thrust', direction='n', frame=6), AnimationFrameID(name='thrust', direction='n', frame=7), None, None, None, None, None], 
			[AnimationFrameID(name='thrust', direction='w', frame=0), AnimationFrameID(name='thrust', direction='w', frame=1), AnimationFrameID(name='thrust', direction='w', frame=2), AnimationFrameID(name='thrust', direction='w', frame=3), AnimationFrameID(name='thrust', direction='w', frame=4), AnimationFrameID(name='thrust', direction='w', frame=5), AnimationFrameID(name='thrust', direction='w', frame=6), AnimationFrameID(name='thrust', direction='w', frame=7), None, None, None, None, None], 
			[AnimationFrameID(name='thrust', direction='s', frame=0), AnimationFrameID(name='thrust', direction='s', frame=1), AnimationFrameID(name='thrust', direction='s', frame=2), AnimationFrameID(name='thrust', direction='s', frame=3), AnimationFrameID(name='thrust', direction='s', frame=4), AnimationFrameID(name='thrust', direction='s', frame=5), AnimationFrameID(name='thrust', direction='s', frame=6), AnimationFrameID(name='thrust', direction='s', frame=7), None, None, None, None, None], 
			[AnimationFrameID(name='thrust', direction='e', frame=0), AnimationFrameID(name='thrust', direction='e', frame=1), AnimationFrameID(name='thrust', direction='e', frame=2), AnimationFrameID(name='thrust', direction='e', frame=3), AnimationFrameID(name='thrust', direction='e', frame=4), AnimationFrameID(name='thrust', direction='e', frame=5), AnimationFrameID(name='thrust', direction='e', frame=6), AnimationFrameID(name='thrust', direction='e', frame=7), None, None, None, None, None], 
			[AnimationFrameID(name='walk', direction='n', frame=0), AnimationFrameID(name='walk', direction='n', frame=1), AnimationFrameID(name='walk', direction='n', frame=2), AnimationFrameID(name='walk', direction='n', frame=3), AnimationFrameID(name='walk', direction='n', frame=4), AnimationFrameID(name='walk', direction='n', frame=5), AnimationFrameID(name='walk', direction='n', frame=6), AnimationFrameID(name='walk', direction='n', frame=7), AnimationFrameID(name='walk', direction='n', frame=8), None, None, None, None], 
			[AnimationFrameID(name='walk', direction='w', frame=0), AnimationFrameID(name='walk', direction='w', frame=1), AnimationFrameID(name='walk', direction='w', frame=2), AnimationFrameID(name='walk', direction='w', frame=3), AnimationFrameID(name='walk', direction='w', frame=4), AnimationFrameID(name='walk', direction='w', frame=5), AnimationFrameID(name='walk', direction='w', frame=6), AnimationFrameID(name='walk', direction='w', frame=7), AnimationFrameID(name='walk', direction='w', frame=8), None, None, None, None], 
			[AnimationFrameID(name='walk', direction='s', frame=0), AnimationFrameID(name='walk', direction='s', frame=1), AnimationFrameID(name='walk', direction='s', frame=2), AnimationFrameID(name='walk', direction='s', frame=3), AnimationFrameID(name='walk', direction='s', frame=4), AnimationFrameID(name='walk', direction='s', frame=5), AnimationFrameID(name='walk', direction='s', frame=6), AnimationFrameID(name='walk', direction='s', frame=7), AnimationFrameID(name='walk', direction='s', frame=8), None, None, None, None], 
			[AnimationFrameID(name='walk', direction='e', frame=0), AnimationFrameID(name='walk', direction='e', frame=1), AnimationFrameID(name='walk', direction='e', frame=2), AnimationFrameID(name='walk', direction='e', frame=3), AnimationFrameID(name='walk', direction='e', frame=4), AnimationFrameID(name='walk', direction='e', frame=5), AnimationFrameID(name='walk', direction='e', frame=6), AnimationFrameID(name='walk', direction='e', frame=7), AnimationFrameID(name='walk', direction='e', frame=8), None, None, None, None], 
			[AnimationFrameID(name='slash', direction='n', frame=0), AnimationFrameID(name='slash', direction='n', frame=1), AnimationFrameID(name='slash', direction='n', frame=2), AnimationFrameID(name='slash', direction='n', frame=3), AnimationFrameID(name='slash', direction='n', frame=4), AnimationFrameID(name='slash', direction='n', frame=5), None, None, None, None, None, None, None], 
			[AnimationFrameID(name='slash', direction='w', frame=0), AnimationFrameID(name='slash', direction='w', frame=1), AnimationFrameID(name='slash', direction='w', frame=2), AnimationFrameID(name='slash', direction='w', frame=3), AnimationFrameID(name='slash', direction='w', frame=4), AnimationFrameID(name='slash', direction='w', frame=5), None, None, None, None, None, None, None], 
			[AnimationFrameID(name='slash', direction='s', frame=0), AnimationFrameID(name='slash', direction='s', frame=1), AnimationFrameID(name='slash', direction='s', frame=2), AnimationFrameID(name='slash', direction='s', frame=3), AnimationFrameID(name='slash', direction='s', frame=4), AnimationFrameID(name='slash', direction='s', frame=5), None, None, None, None, None, None, None], 
			[AnimationFrameID(name='slash', direction='e', frame=0), AnimationFrameID(name='slash', direction='e', frame=1), AnimationFrameID(name='slash', direction='e', frame=2), AnimationFrameID(name='slash', direction='e', frame=3), AnimationFrameID(name='slash', direction='e', frame=4), AnimationFrameID(name='slash', direction='e', frame=5), None, None, None, None, None, None, None], 
			[AnimationFrameID(name='shoot', direction='n', frame=0), AnimationFrameID(name='shoot', direction='n', frame=1), AnimationFrameID(name='shoot', direction='n', frame=2), AnimationFrameID(name='shoot', direction='n', frame=3), AnimationFrameID(name='shoot', direction='n', frame=4), AnimationFrameID(name='shoot', direction='n', frame=5), AnimationFrameID(name='shoot', direction='n', frame=6), AnimationFrameID(name='shoot', direction='n', frame=7), AnimationFrameID(name='shoot', direction='n', frame=8), AnimationFrameID(name='shoot', direction='n', frame=9), AnimationFrameID(name='shoot', direction='n', frame=10), AnimationFrameID(name='shoot', direction='n', frame=11), AnimationFrameID(name='shoot', direction='n', frame=12)], 
			[AnimationFrameID(name='shoot', direction='w', frame=0), AnimationFrameID(name='shoot', direction='w', frame=1), AnimationFrameID(name='shoot', direction='w', frame=2), AnimationFrameID(name='shoot', direction='w', frame=3), AnimationFrameID(name='shoot', direction='w', frame=4), AnimationFrameID(name='shoot', direction='w', frame=5), AnimationFrameID(name='shoot', direction='w', frame=6), AnimationFrameID(name='shoot', direction='w', frame=7), AnimationFrameID(name='shoot', direction='w', frame=8), AnimationFrameID(name='shoot', direction='w', frame=9), AnimationFrameID(name='shoot', direction='w', frame=10), AnimationFrameID(name='shoot', direction='w', frame=11), AnimationFrameID(name='shoot', direction='w', frame=12)], 
			[AnimationFrameID(name='shoot', direction='s', frame=0), AnimationFrameID(name='shoot', direction='s', frame=1), AnimationFrameID(name='shoot', direction='s', frame=2), AnimationFrameID(name='shoot', direction='s', frame=3), AnimationFrameID(name='shoot', direction='s', frame=4), AnimationFrameID(name='shoot', direction='s', frame=5), AnimationFrameID(name='shoot', direction='s', frame=6), AnimationFrameID(name='shoot', direction='s', frame=7), AnimationFrameID(name='shoot', direction='s', frame=8), AnimationFrameID(name='shoot', direction='s', frame=9), AnimationFrameID(name='shoot', direction='s', frame=10), AnimationFrameID(name='shoot', direction='s', frame=11), AnimationFrameID(name='shoot', direction='s', frame=12)], 
			[AnimationFrameID(name='shoot', direction='e', frame=0), AnimationFrameID(name='shoot', direction='e', frame=1), AnimationFrameID(name='shoot', direction='e', frame=2), AnimationFrameID(name='shoot', direction='e', frame=3), AnimationFrameID(name='shoot', direction='e', frame=4), AnimationFrameID(name='shoot', direction='e', frame=5), AnimationFrameID(name='shoot', direction='e', frame=6), AnimationFrameID(name='shoot', direction='e', frame=7), AnimationFrameID(name='shoot', direction='e', frame=8), AnimationFrameID(name='shoot', direction='e', frame=9), AnimationFrameID(name='shoot', direction='e', frame=10), AnimationFrameID(name='shoot', direction='e', frame=11), AnimationFrameID(name='shoot', direction='e', frame=12)], 
			[AnimationFrameID(name='hurt', direction='s', frame=0), AnimationFrameID(name='hurt', direction='s', frame=1), AnimationFrameID(name='hurt', direction='s', frame=2), AnimationFrameID(name='hurt', direction='s', frame=3), AnimationFrameID(name='hurt', direction='s', frame=4), AnimationFrameID(name='hurt', direction='s', frame=5), None, None, None, None, None, None, None]], 
			dtype='object') 

		assert (arr.layouts['universal'].to_array().T == universal).all()

	def test_convert_layout(self, tmpdir):
		import lpctools.arrange

		layout_from_json = lpctools.arrange.load_layout('tests/arrange_files/layout/universal.json')
		layout_builtin = lpctools.arrange.load_layout('universal')
		assert layout_from_json == layout_builtin

		layout_builtin.to_json(tmpdir / 'universal.json')
		assert filecmp.cmp(str(tmpdir / 'universal.json'), 'tests/arrange_files/layout/universal.json')

		layout_builtin.to_image(str(tmpdir / 'universal.png'))
		assert filecmp.cmp(str(tmpdir / 'universal.png'), 'tests/arrange_files/layout/universal.png')


class TestDistribute():
	def test_image_regexs(self):
		from lpctools.arrange import MULTI_FRAME_IMAGE_REGEX, distribute_layers

		assert distribute_layers['main']['pattern'].match('s-cast3-cast6.png').groupdict() == {'d':'s', 'frames':'cast3-cast6'}
		assert distribute_layers['main']['pattern'].match('s-thrust3-thrust4-thrust5-thrust6.png').groupdict() == {'d':'s', 'frames':'thrust3-thrust4-thrust5-thrust6'}
		assert distribute_layers['main']['pattern'].match('n.png').groupdict() == {'d':'n', 'frames':None}

		assert distribute_layers['main']['pattern'].match('bg-n.png') is None
		assert distribute_layers['main']['pattern'].match('behindbody-w.png') is None
		assert distribute_layers['main']['pattern'].match('behindbody-s-hurt5.png') is None


		assert distribute_layers['bg']['pattern'].match('bg-n.png').groupdict() == {'d':'n', 'frames':None}
		assert distribute_layers['behindbody']['pattern'].match('behindbody-s-hurt5.png').groupdict() == {'d':'s', 'frames':'hurt5'}


	def test_load_images(self):
		import re
		from lpctools.arrange import load_images, MULTI_FRAME_IMAGE_REGEX, AnimationFrameID

		images = load_images(glob('tests/arrange_files/shield/spartan/*.png'),
			pattern=re.compile(MULTI_FRAME_IMAGE_REGEX)
			)

		image_paths = {afi:img.filename for afi, img in images.items()}

		expected_image_paths = {
		 AnimationFrameID(name='cast', direction='s', frame=3): 'tests/arrange_files/shield/spartan/s-cast3-cast6.png',
		 AnimationFrameID(name='cast', direction='s', frame=4): 'tests/arrange_files/shield/spartan/s-cast4-cast5.png',
		 AnimationFrameID(name='cast', direction='s', frame=5): 'tests/arrange_files/shield/spartan/s-cast4-cast5.png',
		 AnimationFrameID(name='cast', direction='s', frame=6): 'tests/arrange_files/shield/spartan/s-cast3-cast6.png',
		 AnimationFrameID(name='hurt', direction='s', frame=2): 'tests/arrange_files/shield/spartan/s-hurt2.png',
		 AnimationFrameID(name='hurt', direction='s', frame=3): 'tests/arrange_files/shield/spartan/s-hurt3.png',
		 AnimationFrameID(name='hurt', direction='s', frame=4): 'tests/arrange_files/shield/spartan/s-hurt4.png',
		 AnimationFrameID(name='hurt', direction='s', frame=5): 'tests/arrange_files/shield/spartan/s-hurt5.png',
		 AnimationFrameID(name='shoot', direction='e', frame=None): 'tests/arrange_files/shield/spartan/e-shoot.png',
		 AnimationFrameID(name='shoot', direction='n', frame=None): 'tests/arrange_files/shield/spartan/n-shoot.png',
		 AnimationFrameID(name='shoot', direction='s', frame=None): 'tests/arrange_files/shield/spartan/s-shoot.png',
		 AnimationFrameID(name='shoot', direction='w', frame=None): 'tests/arrange_files/shield/spartan/w-shoot.png',
		 AnimationFrameID(name='thrust', direction='s', frame=2): 'tests/arrange_files/shield/spartan/s-thrust2-thrust7.png',
		 AnimationFrameID(name='thrust', direction='s', frame=4): 'tests/arrange_files/shield/spartan/s-thrust3-thrust4-thrust5-thrust6.png',
		 AnimationFrameID(name='thrust', direction='s', frame=6): 'tests/arrange_files/shield/spartan/s-thrust3-thrust4-thrust5-thrust6.png',
		 AnimationFrameID(name='thrust', direction='s', frame=7): 'tests/arrange_files/shield/spartan/s-thrust2-thrust7.png',
		 AnimationFrameID(name=None, direction='e', frame=None): 'tests/arrange_files/shield/spartan/e.png',
		 AnimationFrameID(name=None, direction='n', frame=None): 'tests/arrange_files/shield/spartan/n.png',
		 AnimationFrameID(name=None, direction='w', frame=None): 'tests/arrange_files/shield/spartan/w.png',
		 AnimationFrameID(name='thrust', direction='s', frame=3): 'tests/arrange_files/shield/spartan/s-thrust3-thrust4-thrust5-thrust6.png',
		 AnimationFrameID(name='thrust', direction='s', frame=5): 'tests/arrange_files/shield/spartan/s-thrust3-thrust4-thrust5-thrust6.png',
		 AnimationFrameID(name=None, direction='s', frame=None): 'tests/arrange_files/shield/spartan/s.png'
		}

		assert image_paths == expected_image_paths

	def test_distribute_hair(self, tmpdir):
		import lpctools.arrange

		# test with list of files as input
		outfile = str(tmpdir / 'hair_plain.png')
		out = lpctools.arrange.distribute(
			image_paths = [glob('tests/arrange_files/hair/hair_plain/*.png')],
			offsets_image = 'tests/arrange_files/hair/reference_points_male.png', 
			masks_image = 'tests/arrange_files/hair/masks_male.png',  
			layout = 'universal', 
			output = outfile)

		# assert filecmp.cmp(outfile, 'tests/arrange_files/hair/hair_plain.png')
		assert_images_equal(outfile, 'tests/arrange_files/hair/hair_plain.png')

		# test with multiple directories as input
		# outfiles = ['tests/arrange_files/hair/hair_page2.png', 'tests/arrange_files/hair/hair_shortknot.png']
		outfiles = [str(tmpdir / 'hair_page2.png'), str(tmpdir / 'hair_shortknot.png')]
		out = lpctools.arrange.distribute(
			image_paths = ['tests/arrange_files/hair/hair_page2', 'tests/arrange_files/hair/hair_shortknot'],
			offsets_image = 'tests/arrange_files/hair/reference_points_male.png', 
			masks_image = 'tests/arrange_files/hair/masks_male.png',  
			layout = 'universal', 
			output = outfiles)

		assert filecmp.cmp(outfiles[0], 'tests/arrange_files/hair/hair_page2.png')
		assert filecmp.cmp(outfiles[1], 'tests/arrange_files/hair/hair_shortknot.png')

		# test with multiple layers
		outfile = str(tmpdir / 'hair_shoulderr.png')
		# outfile = 'tests/arrange_files/hair/hair_shoulderr.png' #str(tmpdir / 'hair_shoulderr.png')
		out = lpctools.arrange.distribute(
			image_paths = [glob('tests/arrange_files/hair/hair_shoulderr/*.png')],
			offsets_image = 'tests/arrange_files/hair/reference_points_male.png', 
			masks_image = 'tests/arrange_files/hair/masks_male.png',  
			layout = 'universal', 
			output = outfile)

		assert filecmp.cmp(outfile, 'tests/arrange_files/hair/hair_shoulderr.png')

	def test_distribute_shield(self, tmpdir):
		import lpctools.arrange

		# test with more complicated set of images with multiple layers
		outfile = str(tmpdir / 'spartan.png')
		# outfile = 'tests/arrange_files/shield/spartan.png' #str(tmpdir / 'spartan.png')
		out = lpctools.arrange.distribute(
			image_paths = [glob('tests/arrange_files/shield/spartan/*.png')],
			offsets_image = 'tests/arrange_files/shield/reference_points_male.png', 
			masks_image = 'tests/arrange_files/shield/masks_male.png',  
			layout = 'universal', 
			output = outfile
			,verbose=True
			)

		assert filecmp.cmp(outfile,'tests/arrange_files/shield/spartan.png')

	def test_distribute_cli(self, tmpdir):
		import lpctools

		outfile = f"{tmpdir}/crusader.png"
		lpctools.main(
			shlex.split(
			f"-v arrange distribute --input tests/arrange_files/shield/crusader/ --output {outfile} --offsets tests/arrange_files/shield/reference_points_male.png --mask tests/arrange_files/shield/masks_male.png"
			)
		)

		assert filecmp.cmp(outfile,'tests/arrange_files/shield/crusader.png')