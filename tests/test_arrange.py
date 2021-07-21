import filecmp
from glob import glob
def assert_dirs_are_same(d1, d2, **kwargs):
	_cmp = filecmp.dircmp(d1, d2, **kwargs)
	assert set(_cmp.diff_files) == set()
	assert set(_cmp.left_only) == set(_cmp.right_only) == set()


class TestAnimationFrameID():
	def test_afi(self):
		from lpctools.arrange import AnimationFrameID

		afi = AnimationFrameID('cast','n','4')
		assert afi == ('cast','n',4)

class TestLayout():
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


class TestDistribute():
	def test_distribute_hair(self, tmpdir):
		import lpctools.arrange

		# test with list of files as input
		outfile = str(tmpdir / 'hair_plain.png')
		out = lpctools.arrange.distribute(
			image_paths = glob('tests/arrange_files/hair/hair_plain/*.png'),
			offsets_image = 'tests/arrange_files/hair/reference_points_male.png', 
			masks_image = 'tests/arrange_files/hair/masks_male.png',  
			layout = 'universal', 
			output = outfile)

		assert filecmp.cmp(outfile, 'tests/arrange_files/hair/hair_plain.png')

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
			image_paths = glob('tests/arrange_files/hair/hair_shoulderr/*.png'),
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
			image_paths = glob('tests/arrange_files/shield/spartan/*.png'),
			offsets_image = 'tests/arrange_files/shield/reference_points_male.png', 
			masks_image = 'tests/arrange_files/shield/masks_male.png',  
			layout = 'universal', 
			output = outfile
			# ,verbose=True
			)

		assert filecmp.cmp(outfile,'tests/arrange_files/shield/spartan.png')