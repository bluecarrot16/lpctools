import os
import subprocess
import shlex
import filecmp

class TestColor():
	def test_color(self):
		from lpctools.recolor import Color

		c1 = Color('#ff0000')
		assert c1 == (255, 0, 0, 255)
		
		c1 = Color('#ff0000aa')
		assert c1 == (255, 0, 0, 0xAA)

		c2 = Color(255,0,0)
		assert c2 == (255, 0, 0, 255)

	def test_to_hex(self):

		from lpctools.recolor import Color

		c1 = Color('#ff0000')
		assert c1.to_hex() == '#ff0000ff'

		c1 = Color('#aabbccee')
		assert c1.to_hex() == '#aabbccee'


class TestPalette():
	def test_load(self):
		from lpctools.recolor import load_palette
		print(os.getcwd())
		pal1 = load_palette('tests/recolor_files/ivory.png')

		pal2 = load_palette('tests/recolor_files/ivory.gpl')


class TestRecolorCLI():
	def test_recolor(self, tmpdir):
		print(tmpdir)
		import lpctools

		lpctools.main(
			shlex.split(f"-v recolor --input tests/recolor_files/hair.png --output '{tmpdir}/%b/%p.%e' --mapping tests/recolor_files/map.png --palettes blonde blue")
		)	

		assert set(os.listdir(tmpdir)) == {'hair'}
		assert set(os.listdir(tmpdir + '/hair')) ==  {'blue.png', 'blonde.png'}

		assert filecmp.cmp(f"{tmpdir}/hair/blonde.png", 'tests/recolor_files/expected_output/hair/blonde.png')
		assert filecmp.cmp(f"{tmpdir}/hair/blue.png", 'tests/recolor_files/expected_output/hair/blue.png')


	def test_recolor2(self, tmpdir):
		print(tmpdir)
		import lpctools

		lpctools.main(
			shlex.split(f"-v recolor --input tests/recolor_files/hair.png tests/recolor_files/hair2.png --output '{tmpdir}/%b/%p.%e' --mapping tests/recolor_files/map.png --palettes blonde blue")
		)	

		assert set(os.listdir(tmpdir)) == {'hair', 'hair2'}
		assert set(os.listdir(f"{tmpdir}/hair")) ==  {'blue.png', 'blonde.png'}
		assert set(os.listdir(f"{tmpdir}/hair2")) ==  {'blue.png', 'blonde.png'}

		assert filecmp.cmp(f"{tmpdir}/hair/blonde.png", 'tests/recolor_files/expected_output/hair/blonde.png')
		assert filecmp.cmp(f"{tmpdir}/hair/blue.png", 'tests/recolor_files/expected_output/hair/blue.png')
		assert filecmp.cmp(f"{tmpdir}/hair2/blonde.png", 'tests/recolor_files/expected_output/hair2/blonde.png')
		assert filecmp.cmp(f"{tmpdir}/hair2/blue.png", 'tests/recolor_files/expected_output/hair2/blue.png')
