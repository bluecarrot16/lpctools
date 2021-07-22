import filecmp

filecmp.DEFAULT_IGNORES = list(set(filecmp.DEFAULT_IGNORES + ['.DS_Store']))

from glob import glob
def assert_dirs_are_same(d1, d2, **kwargs):
	_cmp = filecmp.dircmp(d1, d2, **kwargs)
	assert set(_cmp.diff_files) == set(), f"Differing files: {_cmp.diff_files}"
	assert set(_cmp.left_only) == set(), f"Files only in {d1}: {_cmp.left_only})"
	assert set(_cmp.right_only) == set(), f"Files only in {d2}: {_cmp.right_only})"