import os.path
import filecmp

filecmp.DEFAULT_IGNORES = list(set(filecmp.DEFAULT_IGNORES + ['.DS_Store']))

from glob import glob
def assert_dirs_are_same(d1, d2, **kwargs):
	_cmp = filecmp.dircmp(d1, d2, **kwargs)
	assert set(_cmp.diff_files) == set(), f"Differing files: {_cmp.diff_files}"
	assert set(_cmp.left_only) == set(), f"Files only in {d1}: {_cmp.left_only})"
	assert set(_cmp.right_only) == set(), f"Files only in {d2}: {_cmp.right_only})"



from PIL import Image, ImageChops, ImageStat


def assert_images_equal(im1, im2):
	diff = imagediff(im1, im2, diff_img_file=True)
	assert diff == 0

def imagediff(
    im1_file, im2_file, diff_img_file=None, ignore_alpha=False
):
    """
    Calculate the difference between two images by comparing channel values at the pixel
    level. If the images are different sizes, the second will be resized to match the
    first.
    `delete_diff_file`: removes the diff image after ratio found
    `diff_img_file`: filename to store diff image
    `ignore_alpha`: ignore the alpha channel for ratio calculation, and set the diff
        image's alpha to fully opaque

	Copyright 2017 Nicolas Hahn
	MIT License
	https://github.com/nicolashahn/diffimg/blob/master/diffimg/diff.py
    """
    if diff_img_file is None:
    	diff_img_file = False
    if diff_img_file == True:
    	diff_img_file = f"{im1_file}-{os.path.basename(im2_file)}"
    

    im1 = Image.open(im1_file)
    im2 = Image.open(im2_file)

    # Ensure we have the same color channels (RGBA vs RGB)
    if im1.mode != im2.mode:
        raise ValueError(
            (
                "Differing color modes:\n  {}: {}\n  {}: {}\n"
                "Ensure image color modes are the same."
            ).format(im1_file, im1.mode, im2_file, im2.mode)
        )

    # Coerce 2nd dimensions to same as 1st
    im2 = im2.resize((im1.width, im1.height))

    # Generate diff image in memory.
    diff_img = ImageChops.difference(im1, im2)

    if ignore_alpha:
        diff_img.putalpha(255)

    if diff_img_file:
        diff_img.save(diff_img_file)
        print(diff_img_file)

    # Calculate difference as a ratio.
    stat = ImageStat.Stat(diff_img)
    # stat.mean can be [r,g,b] or [r,g,b,a].
    removed_channels = 1 if ignore_alpha and len(stat.mean) == 4 else 0
    num_channels = len(stat.mean) - removed_channels
    sum_channel_values = sum(stat.mean[:num_channels])
    max_all_channels = num_channels * 255
    diff_ratio = sum_channel_values / max_all_channels

    return diff_ratio