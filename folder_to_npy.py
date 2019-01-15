#!/usr/bin/env python
import numpy as np
import dicom
import os
import matplotlib.image as mpimg
import sys
from skimage.transform import downscale_local_mean


def folder_to_npy(indir, outdir, outname="",save_single_files=False, downscale=-1):
	d = (downscale,downscale)

	# Remove trailing slashes
	if indir[-1] == "/":
		indir = indir[:-1]

	if outdir[-1] == "/":
		outdir = outdir[:-1]

	# Make directory
	if not os.path.exists(outdir):
	    os.makedirs(outdir)
	    print("Creating directory...")

	filenames = os.listdir(indir)
	ext = filenames[0].split()[-1].split(".")[-1]
	num_imgs = len(filenames)
	num_digits = int(np.log10(num_imgs)) + 1

	imgs = []

	print("Reading files...")
	for i in range(num_imgs):
		f = filenames[i]
		ext = filenames[i].split()[-1].split(".")[-1]
		str_id = str(i).zfill(num_digits)

		try:
			arr = "-1"
			if ext == 'dcm':
				ds = dicom.read_file(indir + "/" + f)
				arr = ds.pixel_array 
			else:
				arr = mpimg.imread(indir + "/" + f)

			if downscale != -1:
				arr = downscale_local_mean(arr, d)

			imgs.append(arr)


			if save_single_files:
				np.save(outdir + "/" + outname + str_id + ".npy", arr)

		except:
			continue



	print("Creating numpy matrix...")
	voxel = np.stack(imgs,2)

	if downscale != -1:
		voxel = downscale_local_mean(voxel, (1,1,downscale))

	if outname == "":
		outname = "voxel"
		
	npy_filename = outdir + "/" + outname + ".npy"
	print("Writing numpy file to", npy_filename, "...")
	np.save(npy_filename , voxel)


	###img_id = ''.join(s for s in f.split()[-1] if s.isdigit())
		 