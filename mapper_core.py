import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from  scipy import ndimage
import networkx as nx
from skimage import measure
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu

def create_fixed_bins(voxel, num_bins, overlap):
	flat_voxel = voxel.flatten()
	flat_voxel = flat_voxel[~np.isnan(flat_voxel)]
	interval_starts = []
	interval_ends = []

	vmin = np.min(flat_voxel)
	vmax = np.max(flat_voxel)

	# Create bins
	for i in range(num_bins):
		p = (100 * (i / (float(num_bins))))
		p_next = (100 * ((i+1) / (float(num_bins))))
		
		if (p - overlap >= 0):
			interval_starts.append(p - overlap)
		else:
			interval_starts.append(p)
		
		if (p_next + overlap <= 100):
			interval_ends.append(p_next + overlap)
		else:
			interval_ends.append(p_next)

	for i in range(num_bins):
		interval_starts[i] *= (vmax - vmin)/100
		interval_ends[i] *= (vmax - vmin)/100
	
	for i in range(num_bins):
		interval_starts[i] += vmin
		interval_ends[i] += vmin

	return [(interval_starts[i], interval_ends[i]) for i in range(num_bins)]

def create_percentile_bins(voxel, num_bins, overlap):
	flat_voxel = voxel.flatten()
	flat_voxel = flat_voxel[~np.isnan(flat_voxel)]
	interval_starts = []
	interval_ends = []

	# Create bins
	for i in range(num_bins):
		p = (100 * (i / (float(num_bins))))
		p_next = (100 * ((i+1) / (float(num_bins))))
		
		if (p - overlap >= 0):
			interval_starts.append(p - overlap)
		else:
			interval_starts.append(p)
		
		if (p_next + overlap <= 100):
			interval_ends.append(p_next + overlap)
		else:
			interval_ends.append(p_next)

	# Calculate percentiles
	percentiles = np.nanpercentile(flat_voxel, interval_starts + interval_ends, axis=0)

	percent_interval_starts = percentiles[:num_bins]
	percent_interval_ends = percentiles[num_bins:]
	
	return [(percent_interval_starts[i], percent_interval_ends[i]) for i in range(num_bins)]

# Label pixels
def label_pixels(in_voxel, intervals):
	voxel = np.copy(in_voxel)
	voxel[np.isnan(voxel)] = -1
    
	# Build two label arrays
	labels = [np.zeros(voxel.shape), np.zeros(voxel.shape)]
	
	# Alternate parity as arrays are constructed
	for i in range(len(intervals)):
		if i == 0:
			labels[(i%2)][np.where((voxel>=intervals[i][0]) & (voxel<intervals[i][1]))] = i + 1
		elif i == len(intervals) - 1:
			labels[(i%2)][np.where((voxel>intervals[i][0]) & (voxel<=intervals[i][1]))] = i + 1
		else:
			labels[(i%2)][np.where((voxel>intervals[i][0]) & (voxel<intervals[i][1]))] = i + 1
	
	return labels

# Equivalence relation for network to remove "valence 2" nodes
def valence_equiv(u, v):
	for e in list(set(G.neighbors(u)) & set(G.neighbors(v))):
		if G.degree(e) == 2:
			return True

	return False


def find_clusters(voxel,labels,small_size):
	cluster_arrays = []
	cluster_sizes = []
	running_total = 0
	node_colors = []
	node_locations = []

	# Loop through labeled arrays
	for i in range(len(labels)):
		l = labels[i]
		
		# Label
		labeled_array,num_features = measure.label(l,return_num=True) 

		# Remove small objects
		if small_size != -1:
			ls = remove_small_objects(labeled_array,min_size=small_size)
			labeled_array,num_features = measure.label(ls,return_num=True) 
		
		# Information about clusters for graph drawing later
		props = measure.regionprops(label_image=labeled_array,intensity_image=voxel)

		# Save useful information
		for i in range(num_features):#range(1,num_features+1):
			current_props = props[i]
			node_colors.append(current_props.mean_intensity)
			node_locations.append(current_props.centroid)
			cluster_sizes.append(current_props.area)
		
		# Relabel
		labeled_array[labeled_array != 0] += running_total
		cluster_arrays.append(labeled_array)
		running_total += num_features

	return cluster_arrays, cluster_sizes, running_total, node_colors, node_locations

def find_clusters_old(voxel,labels,small_size=100):
	cluster_arrays = []
	cluster_sizes = []
	running_total = 0
	node_colors = []
	node_locations = []

	# Loop through labeled arrays
	for i in range(len(labels)):
		l = labels[i]
		# Label and remove small objects (noise)
		labeled_array,_ = measure.label(l,return_num=True) 
		ls = remove_small_objects(labeled_array,min_size=small_size)

		# Relabel
		labeled_array,num_features = measure.label(ls,return_num=True) 

		labeled_array[labeled_array != 0] += running_total
		cluster_arrays.append(labeled_array)

		for i in range(running_total+1,running_total+num_features+1):
			overlay = (labeled_array == i)
			avg = np.average(voxel[overlay])
			node_colors.append(avg)
			node_locations.append(np.mean(np.argwhere(overlay),axis=0))
			cluster_sizes.append(np.count_nonzero(overlay))

		running_total += num_features

	return cluster_arrays, cluster_sizes, running_total, node_colors, node_locations

def get_edges(cluster_arrays):
	edge_set = set()
	A = cluster_arrays[0]
	B = cluster_arrays[1]

	# Find overapping regions
	overlap = np.logical_and(A, B)
	overlap_array = np.array(overlap,dtype='int')
	labeled_array,num_features = measure.label(overlap,return_num=True)

	props = measure.regionprops(label_image=labeled_array)
	for i in range(num_features):
		coord = tuple(props[i].coords[0,:])

		edge_set.add((A[coord], B[coord]))

	return edge_set, overlap_array

def get_edges_old(cluster_arrays):
	edge_set = set()
	A = cluster_arrays[0]
	B = cluster_arrays[1]

	# Find overapping regions
	overlap = np.logical_and(A, B)
	overlap_array = np.array(overlap,dtype='int')
	labeled_array,num_features = measure.label(overlap,return_num=True)

	for i in range(1,num_features+1):
		# Get names
		a = A[labeled_array == i][0]
		b = B[labeled_array == i][0]
		#print((a,b))	
		edge_set.add((a,b))

	return edge_set, overlap_array

def cluster_averages(voxel, overlays):
	node_colors = []
	for i in range(len(overlays)):
	    avg = np.average(voxel[overlays[i]])
	    node_colors.append(avg)

	return node_colors