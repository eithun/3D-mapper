from mapper_core import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import sys
import os

def verify_intervals(intervals):
	L = len(intervals)
	for i in range(2, L):
		if intervals[i][0] < intervals[i-2][1]:
			return False

	return True

def mapper(voxel, out_location, num_intervals=False, overlap=False, interval_scheme=False, given_intervals=False, verbose=False,cluster_threshold=100,save_labels=False):
	T0 = time.time()

	## Create binning intervals over domain ##
	if verbose:
		print("Binning data...")

	if given_intervals:
		intervals = given_intervals
	elif not num_intervals or not overlap or not interval_scheme:
		print("Error! Please provide num_intervals, overlap and interval_scheme OR given_intervals")
		return -1
	else:
		intervals = []
		if interval_scheme == "fixed":
			intervals = create_fixed_bins(voxel, num_intervals, overlap)
		if interval_scheme == "percentile":
			intervals = create_percentile_bins(voxel, num_intervals, overlap)

	if not verify_intervals(intervals):
		print("Error! Illegal interval scheme")
		return -1


	# Make directory
	if verbose:
		print("Creating directory...")
	if not os.path.exists(out_location):
		try:
			os.makedirs(out_location)
		except:
			print("Error! Failed to create output directory")
			return -1

	if out_location[-1] == "/":
		out_location = out_location[:-1]

	## Generate labeled regions ##
	if verbose:
		print("Labeling data...")

	t0 = time.time()
	labels = label_pixels(voxel, intervals)
	t1 = time.time()

	if save_labels:
		np.save(out_location + '/labels0.npy',labels[0])
		np.save(out_location + '/labels1.npy',labels[1])


	if verbose:
		print("time:", round(t1-t0,2))

	## Find clusters ##
	if verbose:
		print("Clustering data...")
	t0 = time.time()
	cluster_arrays, cluster_sizes, num_clusters, node_colors, node_locations = find_clusters(voxel, labels, cluster_threshold)
	t1 = time.time()
	if verbose:
		print("time:", round(t1-t0,2))
		print("Clusters found:", num_clusters)
	#print(node_locations)

	## Look at pairwise cluster arrays ##
	if verbose:
		print("Finding edges...")
	edge_set, overlap_array = get_edges(cluster_arrays)

	## Create graph ##
	if verbose:
		print("Creating graph...")
	G = nx.Graph()
	G.add_nodes_from(range(1,num_clusters+1))
	G.add_edges_from(edge_set)

	#node_colors = cluster_averages(voxel, overlays)
	adjusted_sizes = [300*c/max(cluster_sizes) for c in cluster_sizes]

	## Write out data ##
	if verbose:	
		print("Writing data...")
	nx.write_gml(G, out_location + "/graph.gml")
	np.savetxt(out_location + '/node_sizes.csv', adjusted_sizes , delimiter=',') 
	np.savetxt(out_location + '/node_colors.csv', node_colors , delimiter=',') 
	np.savetxt(out_location + '/vdims.csv', np.array([np.nanmin(voxel), np.nanmax(voxel)]) , delimiter=',') 
	np.savetxt(out_location + '/intervals.csv', np.array(intervals) , delimiter=',') 
	np.savetxt(out_location + '/node_locations.csv', np.array(node_locations) , delimiter=',') 

	T1 = time.time()

	if verbose:
		print("total time:", round(T1-T0,2))

	return 0
