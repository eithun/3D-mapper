import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


def plot_interval(ax, y, xstart, xstop):
    color='black'
    epsilon = 0.05*3
    ax.hlines(y, xstart, xstop, color, lw=2)
    ax.vlines(xstart, y+epsilon, y-epsilon, color, lw=2)
    ax.vlines(xstop, y+epsilon, y-epsilon, color, lw=2)

def plot_colorbar(ax, y1, y2, intervals, vmin, vmax, cmap='viridis',orientation='horizontal'):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    I = list(intervals)
    for i in range(len(I)):
        inter = I[i]
        y = y1 if (i % 2) else y2
        plot_interval(ax, y, norm(inter[0]), norm(inter[1]))
    
    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm, 
                                            orientation=orientation, 
                                            ticks=[]#[vmin,vmax]
                                            )
    cb1.outline.set_visible(False)

    # Hacky way to make colorbar smaller
    ax.set_aspect(1/20)

def plot_mapper(ax,plot_layout='centroid',out_location = "./out",cmap='viridis',spring_k = -1,big_component=True,adjust_node_size=False):
    
    if plot_layout not in ['centroid', 'spring']:
        print("Error! Invalid plot_layout")
        return -1
    
    # Get Mapper data
    G = nx.read_gml(out_location + "/graph.gml")
    org_node_sizes = np.loadtxt(out_location + '/node_sizes.csv',  delimiter=',') 
    node_colors = np.loadtxt(out_location + '/node_colors.csv',  delimiter=',') 
    vdims = np.loadtxt(out_location + '/vdims.csv', delimiter=',') 
    intervals = np.loadtxt(out_location + '/intervals.csv', delimiter=',') 
    node_locations = np.loadtxt(out_location + '/node_locations.csv', delimiter=',') 

    node_sizes = org_node_sizes

    if adjust_node_size:
        node_sizes = np.array([(i/300)**(0.4)*300 for i in org_node_sizes])

    giant = G
    if big_component:
        # Get giant component
        sub_graphs = nx.connected_component_subgraphs(G)
        giant = max(sub_graphs, key=len)

    # Find what's in this componenty
    mask = [int(a)-1 for a in list(giant.nodes())]
    giant_sizes = np.array(node_sizes)[mask]
    giant_colors = np.array(node_colors)[mask]

    vmin = vdims[0]
    vmax = vdims[1]

    # Build node locations list
    locations = {}
    
    if len(node_locations[0,:]) == 2:
        for i in range(node_locations.shape[0]):
            locations[str(i+1)] = node_locations[i,:]#(node_locations[i,2], - node_locations[i,0])
    else:
        for i in range(node_locations.shape[0]):
            locations[str(i+1)] = (node_locations[i,2], - node_locations[i,0])

    # Draw graph
    if plot_layout=='centroid':
        nx.draw(giant,
            locations,
            ax = ax,
            node_size = giant_sizes,
            node_color = giant_colors, 
            cmap = cmap,
            vmin = vmin, 
            vmax = vmax,
            )
        #print(ax.get_xlim())

        # Force graphs to have same aspect as orginal image
        ax.set_aspect(1)

    else:
        k = spring_k
        if k == -1:
            k = 2/np.sqrt(len(giant.nodes()))

        nx.draw(giant,
            nx.fruchterman_reingold_layout(giant,k=k),
            ax = ax,
            node_size = giant_sizes,
            node_color = giant_colors, 
            cmap = cmap,
            vmin = vmin, 
            vmax = vmax,
            clip_on=False,
            zorder=100
            )
        #ax.set_aspect(46.49865322064697/456.5517659320158)
        #ax.set_xlim((, 1456.5517659320158))

    
        
    return (vmin, vmax, intervals)


def draw_slices(V, show_slices, ax, cmap, vmin, vmax):
    i = 0
    
    for s in show_slices:
        shape = V[:,s,:].shape
        ax[i].imshow(V[:,s,:], cmap=cmap, vmin = vmin, vmax = vmax,aspect='equal',interpolation="none")#extent=[0,100,0,100/(1512/254)])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis('off')
        i += 1