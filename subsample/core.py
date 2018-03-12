"""
Compute or approximate the most uniformly distributed subset of a given set of points
"""

import sys
import math as mt
import itertools as it
import numpy as np
import scipy.spatial as sp
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ['SpatialGraph', 'SubsampleGraph', 'subsample', 'subsample_df', 'plot_subsample']

class SpatialGraph(nx.Graph):
    def __init__(self, coords, graph='complete', surface='plane'):
        if len(coords) < 2:
            sys.exit('<br>Error!<br>Input must be greater than 1<br>')
        super(SpatialGraph, self).__init__()
        self.coords = coords
        self.graph_type = graph
        self.surface = surface
        self.add_nodes_from(np.arange(coords.shape[0]))
        if graph is 'complete':
            self.complete()
        elif graph is 'delaunay':
            self.delaunay()
        elif graph is 'min_edge':
            self.min_edge()
        else:
            sys.exit('<br>Error!<br><br>"{}" not a valid graph argument<br>'.format(graph))

    def complete(self):
        edges = np.array(list(it.combinations(self.nodes, 2)))
        a = self.coords[edges[:,0]]
        b = self.coords[edges[:,1]]
        if self.surface == 'plane':
            weights = euclidean(a, b)
        elif self.surface == 'sphere':
            weights = haversine(a, b)
        weighted_edges = np.rec.fromarrays((edges[:,0], edges[:,1], weights))
        self.add_weighted_edges_from(weighted_edges)

    def delaunay(self):
        if self.surface is 'sphere':
            sys.exit('Computing a delaunay graph on a sphere is not supported')
        tri = sp.Delaunay(self.coords)
        s = tri.simplices
        edges = np.vstack((s[:,0:2], s[:,1:3], s[:,[0, -1]]))
        a = self.coords[edges[:,0]]
        b = self.coords[edges[:,1]]
        if self.surface == 'plane':
            weights = euclidean(a, b)
        weighted_edges = np.rec.fromarrays((edges[:,0], edges[:,1], weights))
        self.add_weighted_edges_from(weighted_edges)

    def min_edge(self):
        edges = np.array(list(it.combinations(self.nodes, 2)))
        a = self.coords[edges[:,0]]
        b = self.coords[edges[:,1]]
        if self.surface == 'plane':
            weights = euclidean(a, b)
        elif self.surface == 'sphere':
            weights = haversine(a, b)
        weighted_edges = np.rec.fromarrays((edges[:,0], edges[:,1], weights))
        TempG = nx.Graph()
        TempG.add_weighted_edges_from(weighted_edges)
        min_edges = []
        for n in TempG.nodes:
            edges = TempG.edges(n, data=True)
            min_edge = min(edges, key=lambda x:x[2]['weight'])
            min_edges.append(min_edge)
        self.add_edges_from(min_edges)
        TempG.clear()


# ******************************************************************************

# ******************************************************************************
class SubsampleGraph(nx.Graph):
    def __init__(self, G, n, method='iter_drop_shortest'):
        if len(G) < n:
            sys.exit('<br>Error!<br>Subsample must be smaller than sample<br>')
        super().__init__(G)
        self.coords = G.coords
        self.graph_type = G.graph_type
        self.surface = G.surface
        self.N = len(G)
        self.n = n
        if method is 'iter_drop_shortest':
            self.iter_drop_shortest()
        elif method is 'max_sum_of_min_edges':
            self.max_sum_of_min_edges()
        elif method is 'max_mean_area':
            self.max_mean_area()
        # elif method is 'max_sum_of_edges':
            # self.max_sum_of_edges()
        # elif method is 'max_median_area':
        #     self.max_median_area()
        # elif method is 'max_area_sum':
        #     self.max_area_sum()
        # elif method is 'min_area_var':
        #     self.min_area_var()
        else:
            sys.exit('<br>Error!<br><br>"{}" not a valid method argument<br>'.format(method))

    def iter_drop_shortest(self):
        for i in range(self.N - self.n):
            shortest_edge = min(self.edges(data=True), key=lambda x:x[2]['weight'])
            node1 = shortest_edge[0]
            node2 = shortest_edge[1]
            edge1 = sorted(self.edges(node1, data=True), key=lambda x:x[2]['weight'])[1]
            edge2 = sorted(self.edges(node2, data=True), key=lambda x:x[2]['weight'])[1]
            shortest_adj_edge = min([edge1, edge2], key=lambda x:x[2]['weight'])
            self.remove_node(shortest_adj_edge[0])
        self.coords = self.coords[self.nodes]

    def max_sum_of_min_edges(self):
        subgraphs = []
        min_edges_list = []
        sums = []
        for subset in it.combinations(self.nodes, self.n):
            Sg = nx.Graph(self.subgraph(subset))
            min_edges = []
            min_weights = []
            for n in Sg.nodes:
                edges = Sg.edges(n, data=True)
                min_edge = min(edges, key=lambda x:x[2]['weight'])
                min_edges.append(min_edge)
                min_weight = min_edge[2]['weight']
                min_weights.append(min_weight)
            subgraphs.append(Sg)
            min_edges_list.append(min_edges)
            sums.append(sum(min_weights))
        best_ix = sums.index(max(sums))
        best_graph = subgraphs[best_ix]
        self.clear()
        self.add_nodes_from(best_graph.nodes(data=True))
        e1 = []
        e2 = []
        weight = []
        for edge in best_graph.edges(data=True):
            e1.append(edge[0])
            e2.append(edge[1])
            weight.append(edge[2]['weight'])
        self.add_weighted_edges_from(zip(e1, e2, weight))
        self.coords = self.coords[self.nodes]
        self.min_edges = min_edges_list[best_ix]

    def max_mean_area(self):
        if self.surface is 'sphere':
            sys.exit('Cannot yet calculate area on sphere')
        subgraphs = []
        means = []
        for subset in it.combinations(self.nodes, self.n):
            Sg = nx.Graph(self.subgraph(subset))
            Sg.surface = self.surface
            compute_areas(Sg)
            subgraphs.append(Sg)
            means.append(np.mean([r[1]['area'] for r in Sg.nodes(data=True)]))
        best_ix = means.index(max(means))
        best = subgraphs[best_ix]
        self.clear()
        self.add_nodes_from(best.nodes(data=True))
        e1 = []
        e2 = []
        weight = []
        for edge in best.edges(data=True):
            e1.append(edge[0])
            e2.append(edge[1])
            weight.append(edge[2]['weight'])
        self.add_weighted_edges_from(zip(e1, e2, weight))
        self.coords = self.coords[self.nodes]



    # def max_sum_of_edges(self):
    #     subgraphs = []
    #     sums = []
    #     for subset in it.combinations(self.G.nodes, self.n):
    #         Sg = nx.Graph(self.G.subgraph(subset))
    #         weight_sum = Sg.size(weight='weight')
    #         sums.append(weight_sum)
    #         subgraphs.append(Sg)
    #     best_ix = sums.index(max(sums))
    #     self.Sg = subgraphs[best_ix]
    #     self.Sg.surface = self.G.surface
    #
    #
    #
    # def max_area(self):
    #     if self.surface is 'sphere':
    #         sys.exit('Cannont yet calculate area on sphere')
    #     self.G.add_nodes_from(self.G.nodes, radius=0, area=0)
    #     subgraphs = []
    #     sums = []
    #     for subset in it.combinations(self.G.nodes, self.n):
    #         Sg = nx.Graph(self.G.subgraph(subset))
    #         Sg.surface = self.G.surface
    #         compute_radii(Sg)
    #         compute_areas(Sg)
    #         subgraphs.append(Sg)
    #         sums.append(sum([r[1]['area'] for r in Sg.nodes(data=True)]))
    #     best_ix = sums.index(max(sums))
    #     self.Sg = subgraphs[best_ix]
    #
    #
    #
    # def min_area_var(self):
    #     if self.surface is 'sphere':
    #         sys.exit('Cannont yet calculate area on sphere')
    #     subgraphs = []
    #     var = []
    #     for subset in it.combinations(self.G.nodes, self.n):
    #         Sg = nx.Graph(self.G.subgraph(subset))
    #         Sg.surface = self.G.surface
    #         compute_radii(Sg)
    #         compute_areas(Sg)
    #         subgraphs.append(Sg)
    #         var.append(np.var([r[1]['area'] for r in Sg.nodes(data=True)]))
    #     best_ix = var.index(min(var))
    #     self.Sg = subgraphs[best_ix]


# ******************************************************************************

# ******************************************************************************
def euclidean(a, b):
    ax, ay = a[:,0], a[:,1]
    bx, by = b[:,0], b[:,1]
    distances = np.sqrt((ax-bx)**2 + (ay-by)**2)
    return(distances)

def haversine(a, b):
    ax, ay = np.radians(a[:,0]), np.radians(a[:,1])
    bx, by = np.radians(b[:,0]), np.radians(b[:,1])
    dx = bx - ax
    dy = ay - by
    a = np.square(np.sin(dy*0.5))+np.cos(ay)*np.cos(by)*np.square(np.sin(dx*0.5))
    c = 2 * np.arcsin(np.sqrt(a))
    distances = (6367 * c).astype(int)
    return(distances)

def compute_radii(G):
    G.add_nodes_from(G.nodes, radius=0)
    if G.surface is 'plane':
        for edge in sorted(G.edges(data=True), key=lambda x:x[2]['weight']):
            n1 = edge[0]
            r1 = G.node[n1]['radius']
            n2 = edge[1]
            r2 = G.node[n2]['radius']
            if r1 and r2 != 0:
                pass
            elif r1 != 0:
                short_edge = min(G.edges(n2, data=True), key=lambda x:(x[2]['weight'] -  G.node[x[1]]['radius']))
                G.node[n2]['radius'] = short_edge[2]['weight'] - G.nodes[short_edge[1]]['radius']
            elif r2 != 0:
                short_edge = min(G.edges(n1, data=True), key=lambda x:(x[2]['weight'] - G.node[x[1]]['radius']))
                G.node[n1]['radius'] = short_edge[2]['weight'] - G.nodes[short_edge[1]]['radius']
            else:
                radius = edge[2]['weight'] / 2
                G.node[n1]['radius'] = radius
                G.node[n2]['radius'] = radius
    elif G.surface is 'sphere':
        sys.exit('Cannot yet compute radii on a sphere')

def compute_areas(G):
    if 'radius' not in nx.get_node_attributes(G, 'radius'):
        compute_radii(G)
    G.add_nodes_from(G.nodes, area=0)
    for n in G.nodes(data=True):
        radius = n[1]['radius']
        area = mt.pi * radius**2
        G.node[n[0]]['area'] = area

def _min_edge_graph(G):
    Mg = nx.Graph()
    Mg.add_nodes_from(G.nodes)
    min_edges = []
    for n in G.nodes:
        edges = G.edges(n, data=True)
        min_edge = min(edges, key=lambda x:x[2]['weight'])
        min_edges.append(min_edge)
    Mg.add_edges_from(min_edges)
    return(Mg)

def plot_subsample(G, Sg, radii=False, min_dist=False, show=True, save=False):
    pos = G.coords[G.nodes]
    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, ax=ax, node_size=20, node_color='b', width=0, with_labels=False)
    nx.draw_networkx(Sg, pos=pos, ax=ax, node_size=20, node_color='r', width=0, with_labels=False)
    if min_dist:
        if hasattr(Sg, 'min_edges'):
            Mg = nx.Graph()
            Mg.add_nodes_from(Sg.nodes)
            Mg.add_edges_from(Sg.min_edges)
        else:
            Mg = _min_edge_graph(Sg)
        nx.draw_networkx(Mg, pos=pos, ax=ax, node_size=0, node_color='r', width=1, with_labels=False)
    if radii:
        if len(nx.get_node_attributes(Sg, 'radius')) != len(Sg):
            compute_radii(Sg)
        radii = np.array([r[1]['radius'] for r in Sg.nodes(data=True)])
        circles = np.stack((Sg.coords[:,0], Sg.coords[:,1], radii), axis=1)
        for i in circles:
            circle = plt.Circle((i[0], i[1]), i[2], fill=False, edgecolor='b')
            ax.add_artist(circle)
    plt.axis('equal')
    if save:
        plt.savefig('subsample-plot.png')
    elif show:
        plt.show()

def subsample(coords, n, method='iter_drop_shortest', graph='complete', surface='plane', show_plot=False, save_plot=False):
    G = SpatialGraph(coords, graph=graph, surface=surface)
    Sg = SubsampleGraph(G, n, method=method)
    ix = Sg.nodes()
    sub_coords = Sg.coords
    if save_plot:
        plot_subsample(G, Sg, show=False, save=True)
    if show_plot:
        plot_subsample(G, Sg, show=True, save=False)
    return(ix, sub_coords)

def subsample_df(df, n, x='lon', y='lat', method='iter_drop_shortest', graph='complete', surface='sphere', show_plot=False, save_plot=False):
    coords = df.as_matrix(columns=[x, y])
    ix, sub_coords = subsample(coords, n, method=method, graph=graph, surface=surface, show_plot=show_plot, save_plot=save_plot)
    sub_df = df.iloc[list(ix)]
    return(sub_df)
