# import os.path as p
# import click
# import pandas as pd
# import spatial_nx as sn
# import subsample as ss
#
# @cli.command()
# @click.argument('csv')
# @click.argument('samplesize')
# @click.option('-x', default='lon', help='Column header for x axis or longitudinal coordinates')
# @click.option('-y', default='lat', help='Column header for y axis or latitudinal coordinates')
# @click.option('--dist', '-d', default='haversine', help="Function to use for calculating distance. Valid options are 'euclidian' or 'haversine'. Defaults to haversine for calculating distance on a sphere")
# @click.option('--output', '-o', default=None, help='Path to output file, defaults to ./<csvfilename>.subsample.csv')
# # @click.option('--groups', 'g', default=None, help='Column header containing group names. If argument provided subsampling is done within groups, defaults to None')
#
# def cli(csv, samplesize, x, y, dist, output):
#     if output is None:
#         dirname, basename = p.split(csv)
#         filename, extension = p.splitext(basename)
#         newfilename = '{}.subsample{}'.format(filename, extension)
#         output = p.join(dirname, newfilename)
#
#     df = pd.read_csv(csv)
#     G = sn.SpatialGraph(df[[x, y]].values)
#     Sub = ss.Subsample(G, n)
#     sub_df = df.take(Sub.Sg.ix)
#     sub_df.to_csv(output, index=False)
