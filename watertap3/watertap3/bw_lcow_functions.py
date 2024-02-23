import os
os.environ["OMP_NUM_THREADS"] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.cluster import KMeans, DBSCAN
import geopandas as gp

import math
from shapely.geometry import MultiPoint, Point, Polygon
from shapely import wkt
import haversine as hs

from watertap3.truck_pipe_cost_functions import elevation,pipe_costing
from watertap3.utils import watertap_setup, get_case_study, run_model 
from watertap3.utils import run_watertap3, run_model_no_print, run_and_return_model
from watertap3.utils.post_processing import get_results_table
from watertap3.truck_pipe_cost_functions import elevation

from wells_dijkstra_algorithm import *


# Read relevant files

# Brackish water USGS files
# bw_df = pd.read_csv('/Users/mhardika/Documents/AMO/GeoToolAll_Methods/Water Source Data/Brackish/brackish_sites_with_metrics_baseline_dwi_updated_costs_transport_updated_basis_1.csv')
bw_df = pd.read_csv('/Users/mhardika/Documents/AMO/GeoToolAll_Methods/Water Source Data/Brackish/brackish_sites_baseline_dwi_5Feb23.csv')

# Function to form cluster

def form_cluster(df,n_clusters = 3):

    kmeans = KMeans(n_clusters = n_clusters, init ='k-means++', random_state=42)
    kmeans.fit(df[df.columns[2:4]])  # Compute k-means clustering.
    
    # Assign cluster IDs to the column
    df['cluster_id'] = kmeans.fit_predict(df[df.columns[2:4]])

    # Coordinates of cluster centers.
    centers = kmeans.cluster_centers_
    centers_array = []
    for idx,row in df.iterrows():
        centers_array.append(centers[int(row['cluster_id'])])
    df['centers'] = centers_array
    
    return df


def create_cluster_id_df(state_alpha):

    bw_cluster_kmeans = pd.DataFrame()

    # Copy the rows for a state and copy select columns for those states
    bw_state_df = bw_df[bw_df['state_alpha'] == state_alpha].copy()
    bw_state_df_loc = bw_state_df[['state_alpha','unique_site_ID','Latitude','Longitude','county_nm',
                                       'well_depth_ft','well_yield','TDS_kgm3','elec_price','well_field_lcow']]

    # kmeans doesn't work for fewer than 3 points. Calculate centroid for these cases
    if len(bw_state_df)<=3:
        bw_state_df_loc['cluster_id'] = 0
        bw_state_df_loc['well_yield'] = bw_state_df['well_yield']
        bw_cluster_kmeans = pd.concat([bw_cluster_kmeans,bw_state_df_loc])
    
    else:
        # First pass at clustering for a state and return dataframe with column 'centers'
        bw_state_df_loc = form_cluster(bw_state_df_loc)

        # Assign well yield
        bw_state_df_loc['well_yield'] = bw_state_df['well_yield']
        bw_state_df_loc = bw_state_df_loc.sort_values(['cluster_id'])

        # To keep track of clusters and new created clusters if the maximum capacity of a treatment plant is exceeded
        prev_max_cluster_label = 0

        # Dataframe for each state with their respective cluster id
        cluster_state_df = pd.DataFrame()

        for cluster_id in bw_state_df_loc['cluster_id'].unique():
            cluster_sub_df = pd.DataFrame()
            temp_df = pd.DataFrame()

            # Check if maximum capacity of treatment plant is exceeded
            well_yield_total = sum(bw_state_df_loc[bw_state_df_loc['cluster_id']==cluster_id]['well_yield'])

            max_capacity = math.ceil(27.5*0.043813)  #---> Carlsbad is 50 MGD. Kay Bailey is 27.5 MGD
            if well_yield_total > max_capacity: 
                # If exceed increase number of clusters in the original cluster and redo
                cluster_sub_df = bw_state_df_loc[bw_state_df_loc['cluster_id']==cluster_id].copy()

                temp_df = form_cluster(cluster_sub_df, n_clusters = math.ceil(well_yield_total/max_capacity))
                temp_df = temp_df.sort_values(['cluster_id'])

                if cluster_id == 0:
                    temp_df['cluster_id'].update(prev_max_cluster_label + temp_df['cluster_id'])
                else:
                    temp_df['cluster_id'].update(prev_max_cluster_label + temp_df['cluster_id'] + 1)
                prev_max_cluster_label = max(temp_df['cluster_id'])
                
                cluster_state_df = pd.concat([cluster_state_df,temp_df],ignore_index=True)
            else:
                # If maximum capacity is not exceeded add to state data frame
                temp_bw_state_df_loc = bw_state_df_loc[bw_state_df_loc['cluster_id']==cluster_id].copy()
                if cluster_id == 0:
                    temp_bw_state_df_loc['cluster_id'] = prev_max_cluster_label 
                else:
                    temp_bw_state_df_loc['cluster_id'] = prev_max_cluster_label + 1

                cluster_state_df = pd.concat([cluster_state_df,temp_bw_state_df_loc],ignore_index=True)
                prev_max_cluster_label = max(cluster_state_df['cluster_id'])

            print(cluster_state_df['cluster_id'].unique())

        bw_cluster_kmeans = pd.concat([bw_cluster_kmeans,cluster_state_df])

    return bw_cluster_kmeans

# Function to find well closest to the center of the cluster
def select_closest_well(df):
    max_dist = 1000

    for idx, row in df.iterrows():
        bw_long = row['Longitude']
        bw_lat = row['Latitude']

        bw_loc = (bw_lat,bw_long)

        cent_long = row['centers'][1]
        cent_lat = row['centers'][0]
        
        cent_loc = (cent_lat,cent_long)

        dist_km = hs.haversine(cent_loc,bw_loc)

        if dist_km < max_dist:
            max_dist = dist_km
            well = row['unique_site_ID']
    
    return well

# Function to find well closest to the center of the cluster
def select_closest_well_subcluster(df):
    max_dist = 1000

    for idx, row in df.iterrows():
        bw_long = row['Longitude']
        bw_lat = row['Latitude']

        bw_loc = (bw_lat,bw_long)

        cent_long = row['subcluster_centers'][1]
        cent_lat = row['subcluster_centers'][0]
        
        cent_loc = (cent_lat,cent_long)

        dist_km = hs.haversine(cent_loc,bw_loc)

        if dist_km < max_dist:
            max_dist = dist_km
            well = row['unique_site_ID']
    
    return well


# Function to form sub-clusters and find the centroid of the subcluster
def find_subcluster_centroid(coords,cluster_range = 5):
    kms_per_radian = 6371.0088
    range_km = cluster_range*1.609343502101154
    epsilon = range_km/ kms_per_radian

    # Form subcluster
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_

    # get the number of clusters
    num_clusters = len(set(cluster_labels))

    # turn the clusters in to a pandas series, where each element is a cluster of points
    clusters = pd.Series((coords[cluster_labels==n] for n in range(num_clusters)))
    centroids = []

    for ea in clusters:
        centroids.append((MultiPoint(ea).centroid.x, MultiPoint(ea).centroid.y))

    centroids = np.array(centroids)
    return [centroids,cluster_labels]


# Create subclusters to calculate pipe transport costs
def create_subcluster(state_df):
    subcluster_df = pd.DataFrame(columns = ['subcluster_id','centroid'])

    for cluster_id in state_df['cluster_id'].unique():
        
        coords_input = state_df[state_df['cluster_id']==cluster_id][['Latitude', 'Longitude']].to_numpy()
        centroids,cluster_labels = find_subcluster_centroid(coords_input,10)

        loc = [Point(xy) for xy in zip(centroids[:,1],centroids[:,0])]
        loc = gp.GeoDataFrame(geometry = loc, crs='EPSG:4326')
        loc.geometry = loc.geometry.to_crs('EPSG:4326')

        # Assign subcluster label
        temp = pd.DataFrame(columns = ['subcluster_id','centroid'])
        temp['subcluster_id'] = cluster_labels
        temp['centroid'] = loc.geometry[cluster_labels].values

        subcluster_df= pd.concat([subcluster_df,temp])

    return [subcluster_df['subcluster_id'].values.tolist(), 
            subcluster_df['centroid'].values.x.tolist(), 
            subcluster_df['centroid'].values.y.tolist()]


# Function to create brackish well table that includes cluster IDs and subcluster IDs. This done for one state at a time
def create_bw_cluster_subcluster_df(state_alpha,bw_cluster_kmeans):
    well_elevation_list = []
    centroid_long_list = []
    centroid_lat_list = []
    centroid_elevation_list = []
    subcluster_id_list = []
    subcluster_centroid_long_list = []
    subcluster_centroid_lat_list = []
    subcluster_centers_list = []
    subcluster_elevation_list = []
    subcluster_dist_list = []

    temp_state = bw_cluster_kmeans[bw_cluster_kmeans.state_alpha == state_alpha].copy()

    # Read state elevation file
    PATH = r'\Users\mhardika\Documents\AMO\GeoToolAll_Methods\Water Source Data\Brackish\elevation_data\\'
    elev_db = pd.read_csv(PATH + '\\' + state_alpha.lower()+'.csv', index_col = 0)
    elev_db = elev_db.set_index('unique_site_ID')

    # Iterate through all the cluster IDs
    for cluster_id in temp_state.cluster_id.unique():
        temp_cluster = temp_state[temp_state.cluster_id==cluster_id].copy()
        centroid_well = select_closest_well(temp_cluster)

        centroid_long = temp_cluster[temp_cluster['unique_site_ID']==centroid_well]['Longitude'].values[0]
        centroid_lat = temp_cluster[temp_cluster['unique_site_ID']==centroid_well]['Latitude'].values[0]

        [subcluster_id,subcluster_centroid_long,subcluster_centroid_lat] = create_subcluster(temp_state)

        # Iterate through the list of wells to find the well elevation and assign the location of the centroid
        for well in temp_cluster['unique_site_ID'].unique():
            bw_long = temp_cluster[temp_cluster['unique_site_ID']==well]['Longitude'].values[0]
            bw_lat = temp_cluster[temp_cluster['unique_site_ID']==well]['Latitude'].values[0]

            well_elevation_list.append(elev_db.loc[well].values[0])
            # try:
            #     well_elevation_list.append(elevation(bw_lat,bw_long))
            # except KeyError:
            #     print('Something went wrong at well:',bw_lat,bw_long)
            #     well_elevation_list.append(0)

            centroid_long_list.append(centroid_long)
            centroid_lat_list.append(centroid_lat) 

    # Add the sub-cluster id and location
    subcluster_id_list.extend(subcluster_id)

    for i in range(0,len(subcluster_centroid_lat)):
        subcluster_centers_list.append([subcluster_centroid_lat[i],subcluster_centroid_long[i]])

    bw_cluster_kmeans['well_elevation'] = well_elevation_list
    bw_cluster_kmeans['centroid_long'] = centroid_long_list
    bw_cluster_kmeans['centroid_lat'] = centroid_lat_list
    bw_cluster_kmeans['subcluster_id'] = subcluster_id_list
    bw_cluster_kmeans['subcluster_centers'] = subcluster_centers_list

    
    # Assign well closest to subcluster centroid as subcluster centroid
    for cluster_id in bw_cluster_kmeans.cluster_id.unique():
        centroid_list = []
        temp_cluster = bw_cluster_kmeans[bw_cluster_kmeans.cluster_id==cluster_id].copy()
        for subcluster_id in temp_cluster.subcluster_id.unique():
            temp_subcluster = temp_cluster[temp_cluster.subcluster_id==subcluster_id].copy()
            subcluster_centroid = select_closest_well_subcluster(temp_subcluster)
            centroid_list.append(subcluster_centroid)

        # for well in temp_cluster['unique_site_ID'].unique():
        for idx, row in temp_cluster.iterrows():
            # Identify the subcluster id
            temp_subcluster_id = row['subcluster_id']
            
            try:
                subcluster_centroid_long_list.extend(temp_cluster[temp_cluster['unique_site_ID']==centroid_list[temp_subcluster_id]]['Longitude'].values[0])
                subcluster_centroid_lat_list.extend(temp_cluster[temp_cluster['unique_site_ID']==centroid_list[temp_subcluster_id]]['Latitude'].values[0])

            except:
                subcluster_centroid_long_list.extend(temp_cluster[temp_cluster['unique_site_ID']==centroid_list[temp_subcluster_id]]['Longitude'].values)
                subcluster_centroid_lat_list.extend(temp_cluster[temp_cluster['unique_site_ID']==centroid_list[temp_subcluster_id]]['Latitude'].values)        


    bw_cluster_kmeans['subcluster_long'] = subcluster_centroid_long_list
    bw_cluster_kmeans['subcluster_lat'] = subcluster_centroid_lat_list

    # Add distance between sub-cluster centroid and cluster centroid
    for idx, row in bw_cluster_kmeans.iterrows():
        subcluster_loc = (row['subcluster_lat'],row['subcluster_long'])
        cent_loc = (row['centroid_lat'],row['centroid_long'])

        dist_km = hs.haversine(cent_loc,subcluster_loc)
        subcluster_dist_list.append(dist_km)

        # Use well elevation list to find cluster elevation
        try:
            temp_elevation_cluster = bw_cluster_kmeans[(bw_cluster_kmeans['Latitude']==row['centroid_lat']) & (bw_cluster_kmeans['Longitude']==row['centroid_long'])]['well_elevation'].values[0]
        except:
            temp_elevation_cluster = bw_cluster_kmeans[(bw_cluster_kmeans['Latitude']==row['centroid_lat']) & (bw_cluster_kmeans['Longitude']==row['centroid_long'])]['well_elevation']
        

        centroid_elevation_list.append(temp_elevation_cluster)
        
        # Use well elevation list to find subcluster elevation
        try:
            temp_elevation_subcluster = bw_cluster_kmeans[(bw_cluster_kmeans['Latitude']==row['subcluster_lat']) & (bw_cluster_kmeans['Longitude']==row['subcluster_long'])]['well_elevation'].values[0]
        except:
            temp_elevation_subcluster = bw_cluster_kmeans[(bw_cluster_kmeans['Latitude']==row['subcluster_lat']) & (bw_cluster_kmeans['Longitude']==row['subcluster_long'])]['well_elevation']


        subcluster_elevation_list.append(temp_elevation_subcluster)


    bw_cluster_kmeans['centroid_elevation'] = centroid_elevation_list
    bw_cluster_kmeans['subcluster_to_centroid_dist_km'] = subcluster_dist_list
    bw_cluster_kmeans['subcluster_elevation'] = subcluster_elevation_list

    # Add cost of pipe transport from well to the subcluster centroid
    pipe_lcow_well_subcluster_list = []

    for idx, row in bw_cluster_kmeans.iterrows():

        # Distance between subcluster centroid and well
        subcluster_loc = (row['subcluster_lat'],row['subcluster_long'])
        well_loc = (row['Latitude'],row['Longitude'])
        dist_km_well = hs.haversine(well_loc,subcluster_loc)

        # Elevation gain between subcluster centroid and well
        elev_gain = row['well_elevation'] - row['subcluster_elevation']

        if elev_gain<0:
            elev_gain = 1e-5    

        pipe_lcow_well_subcluster = pipe_costing(row['well_yield']*3600*24, dist_km_well,
                                                elev_gain=elev_gain,electricity_rate=row['elec_price'])
    
        pipe_lcow_well_subcluster_list.append(pipe_lcow_well_subcluster)

    bw_cluster_kmeans['pipe_lcow_well_subcluster'] = pipe_lcow_well_subcluster_list


    return bw_cluster_kmeans


# Function to intialize dictionary for the Dijkstra algorithm
def intialise_dict(df):

    # nodes are the well unique ids
    nodes = np.concatenate((['treatment_node'],df['subcluster_id'].unique()),axis=0)
    init_graph = {}

    # Use the cluster centroid as the treatment centroid
    treatment_node_loc = (df['centroid_lat'].values[0],df['centroid_long'].values[0])

    # Create a dictionary for distances
    for node in nodes[1::]:
        init_graph[node] = {}

    # Iterate through each node
    for i in range(1,len(nodes[1::])): 
        node0 = df[df['subcluster_id']==int(nodes[i])]
        node0_long = node0['subcluster_long'].values[0]
        node0_lat =  node0['subcluster_lat'].values[0]
        node0_loc = (node0_lat,node0_long)

        min_sub_node = ''
        min_dist = 1e9
        
        for sub_node in nodes[i::]:
            # Skip itself in the interval iteration
            if sub_node == nodes[i]:
                continue
            else:
                node_other = df[df['subcluster_id']== int(sub_node)]
                node_other_long = node_other['subcluster_long'].values[0]
                node_other_lat =  node_other['subcluster_lat'].values[0]
                node_other_loc = (node_other_lat,node_other_long)
                dist = hs.haversine(node0_loc,node_other_loc)
                
                # Find the node closest to the outer loop node
                if dist<min_dist:
                    min_dist=dist
                    min_sub_node=sub_node           
        init_graph[nodes[i]][min_sub_node] = min_dist

    # Check if each node is connected and connect only nodes less than 160 km to cluster centroid
    for node in nodes[1::]:
        node_other = df[df['subcluster_id']== int(node)]
        node_other_long = node_other['subcluster_long'].values[0]
        node_other_lat =  node_other['subcluster_lat'].values[0]
        node_other_loc = (node_other_lat,node_other_long)

        dist = hs.haversine(node_other_loc,treatment_node_loc)
        
        # Checks if the node and the treatment node are the same
        if dist == 0:
            init_graph[node]['treatment_node'] = dist + 1e-10
        # If node is not connecting to any other node and the node is close to the treatment node, connect with treatment node
        elif len(init_graph[node])==0 and dist< 160:
            init_graph[node]['treatment_node'] = dist

    return nodes,init_graph


def find_shortest_path(cluster_df):
    cluster_df_sort = cluster_df.sort_values(['state_alpha','cluster_id','subcluster_centroid_dist'],
                                                         ascending=[True,True,False]).groupby(['state_alpha','cluster_id']).apply(pd.DataFrame)

    nodes,init_graph = intialise_dict(cluster_df_sort)

    graph = Graph(nodes, init_graph)

    path_array = []
    for node in nodes[1::]:
        previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node= node)
        path = print_result(previous_nodes, shortest_path, start_node=node, target_node="treatment_node")
        path_array.append(path)
    
    # Reversing so that the direction is from subcluster well to centroid of cluster
    path_array = path_array[::-1]
    return(path_array, nodes, init_graph)


# Function to calculate the pipe volume at each subcluster node

def calc_pipe_vol(nodes,path_array,sub_cluster_well_vol):

    # Build dictionary of levels in the shortest path
    levels = {}
    # Iterate through the shortest path
    for i in range(0,len(path_array)):
        level = 0
        # This is to remove the shortest path
        path_array_1 = path_array[i][:-1]
        for j in range(1,len(path_array_1)+1):
            flag = 0
            # If it is the first shortest path
            if i == 0:
                levels[level] = [path_array_1[-j]]
            else:
                # Check if the node is already included in the previous level
                for check_level in levels.keys():
                    if path_array_1[-j] in levels[check_level]:
                        flag = 1
                if flag==1:
                    level = level + 1
                    continue
                else:
                    try:                     
                        levels[level].append(path_array_1[-j])   
                    except KeyError:
                        levels[level] = [path_array_1[-j]]
                    level = level + 1

    # Build dictionary of nodes connected to each node
    nodes_list = {}

    for i in range(0,len(path_array)):
        path_array_1 = path_array[i][:-1]
        for j in range(0,len(path_array_1)):
            nodes_list[path_array_1[j]] = [path_array_1[j]]

    for i in range(0,len(path_array)):
        path_array_1 = path_array[i][:-1]
        for j in range(1,len(path_array_1)):
            if path_array_1[j-1] in nodes_list[path_array_1[j]]:
                continue
            else:
                nodes_list[path_array_1[j]].append(path_array_1[j-1])

    # Function to iterate through levels and calculate pipe vol at each node
    pipe_vol = {}
    # Initialize the pipe volume
    for node in nodes[1::]:
        pipe_vol[node]=0

    level_keys = list(levels.keys())
    # Iterate through the levels
    for i in level_keys[::-1]:
        print()
        # Iterate through list of nodes at that level
        for j in range(0,len(levels[i])):
            print('Node:',levels[i][j])
            # Iterate through the nodes list and calculate the pipe volume
            for k in nodes_list[levels[i][j]]:
                print('Nodes included',k)
                if pipe_vol[levels[i][j]] == 0:
                    pipe_vol[levels[i][j]] = sub_cluster_well_vol[k]
                else:
                    pipe_vol[levels[i][j]] = pipe_vol[levels[i][j]] + pipe_vol[str(k)]
    
    return pipe_vol

# Function to calculate the pipe transport costing from subcluster centroid to cluster centroid as a function of shortest path
def calc_pipe_cost_subcluster_cluster(cluster_df,path_array,nodes,init_graph):
    # Assign the total volume of water produced by each sub-cluster
    sub_cluster_well_vol={}
    # nodes = cluster_df['subcluster_id'].unique()

    for node in nodes[1::]:
        sub_cluster_well_vol[node] = cluster_df[cluster_df['subcluster_id']==int(node)]['total_well_yield'].sum()

    # Distance at each node is fixed
    # Iterate to calculate volume
    visited_node = []
    pipe_dist = {}

    # Calculate pipe length in each section
    for i in range(0,len(path_array)):
        for j in range(0,len(path_array[i])-1):
            if path_array[i][j] in visited_node:
                continue
            else:
                dist = init_graph[path_array[i][j]][path_array[i][j+1]]
                visited_node.append(path_array[i][j])
                pipe_dist[path_array[i][j]] = dist

    # Calculate pipe volume going forward from each node

    pipe_vol = calc_pipe_vol(nodes,path_array,sub_cluster_well_vol)

    # Calculate the pipe transport cost

    cluster_pipe_to_treatment_lcow_dict = {}
    cluster_pipe_to_treatment_cost = 0

    for node in nodes[1::]:
        sub_cluster_lat = cluster_df[cluster_df['subcluster_id']==int(node)]['subcluster_lat'].unique()[0]
        sub_cluster_long = cluster_df[cluster_df['subcluster_id']==int(node)]['subcluster_long'].unique()[0]
        sub_cluster_elev = cluster_df[cluster_df['subcluster_id']==int(node)]['subcluster_elev'].unique()[0]

        centeroid_lat = cluster_df[cluster_df['subcluster_id']==int(node)]['centroid_lat'].unique()[0]
        centeroid_long = cluster_df[cluster_df['subcluster_id']==int(node)]['centroid_long'].unique()[0]
        cluster_elev =  cluster_df[cluster_df['subcluster_id']==int(node)]['centroid_elev'].unique()[0]

        elec_price = cluster_df[cluster_df['subcluster_id']==0]['elec_price'].unique()[0]

        elev_gain = cluster_elev - sub_cluster_elev
        
        # To avoid the zero elevation gain problem
        if elev_gain <= 0:
            elev_gain = 1e-5  

        cluster_pipe_to_treatment_lcow_dict[node] = pipe_costing(pipe_vol[node]*3600*24, pipe_dist[node], elev_gain = elev_gain,
                                        electricity_rate = elec_price)
        cluster_pipe_to_treatment_cost = cluster_pipe_to_treatment_cost + cluster_pipe_to_treatment_lcow_dict[node] * pipe_vol[node]

    cluster_pipe_to_treatment_lcow = cluster_pipe_to_treatment_cost/sum(pipe_vol.values())

    return cluster_pipe_to_treatment_lcow
   

# Create a condensed to subcluster level table. bw_cluster_kmeans for a single state is passed
def condense_subcluster_table(bw_df):

    # Inline functions to calculate weighted average tds and lcows
    calc_avg_tds = lambda x: np.average(x, weights=bw_df.loc[x.index, "well_yield"])
    calc_well_field_lcow = lambda x: np.average(x, weights=bw_df.loc[x.index, "well_yield"])
    calc_pipe_lcow = lambda x: np.average(x, weights=bw_df.loc[x.index, "well_yield"])

    condensed_subcluster_bw_df = pd.DataFrame()

    for cluster in bw_df['cluster_id'].unique():
        temp_cluster_df = bw_df[bw_df['cluster_id']==cluster]

        group_table  = temp_cluster_df.groupby(['state_alpha','cluster_id','subcluster_id']).agg(
                centroid_lat = ('centroid_lat','mean'),
                centroid_long = ('centroid_long','mean'),
                centroid_elev = ('centroid_elevation','mean'),

                subcluster_lat = ('subcluster_lat','mean'),
                subcluster_long = ('subcluster_long','mean'),
                subcluster_elev = ('subcluster_elevation','mean'),
                subcluster_centroid_dist = ('subcluster_to_centroid_dist_km','mean'),

                unique_site_ID = ('unique_site_ID',','.join),
                well_lat = ('Latitude', pd.Series.to_list),
                well_long = ('Longitude',pd.Series.to_list),
                county_nm = ('county_nm',pd.Series.to_list),
                well_elevation = ('well_elevation',pd.Series.to_list),
                well_depth_ft = ('well_depth_ft', pd.Series.to_list),
                well_yield = ('well_yield',pd.Series.to_list),
                total_well_yield = ('well_yield','sum'),
                tds_kgm3 = ('TDS_kgm3',pd.Series.to_list),
                avg_TDS_kgm3 = ('TDS_kgm3', calc_avg_tds),
                elec_price = ('elec_price','mean'),
                well_field_lcow = ('well_field_lcow',pd.Series.to_list),
                avg_well_field_lcow = ('well_field_lcow',calc_well_field_lcow),
                pipe_lcow = ('pipe_lcow_well_subcluster', pd.Series.to_list),
                avg_subcluster_pipe_lcow = ('pipe_lcow_well_subcluster', calc_pipe_lcow),
            ).reset_index()

        condensed_subcluster_bw_df = pd.concat((condensed_subcluster_bw_df,group_table))

    condensed_subcluster_bw_df.reset_index(inplace=True,drop=True)

    return condensed_subcluster_bw_df


def treatment_only_LCOW(capacity, tds, well_depth, elec_price):
    case_study = 'big_spring'
    scenario = 'dwi_a'
    desired_recovery = 1
    ro_bounds = 'other' # or 'seawater'

    m = watertap_setup(case_study=case_study, scenario=scenario)
    m = get_case_study(m=m)
    m = run_watertap3(m, desired_recovery=desired_recovery, ro_bounds=ro_bounds)

    m.fs.reverse_osmosis.membrane_area.unfix()
    m.fs.reverse_osmosis.feed.pressure.unfix()

    m.fs.big_spring_feed.flow_vol_in.fix(capacity) # capacity in m3s
    m.fs.big_spring_feed.conc_mass_in[0, 'tds'].fix(tds) # tds in kg/m3
    m.fs.well_field.lift_height.fix(well_depth)

    m.fs.costing_param.electricity_price = elec_price 
    m = run_and_return_model(m, objective=True,print_it=True)
    m, df = get_results_table(m=m, case_study='test', scenario=scenario)    
    lcow = m.fs.costing.LCOW.value()
    well_field_lcow = m.fs.well_field.LCOW()
    recovery = m.fs.costing.system_recovery()*100
    # clear_output(wait=True)
    # print(well_field_lcow)
    # Exclude the well field LCOW from the cost ($/m3 avg well field flow), well lcow ($/m3 of only that well) and brine volume in m3/day
    return (lcow-well_field_lcow,recovery, m.fs.deep_well_injection.flow_vol_in[0].value*3600*24)


# Calculate the LCOW of adding each cluster of wells one state at a time
def condense_cluster_table(condensed_subcluster_bw_df):

    # Inline functions to calculate weighted average tds and lcows
    calc_avg_tds = lambda x: np.average(x, weights=condensed_subcluster_bw_df.loc[x.index, "total_well_yield"])
    calc_well_field_lcow = lambda x: np.average(x, weights=condensed_subcluster_bw_df.loc[x.index, "total_well_yield"])
    calc_pipe_lcow = lambda x: np.average(x, weights=condensed_subcluster_bw_df.loc[x.index, "total_well_yield"])
    
    condensed_cluster_bw_df = pd.DataFrame()
    cluster_avg_pipe_subcluster_cluster_lcow_list = []

    # Condense the table further to the cluster level
    for cluster in condensed_subcluster_bw_df['cluster_id'].unique():
        cluster_df = condensed_subcluster_bw_df[condensed_subcluster_bw_df['cluster_id'] == cluster]

        group_table  = cluster_df.groupby(['state_alpha','cluster_id']).agg(
                centroid_lat = ('centroid_lat','mean'),
                centroid_long = ('centroid_long','mean'),
                centroid_elev = ('centroid_elev','mean'),

                subcluster_lat = ('subcluster_lat',pd.Series.to_list),
                subcluster_long = ('subcluster_long',pd.Series.to_list),
                subcluster_elev = ('subcluster_elev',pd.Series.to_list),
                subcluster_centroid_dist = ('subcluster_centroid_dist',pd.Series.to_list),
                unique_site_ID = ('unique_site_ID',','.join),

                cluster_total_well_yield = ('total_well_yield','sum'),
                cluster_avg_TDS_kgm3 = ('avg_TDS_kgm3', calc_avg_tds),
                elec_price = ('elec_price','mean'),
                cluster_avg_well_field_lcow = ('avg_well_field_lcow',calc_well_field_lcow),
                cluster_avg_pipe_well_subcluster_lcow = ('avg_subcluster_pipe_lcow', calc_pipe_lcow),
            ).reset_index()

        # Pipe transport lcow from subcluster to cluster along shortest path
        path_array,nodes,init_graph = find_shortest_path(cluster_df)
        cluster_avg_pipe_subcluster_cluster_lcow_list.append(calc_pipe_cost_subcluster_cluster(cluster_df,path_array,nodes,init_graph))

        condensed_cluster_bw_df = pd.concat((condensed_cluster_bw_df,group_table))

    condensed_cluster_bw_df.reset_index(inplace=True,drop=True)

    condensed_cluster_bw_df['cluster_avg_pipe_subcluster_cluster_lcow'] = cluster_avg_pipe_subcluster_cluster_lcow_list

    return condensed_cluster_bw_df

# Function to calculate all costs
def calc_cluster_lcow(condensed_cluster_bw_df,dist_to_dwi = 16.0934):

    cluster_treatment_lcow_list = []
    cluster_pipe_brine_lcow_list = []
    recovery_list = []

    for cluster in condensed_cluster_bw_df['cluster_id'].unique():
        cluster_df = condensed_cluster_bw_df[condensed_cluster_bw_df['cluster_id'] == cluster]
    # Calculate treatment cost
        treatment_lcow, recovery, brine_vol =  treatment_only_LCOW(cluster_df['cluster_total_well_yield'].values[0], 
                                cluster_df['cluster_avg_TDS_kgm3'].values[0], 
                                100, # Arbitrarily set because the well will be removed from cost
                                cluster_df['elec_price'].values[0])
        
        pipe_brine_lcow = pipe_costing(brine_vol,dist_to_dwi,elev_gain = 1e-5, electricity_rate=cluster_df['elec_price'].values[0])*(100-recovery)/recovery

        cluster_treatment_lcow_list.append(treatment_lcow)
        cluster_pipe_brine_lcow_list.append(pipe_brine_lcow)
        recovery_list.append(recovery)

    condensed_cluster_bw_df['treatment_lcow']  = cluster_treatment_lcow_list
    condensed_cluster_bw_df['cluster_pipe_brine_lcow'] = cluster_pipe_brine_lcow_list
    condensed_cluster_bw_df['recovery'] = recovery_list

    # Correct LCOW calculation basis
    treated_vol = cluster_df['cluster_total_well_yield'].values[0]*recovery/100

    condensed_cluster_bw_df['cluster_avg_well_field_lcow'] = condensed_cluster_bw_df['cluster_avg_well_field_lcow']/(recovery/100)
    condensed_cluster_bw_df['cluster_avg_pipe_well_subcluster_lcow'] = condensed_cluster_bw_df['cluster_avg_pipe_well_subcluster_lcow']/(recovery/100)
    condensed_cluster_bw_df['cluster_avg_pipe_subcluster_cluster_lcow'] = condensed_cluster_bw_df['cluster_avg_pipe_subcluster_cluster_lcow']/(recovery/100)

    # Total LCOW of the cluster
    condensed_cluster_bw_df['cluster_lcow'] = condensed_cluster_bw_df['cluster_avg_well_field_lcow'] \
                                                + condensed_cluster_bw_df['cluster_avg_pipe_well_subcluster_lcow'] \
                                                + condensed_cluster_bw_df['cluster_avg_pipe_subcluster_cluster_lcow'] \
                                                + condensed_cluster_bw_df['treatment_lcow'] \
                                                + condensed_cluster_bw_df['cluster_pipe_brine_lcow']


    return condensed_cluster_bw_df

# Function to plot map of wells and cluster centroids
def plot_well_centroid(bw_cluster_kmeans):
    state_df = bw_cluster_kmeans

    us_counties = gp.read_file(r'\Users\mhardika\Documents\AMO\GeoToolAll_Methods\GeoData\US_County_Boundaries\US_CountyBndrys.shp')
    us_counties = us_counties.to_crs("EPSG:4326")
    us_states = gp.read_file(r'C:\Users\mhardika\Documents\AMO\2050\analysis_files\tl_rd22_us_state\tl_rd22_us_state.shp')
    us_states = us_states.to_crs("EPSG:4326")
    
    state_code_df =  pd.read_csv(r'/Users/mhardika/Documents/watertap3/WaterTAP3/watertap3/watertap3/data/state_geocode.csv',index_col='abbv')
    abbv_code = state_df.state_alpha.map(state_code_df.state_id)

    state_code = f'{abbv_code[0]:02d}'
    # state_code = '09'
    state_geo = us_counties.loc[us_counties['STATEFP']==state_code]
    state_border = us_states.loc[us_states['STATEFP']==state_code]

    centers_long = []
    centers_lat = []

    for idx , row in state_df.iterrows():
        centers_long.append(row['centroid_long'])
        centers_lat.append(row['centroid_lat'])

    fig, (ax,ax0) = plt.subplots(1,2, figsize = (14,6))

    # ax.axes.set_facecolor(color='white')
    # ax0.axes.set_facecolor(color='white')

    cm = plt.cm.get_cmap('tab20b')

    state_border.plot(ax=ax,facecolor ='none',edgecolor ='black')
    state_geo.plot(ax=ax,facecolor ='none',edgecolor ='gray',alpha = 0.5)

    state_border.plot(ax=ax0,facecolor ='none',edgecolor ='black')
    state_geo.plot(ax=ax0,facecolor ='none',edgecolor ='gray',alpha = 0.5)

    sc = ax0.scatter(x = state_df['Longitude'], y = state_df['Latitude'], s=50, 
                c=state_df['cluster_id'].values, cmap=cm, edgecolor ='black',alpha = 0.8)

    for cluster in state_df['cluster_id'].unique():
        df_0 = state_df[(state_df['cluster_id']==cluster)]

        ax.scatter(x = df_0['Longitude'], y = df_0['Latitude'], s=80, 
                c=df_0['subcluster_id'].values,cmap=cm)    

        for i, txt in enumerate(df_0['subcluster_id'].unique()):
            ax.annotate(txt, 
                        (df_0[df_0['subcluster_id']==i]['subcluster_long'].values[0]+0.2, 
                        df_0[df_0['subcluster_id']==i]['subcluster_lat'].values[0]),fontsize =8)
        
        # Plot subcluster centroid
        ax.scatter(df_0['subcluster_long'].values, df_0['subcluster_lat'].values, 
                c=df_0['subcluster_id'].values,marker='*',s=120,cmap='tab20b',linewidths=0.5,edgecolor ='black')
        
        # Plot the centroid in each cluster
        ax.scatter(df_0['centroid_long'], df_0['centroid_lat'], c='black', s=120, marker = '^',edgecolor ='black')
        ax0.scatter(df_0['centroid_long'], df_0['centroid_lat'], c='black', s=120, marker = '^',edgecolor ='black')

    # cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8]) 
    if len(state_df['cluster_id'].unique())%2 == 0:
        ticks_var = len(state_df['cluster_id'].unique()) 
    else:
        ticks_var = len(state_df['cluster_id'].unique()) + 1

    cbar = plt.colorbar(sc,ax=ax0,ticks = plt.MaxNLocator(ticks_var))
    cbar.ax.set_title('Cluster ID',fontsize = 14)
    cbar.ax.tick_params(labelsize=14)

    # plt.tick_params(left = False, right = False , labelleft = False ,
    #                 labelbottom = False, bottom = False)
    
    return fig

# Function to plot LCOW supply curve
def plot_supply_curve(bw_cluster_kmeans,condensed_cluster_bw_df,state_alpha):

    # Plot supply curve with and without clustering
    fig, (ax,ax1) = plt.subplots(1,2, figsize = (12,6))

    # Without clustering
    bw_df_temp = bw_df[bw_df['state_alpha']== state_alpha].copy()
    # Calculate total cost
    # Treatment + brine disposal cost included in lcow 
    bw_df_temp['lcow_t'] = bw_df_temp['lcow'] + bw_df_temp['well_field_lcow']/(bw_df_temp['recovery']/100)

    bw_df_temp_sorted = bw_df_temp.sort_values('lcow_t')
    colors = plt.cm.get_cmap('tab20b')
    cluster_id_list = []
    for well in bw_df_temp_sorted['unique_site_ID'].unique():
        cluster_id_list.append(bw_cluster_kmeans[bw_cluster_kmeans['unique_site_ID']==well]['cluster_id'].values[0])

    # Calculating LCOW as a function of well yield without clustering
    flow = bw_df_temp_sorted['well_yield']*bw_df_temp_sorted['recovery']/100
    cum_flow_1 = flow.cumsum()

    # Treatment + brine disposal cost included in lcow 
    lcow_t = bw_df_temp_sorted['lcow_t'] 
    cost = lcow_t*flow
    cum_cost_1 = cost.cumsum()

    avg_lcow_1 = np.divide(cum_cost_1, cum_flow_1, out=np.zeros_like(cum_cost_1), where=cum_flow_1!=0) 
    
    ax.scatter(cum_flow_1,avg_lcow_1,c = cluster_id_list,cmap=colors)
    ax.set_title('Without Clustering')

    # With clustering
    condensed_cluster_bw_df_sort = condensed_cluster_bw_df.sort_values(by=['cluster_lcow'])

    flow = condensed_cluster_bw_df_sort['cluster_total_well_yield']*condensed_cluster_bw_df_sort['recovery']/100
    cum_flow = flow.cumsum()

    cost = condensed_cluster_bw_df_sort['cluster_lcow']*flow
    cum_cost = cost.cumsum()

    avg_lcow = np.divide(cum_cost, cum_flow) 

    ax1.set_title('With Clustering')
    sc = ax1.scatter(cum_flow,avg_lcow,c=condensed_cluster_bw_df_sort['cluster_id'].values,cmap=colors)

    if len(condensed_cluster_bw_df_sort['cluster_id'].unique())%2 == 0:
        ticks_var = len(condensed_cluster_bw_df_sort['cluster_id'].unique())
    else:
        ticks_var = len(condensed_cluster_bw_df_sort['cluster_id'].unique()) + 1 

    cbar = plt.colorbar(sc,ax=ax1,ticks = plt.MaxNLocator(ticks_var))

    for a in (ax,ax1):
        a.set_xlim([0,max(cum_flow_1)+0.5])
        a.set_ylim([0,max(avg_lcow_1)+0.2])
        a.set_ylabel('Average LCOW ($/m3 water treated)')
        a.set_xlabel('Cumulative flow (m3/s)')

    return fig

if __name__=='__main__':
    state_alpha = 'WI'
    print(state_alpha)
    bw_cluster_kmeans = create_cluster_id_df(state_alpha)
    bw_cluster_kmeans = create_bw_cluster_subcluster_df(state_alpha, bw_cluster_kmeans)
    condensed_subcluster_bw_df = condense_subcluster_table(bw_cluster_kmeans)
    condensed_cluster_bw_df = condense_cluster_table(condensed_subcluster_bw_df)
    condensed_cluster_bw_df = calc_cluster_lcow(condensed_cluster_bw_df,dist_to_dwi = 16.0934)
    print(condensed_cluster_bw_df.head(5))


