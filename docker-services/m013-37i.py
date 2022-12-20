# Same as m013-37 but
# used for inference
import os
import pickle

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.cluster import AffinityPropagation, KMeans, OPTICS
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Parameters
print ( "Setting configuration" )

input_dir      = "/opt/radiant/docker-solution/data/input" # "infer/input"
output_dir     = "/opt/radiant/docker-solution/data/output" # "infer/input"
workspace_dir  = "/opt/radiant/docker-solution/workspace"
models_dir     = "/opt/radiant/docker-solution/models/main"

n_folds        = 1 #    10

model_basename = "m013-37"

tiles_fname    = os.path.join ( workspace_dir, "d013i-tiles.csv"                       )
fields_fname   = os.path.join ( workspace_dir, "d013i-fields+tiles.csv"                )
oth_fname      = os.path.join ( workspace_dir, "d013i-oth.csv"                         )
hist_fname     = os.path.join ( workspace_dir, "d013i-hist.csv"                        )
nearby_fname   = os.path.join ( workspace_dir, "dict_predictions_neighbours.pickle"    )
sub_fname      = os.path.join ( output_dir, "sub-" + model_basename + "i.csv"       )
subw_fname     = os.path.join ( output_dir, "sub-" + model_basename + "i-w.csv"     )


bands          = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12" ]
derived_bands1 = [ "NDVI", "BNDVI", "GNDVI", "GBNDVI", "GRNDVI", "RBNDVI", "GARI", "NBR", "NDMI", "NPCRI", "AVI", "BSI" ]
derived_bands2 = [ "SI", "BRI", "MSAVI", "NDSI", "NDRE", "NGRDI", "RDVI", "SIPI", "GCI", "GBNDV2", "GRNDV2" ]
derived_bands3 = [ "REIP", "SLAVI", "TCARI", "TCI", "WDRVI", "ARI", "MYVI", "FE2", "CVI", "VARIG" ]
derived_bands4 = [ "EVI", "MI", "SAVI" ]
derived_bands  = derived_bands1 + derived_bands2 + derived_bands3 + derived_bands4
selected_bands = bands + derived_bands
cluster_bands  = bands
mask_labels    = [ "F0", "F+3", "F+20", "F+40" ]
hist_numbins   = [ 3, 5, 7 ]
value_cols     = [ "min", "max", "mean", "median", "mode", "mean_tile", "median_tile", "mode_tile", "std", "skew", "kurt" ]

for i in hist_numbins :

	value_cols.append ( f"lowerlimit_{ i }" )
	value_cols.append ( f"binsize_{ i }" )
	value_cols.append ( f"extrapoints_{ i }" )

	for j in range ( i ) :

		value_cols.append ( f"bin_{ j }_{ i }" )

value_calc_cols = value_cols + [ "diff", "diffm", "range" ]

# Transform -> mask -> features
used_features  = { "none"     : { "F0"   : [ "mean", "mean_tile", "median", "median_tile", "mode", "mode_tile", "binsize_5", "bin_0_3", "bin_1_3", "bin_2_3" ],
				  "F+3"  : [ "mean", "mean_tile", "median", "median_tile", "mode", "mode_tile", "binsize_5" ],
				  "F+20" : [ "mean", "median", "mode", "binsize_7", "bin_0_7", "bin_1_7", "bin_2_7", "bin_3_7", "bin_4_7", "bin_5_7", "bin_6_7" ],
				  "F+40" : [ "mean", "median", "mode", "binsize_7", "bin_0_7", "bin_1_7", "bin_2_7", "bin_3_7", "bin_4_7", "bin_5_7", "bin_6_7" ] } }

# Cluster features
small_features = { "none"     : { "F0"   : [ "mean", "median", "mode", "binsize_3" ],
 				  "F+3"  : [ "mean", "median", "mode", "binsize_3" ],
				  "F+20" : [ "mean", "median", "mode", "binsize_5", "bin_0_3", "bin_1_3", "bin_2_3" ] } }
medm_features  = { "none"     : { "F0"   : [ "mean", "median", "mode", "binsize_5" ],
 				  "F+3"  : [ "mean", "median", "mode", "binsize_5" ],
				  "F+20" : [ "mean", "median", "mode", "binsize_7", "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5" ] } }
large_features = { "none"     : { "F0"   : [ "mean", "median", "mode", "binsize_5", "bin_0_3", "bin_1_3", "bin_2_3" ],
 				  "F+3"  : [ "mean", "median", "mode", "binsize_5", "bin_0_3", "bin_1_3", "bin_2_3" ],
				  "F+20" : [ "mean", "median", "mode", "binsize_7", "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5" ] } }
huge_features  = { "none"     : { "F0"   : [ "mean", "median", "mode", "binsize_7", "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5" ],
				  "F+3"  : [ "mean", "median", "mode", "binsize_7", "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5" ],
				  "F+20" : [ "mean", "median", "mode", "binsize_7", "bin_0_7", "bin_1_7", "bin_2_7", "bin_3_7", "bin_4_7", "bin_5_7", "bin_6_7" ] } }

group_col      = "field_id"
criterion      = "log_loss"
crop_dict      =  {  1 : "Wheat",
		     2 : "Mustard",
		     3 : "Lentil",
		     4 : "No Crop",
		     6 : "Sugarcane",
		     8 : "Garlic",
		    15 : "Potato",
		     5 : "Green pea",
		    16 : "Bersem",
		    14 : "Coriander",
		    13 : "Gram",
		     9 : "Maize",
		    36 : "Rice" }
crop_distro   =  {   1 : 2031 / 21476,
		     2 : 1980 / 21476,
		     3 :  309 / 21476,
		     4 : 6564 / 21476,
		     5 :  115 / 21476,
		     6 :  978 / 21476,
		     8 :  384 / 21476,
		     9 : 2637 / 21476,
		    13 :  767 / 21476,
		    14 :  196 / 21476,
		    15 :  615 / 21476,
		    16 :  256 / 21476,
		    36 : 4644 / 21476 }

# Same order as in sample submission
labels_text   = [ "Wheat", "Mustard", "Lentil", "No Crop", "Green pea", "Sugarcane", "Garlic", "Maize", "Gram", "Coriander", "Potato", "Bersem", "Rice" ]

labels_known  = np.arange ( len ( labels_text ), dtype = int )

crop_dict_inv = { j : i for i, j in crop_dict.items () }

my_labels     = { i : crop_dict_inv [ j ] for i, j in enumerate ( labels_text ) }
my_labels_inv = { j : i for i, j in my_labels.items () }
n_labels      = len ( labels_text )

crop_distro   = { i : crop_distro [ crop_dict_inv [ j ] ] for i, j in enumerate ( labels_text ) }

# Read and prepare inputs
print ( "Loading data" )

hist_join_cols = [ "tile_id", "field_id", "band", "transformation", "mask" ]

tiles_data     = pd.read_csv ( tiles_fname  )
fields_data    = pd.read_csv ( fields_fname )
oth_data       = pd.read_csv ( oth_fname    ).set_index ( [ "tile_id", "field_id" ] )
hist_data      = pd.read_csv ( hist_fname   ).set_index ( hist_join_cols )

# Join
print ( "Joining" )

join_cols      = [ "tile_id", "band", "transformation" ]
fields_data    = fields_data.join ( tiles_data.set_index ( join_cols ), on = join_cols, rsuffix = "_tile" ).join ( hist_data, on = hist_join_cols )

# Only keep certain transformations and masks
print ( "Filtering" )

features_dict  = dict ()

for d, D in zip ( [ used_features, small_features, medm_features, large_features, huge_features ], [ "data", "small", "medium", "large", "huge" ] ) :

	print ( f"\t{ D }" )

	data         = None
	cols         = []
	cluster_cols = []

	for i in d :

		print ( f"\t\t{ i }" )

		# Filter
		i_data = fields_data.loc [ fields_data [ "transformation" ] == i ]
		i_data = i_data     .loc [ i_data      [ "mask"           ].isin ( used_features [ i ] ) ]

		# Pivot long to wide
		i_data = i_data.pivot ( index = [ "tile_id", "field_id" ], columns = [ "band", "mask" ], values = value_cols )

		if data is None :

			data = pd.DataFrame ( index = i_data.index )

		# Calculate and add to data
		for j in selected_bands :

			for k in d [ i ] :

				# Some columns
				col_n      = ( "n"          , j, k )
				col_range  = ( "range"      , j, k )
				col_min    = ( "min"        , j, k )
				col_max    = ( "max"        , j, k )

				col_diff   = ( "diff"       , j, k )
				col_diffm  = ( "diffm"      , j, k )
				col_diffm0 = ( "diffm0"     , j, k )
				col_mean   = ( "mean"       , j, k )
				col_median = ( "median"     , j, k )
				col_mode   = ( "mode"       , j, k )
				col_tile   = ( "mean_tile"  , j, k )
				col_tilem  = ( "median_tile", j, k )
				col_tilem0 = ( "mode_tile"  , j, k )

				# Calculate
				i_data [ col_range  ] = i_data [ col_max    ] - i_data [ col_min    ]
				i_data [ col_diff   ] = i_data [ col_mean   ] - i_data [ col_tile   ]
				i_data [ col_diffm  ] = i_data [ col_median ] - i_data [ col_tilem  ]
				i_data [ col_diffm0 ] = i_data [ col_mode   ] - i_data [ col_tilem0 ]

				# Add selected columns
				for l in value_calc_cols :

					if l in d [ i ][ k ] :

						col          = f"{ i }-{ j }-{ k }-{ l }"
						data [ col ] = i_data [ ( l, j, k ) ]

						cols.append ( col )

						if j in cluster_bands :

							cluster_cols.append ( col )

						# Defrag
						data         = data.copy ()

				# Defrag
				i_data = i_data.copy ()

	print ( "\t\tfilling missing values" )
	features_dict [ D ] = { "data" : data.fillna ( 0 ), "cols" : cols, "cluster_cols" : cluster_cols }

for i in features_dict :

	print ( f"\t{ i } { features_dict [ i ][ 'data' ].shape }" )

data           = features_dict [ "data"  ][ "data" ]
x_cols         = features_dict [ "data"  ][ "cols" ]

# Add other data
print ( "Add other data" )

data           = data.join ( oth_data, how = "left" )
x_cols.extend ( list ( oth_data.columns ) )

print ( f"\t{ data.shape }" )

# Add moto main and nearby neighbour classification
print ( "Add main and nearby neighbour classification" )

class_indices  = { "AnnualCrop"           : 0,
		   "Forest"               : 1,
		   "HerbaceousVegetation" : 2,
		   "Highway"              : 3,
		   "Industrial"           : 4,
		   "Pasture"              : 5,
		   "PermanentCrop"        : 6,
		   "Residential"          : 7,
		   "River"                : 8,
		   "SeaLake"              : 9 }
class_names    = [ "Annual", "Forest", "Vegetation", "Highway", "Industrial", "Pasture", "Permanent", "Residential", "River", "Sea" ]
class_dict     = dict ()

for i in [ "main", "north", "east", "west", "south" ] :

	class_dict [ i ] = [ j + " " + i for j in class_names ]
	x_cols.extend ( class_dict [ i ] )

with open ( nearby_fname, "rb" ) as i :

	dict_predictions_nearby = pickle.load ( i )

nearby_fnames  = dict_predictions_nearby [ "filenames"   ]
nearby_preds   = dict_predictions_nearby [ "predictions" ]

for nearby_fname, nearby_pred in zip ( nearby_fnames, nearby_preds ) :

	j = nearby_fname.split ( "/" ) [ - 1 ].split ( "_" )
	k = j [   0 ]
	l = j [   1 ]
	m = j [ - 1 ].split ( "." ) [ 0 ]

	if ( k, int ( l ) ) in data.index :

		print ( f"\tadding tile { k }, field { l } classification towards { m }" )
		data.loc [ ( k, int ( l ) ), class_dict [ m ] ] = nearby_pred

print ( f"\t{ data.shape }" )

# Calculate label distribution per tile
print ( "Field distribution per tile" )
print ( "\tfor inference, assume same as train" )

for i in range ( n_labels ) :

	x_col = "l_" + str ( i )
	data.loc [ :, x_col ] = crop_distro [ i ]
	x_cols.append ( x_col )

# Clusters
print ( "Clusters" )

cluster_labels = [ "affinity_propagation", "kmeans_40", "kmeans_20", "kmeans_10", "optics" ]
cluster_index  = []

small_rows     =   data [ "pixels" ] <  10
medium_rows    = ( data [ "pixels" ] >= 10 ) & ( data [ "pixels" ] <  18 )
large_rows     = ( data [ "pixels" ] >= 18 ) & ( data [ "pixels" ] <  50 )
huge_rows      =   data [ "pixels" ] >= 50

features_dict [ "small"  ][ "rows" ] =  small_rows
features_dict [ "medium" ][ "rows" ] = medium_rows
features_dict [ "large"  ][ "rows" ] =  large_rows
features_dict [ "huge"   ][ "rows" ] =   huge_rows

for j in cluster_labels :

	print ( f"\t{ j }" )

	# Two variations
	j0                 = j + "_0"
	j1                 = j + "_1"

	data.loc [ :, j0 ] = 0
	data.loc [ :, j1 ] = 0

	offset_0           = 0
	offset_1           = 0

	for k in [ "small", "medium", "large", "huge" ] :

		print ( f"\t\t{ k } selected bands" )

		print ( "\t\tloading" )
		i     = load ( os.path.join ( models_dir, f"{ model_basename }-{ j }-{ k }-selected.joblib" ) )

		if features_dict [ k ][ "rows" ].sum () > 0 :

			if j == "optics" :

				data.loc [ features_dict [ k ][ "rows" ], j0 ] = - 1

			else :

				y     = i.predict ( features_dict [ k ][ "data" ].loc [ features_dict [ k ][ "rows" ], features_dict [ k ][ "cols" ] ] )
				data.loc [ data.loc [ features_dict [ k ][ "rows" ] ].loc [ y <  0 ].index, j0 ] = - 1
				data.loc [ data.loc [ features_dict [ k ][ "rows" ] ].loc [ y >= 0 ].index, j0 ] = offset_0 + y [ y >= 0 ]

		y_min = min ( i.labels_ )
		y_max = max ( i.labels_ )

		print ( f"\t\t{ y_min } - { y_max }" )

		offset_0 += 1 + y_max

		print ( f"\t\t{ k } cluster bands" )

		print ( "\t\tloading" )
		i     = load ( os.path.join ( models_dir, f"{ model_basename }-{ j }-{ k }-cluster.joblib" ) )

		if features_dict [ k ][ "rows" ].sum () > 0 :

			if j == "optics" :

				data.loc [ features_dict [ k ][ "rows" ], j1 ] = - 1

			else :

				y     = i.predict ( features_dict [ k ][ "data" ].loc [ features_dict [ k ][ "rows" ], features_dict [ k ][ "cluster_cols" ] ] )
				data.loc [ data.loc [ features_dict [ k ][ "rows" ] ].loc [ y <  0 ].index, j1 ] = - 1
				data.loc [ data.loc [ features_dict [ k ][ "rows" ] ].loc [ y >= 0 ].index, j1 ] = offset_1 + y [ y >= 0 ]

		y_min = min ( i.labels_ )
		y_max = max ( i.labels_ )

		print ( f"\t\t{ y_min } - { y_max }" )

		offset_1 += 1 + y_max

	cluster_index.append ( len ( x_cols ) )
	x_cols.append ( j0 )
	cluster_index.append ( len ( x_cols ) )
	x_cols.append ( j1 )

# Predicting
print ( "Predicting" )

pred_test      = np.zeros ( [ data.shape [ 0 ], n_labels ], dtype = float )
test           = data.loc [ :, x_cols ].to_numpy ( dtype = float )

print ( "pred_test" )
print ( pred_test.shape )
print ( "test" )
print ( test.shape )

for i in range ( n_folds ) :

	print ( f"\t{ i + 1 } / { n_folds }" )

	print ( "\t\t\tloading" )
	model_fname = os.path.join ( models_dir, f"{ model_basename }-{ i }.joblib" )
	model = load ( model_fname )

	print ( "\t\tpredicting" )
	pred_test += model.predict_proba ( test ) / n_folds

# Make submission file
print ( "Submitting" )

sub = pd.DataFrame ( columns = [ "field_id" ] + labels_text )
sub.loc [ :, "field_id"  ] = data.index.get_level_values ( "field_id" )
sub.loc [ :, labels_text ] = pred_test
sub = sub.groupby ( by = "field_id" ).mean ()
sub.to_csv ( sub_fname )

# Submission with weight based on field size
pix  = data [ "pixels" ].to_numpy ( dtype = float )
subw = pd.DataFrame ( columns = [ "field_id", "pixels" ] + labels_text )
subw.loc [ :, "field_id"  ] = data.index.get_level_values ( "field_id" )
subw.loc [ :, "pixels"    ] = pix
subw.loc [ :, labels_text ] = pred_test

for i in labels_text :

	subw [ i ] = subw [ i ] * subw [ "pixels" ]

subw = subw.groupby ( by = "field_id" ).sum ()

for i in labels_text :

	subw [ i ] = subw [ i ] / subw [ "pixels" ]

subw = subw.drop ( "pixels", axis = 1 )
subw.to_csv ( subw_fname )

# Done
print ( "All done"    )
print ( "MG Ferreira" )
print ( "2022"        )
