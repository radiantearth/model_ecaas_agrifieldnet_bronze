import os
import pickle

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC

from joblib import dump, load

from sklearn.cluster import AffinityPropagation, KMeans, OPTICS
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Parameters
print ( "Setting configuration" )

model_basename = "m013-35"

tiles_fname    = os.path.join ( "data", "d013-tiles.csv"                       )
fields_fname   = os.path.join ( "data", "d013-fields+tiles.csv"                )
oth_fname      = os.path.join ( "data", "d013-oth.csv"                         )
hist_fname     = os.path.join ( "data", "d013-hist.csv"                        )
nearby_fname   = os.path.join ( "data", "dict_predictions_neighbours.pickle"   )
labels_fname   = os.path.join ( "data", "labels.csv"                           )
data_fname     = os.path.join ( "data", model_basename + "-data.csv"           )
small_fname    = os.path.join ( "data", model_basename + "-small.csv"          )
medium_fname   = os.path.join ( "data", model_basename + "-medium.csv"         )
large_fname    = os.path.join ( "data", model_basename + "-large.csv"          )
huge_fname     = os.path.join ( "data", model_basename + "-huge.csv"           )
clusters_fname = os.path.join ( "data", model_basename + "-clusters.csv"       )
oof_fname      = os.path.join ( "data", "oof-" + model_basename + ".csv"       )
ooft_fname     = os.path.join ( "data", "oof-" + model_basename + "-tiles.csv" )
sub_fname      = os.path.join ( "subs", "sub-" + model_basename + ".csv"       )
subw_fname     = os.path.join ( "subs", "sub-" + model_basename + "-w.csv"     )
imp_fname      = os.path.join ( "data", model_basename + "-imp.csv"            )

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
used_features  = { "none"     : { "F0"   : [ "median", "median_tile", "mean", "mean_tile", "mode", "mode_tile", "binsize_5" ],
				  "F+3"  : [ "median", "median_tile", "mean", "mean_tile", "mode", "mode_tile", "binsize_5" ],
				  "F+20" : [ "median", "mean", "mode", "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5", "binsize_7" ],
				  "F+40" : [ "median", "mean", "mode", "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5", "binsize_7" ] } }

# Cluster features
small_features = { "none"     : { "F0"   : [ "mean", "median", "mode" ],
 				  "F+3"  : [ "mean", "median", "mode" ],
				  "F+20" : [ "bin_0_3", "bin_1_3", "bin_2_3", "mean", "median", "mode" ] } }
medm_features  = { "none"     : { "F0"   : [ "mean", "median", "mode", "binsize_3" ],
 				  "F+3"  : [ "mean", "median", "mode", "binsize_5" ],
				  "F+20" : [ "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5", "mean", "median", "mode", "binsize_7" ] } }
large_features = { "none"     : { "F0"   : [ "bin_0_3", "bin_1_3", "bin_2_3", "mean", "median", "mode", "binsize_3" ],
 				  "F+3"  : [ "bin_0_3", "bin_1_3", "bin_2_3", "mean", "median", "mode", "binsize_5" ],
				  "F+20" : [ "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5", "mean", "median", "mode", "binsize_7" ] } }
huge_features  = { "none"     : { "F0"   : [ "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5", "mean", "median", "mode", "binsize_3" ],
				  "F+3"  : [ "bin_0_5", "bin_1_5", "bin_2_5", "bin_3_5", "bin_4_5", "mean", "median", "mode", "binsize_5" ],
				  "F+20" : [ "bin_0_7", "bin_1_7", "bin_2_7", "bin_3_7", "bin_4_7", "bin_5_7", "bin_6_7", "mean", "median", "mode", "binsize_7" ] } }

group_col      = "field_id"
n_folds        =     10
rs             = 222213
n_estimators   =   2500
criterion      = "log_loss"
max_depth      =    250

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

# Same order as in sample submission
labels_text   = [ "Wheat", "Mustard", "Lentil", "No Crop", "Green pea", "Sugarcane", "Garlic", "Maize", "Gram", "Coriander", "Potato", "Bersem", "Rice" ]

labels_known  = np.arange ( len ( labels_text ), dtype = int )

crop_dict_inv = { j : i for i, j in crop_dict.items () }

my_labels     = { i : crop_dict_inv [ j ] for i, j in enumerate ( labels_text ) }
my_labels_inv = { j : i for i, j in my_labels.items () }

# Read and prepare inputs
print ( "Loading data" )

hist_join_cols = [ "tile_id", "field_id", "band", "transformation", "mask" ]

tiles_data     = pd.read_csv ( tiles_fname  )
fields_data    = pd.read_csv ( fields_fname )
oth_data       = pd.read_csv ( oth_fname    ).set_index ( [ "tile_id", "field_id" ] )
hist_data      = pd.read_csv ( hist_fname   ).set_index ( hist_join_cols )
labels_data    = pd.read_csv ( labels_fname, dtype = int )

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

	data.loc [ ( k, int ( l ) ), class_dict [ m ] ] = nearby_pred

print ( f"\t{ data.shape }" )

# Add labels
print ( "Add labels" )

data           = data.join ( labels_data.set_index ( "field_id" ), how = "left" )

# Fix labes
print ( "Fix labels" )

data [ "label" ] = data [ "label" ].fillna ( - 1 ).astype ( int ).apply ( lambda x : my_labels_inv [ x ] if x >= 0 else - 1 )
n_labels       = data [ "label" ].max () + 1
print ( f"\t{ n_labels } labels" )
print ( f"\t{ data.shape }" )

# Calculate label distribution per tile
print ( "Field distribution" )

for i in range ( n_labels ) :

	x_col = "l_" + str ( i )
	data.loc [ :, x_col ] = 0.0
	x_cols.append ( x_col )

for i in data.index.get_level_values ( "tile_id" ).unique () :

	print ( f"\t{ i }" )
	i_data  = data.loc [ data.index.get_level_values ( "tile_id" ) == i ]

	for j in i_data.index.get_level_values ( "field_id" ).unique () :

		j_data = i_data.loc [ i_data.index.get_level_values ( "field_id" ) != j ]
		j_data = j_data.loc [ j_data [ "label" ] >= 0 ]
		n_data = j_data.shape [ 0 ]

		if n_data > 0 :

			for k in range ( n_labels ) :

				p = np.sum ( j_data [ "label" ] == k ) / n_data
				data.loc [ ( i, j ), "l_" + str ( k ) ] = p

	data = data.copy ()

# Some calculations
print ( "Calculating" )

y_col          = "label"
train_rows     = data [ y_col ] >=   0
test_rows      = data [ y_col ] == - 1
groups         = data.loc [ train_rows ].index.get_level_values ( group_col )

# Write to output file
print ( "Storing" )

for i, j, k in zip ( [ data, None, None, None, None ], [ data_fname, small_fname, medium_fname, large_fname, huge_fname ], [ "data", "small", "medium", "large", "huge" ] ) :

	print ( f"\t{ k }" )

	if i is None :

		d = features_dict [ k ][ "data" ]

	else :

		d = i

	print ( d )
	d.to_csv ( j )

# Create folds
print ( "Making folds" )

# Now we can model
train_indices  = []
valid_indices  = []

kf             = StratifiedKFold ( n_splits = n_folds )

for i, j in kf.split ( np.arange ( np.sum ( train_rows ) ), data.loc [ data [ y_col ] >= 0, "label" ].to_numpy ( dtype = int ) ) :

	train_indices.append ( i )
	valid_indices.append ( j )

# Clusters
print ( "Clusters" )

cluster_labels = [ "affinity_propagation", "kmeans_40", "kmeans_20", "kmeans_10", "optics" ]
cluster_index  = []

if os.path.isfile ( clusters_fname ) :

	print ( "\tloading" )
	clusters_df = pd.read_csv ( clusters_fname ).set_index ( [ "tile_id", "field_id" ] )

	for j in clusters_df.columns :

		cluster_index.append ( len ( x_cols ) )
		x_cols.append ( j )
		data [ j ] = clusters_df [ j ]

else :

	print ( "\tfitting" )

	small_rows     =   data [ "pixels" ] <  10
	medium_rows    = ( data [ "pixels" ] >= 10 ) & ( data [ "pixels" ] <  18 )
	large_rows     = ( data [ "pixels" ] >= 18 ) & ( data [ "pixels" ] <  50 )
	huge_rows      =   data [ "pixels" ] >= 50

	features_dict [ "small"  ][ "rows" ] =  small_rows
	features_dict [ "medium" ][ "rows" ] = medium_rows
	features_dict [ "large"  ][ "rows" ] =  large_rows
	features_dict [ "huge"   ][ "rows" ] =   huge_rows

	for i, j in zip ( [ AffinityPropagation ( max_iter = 1000, random_state = rs ),
			    KMeans ( n_clusters = 40, random_state = rs + 41 ),
			    KMeans ( n_clusters = 20, random_state = rs + 19 ),
			    KMeans ( n_clusters = 10, random_state = rs + 11 ),
			    OPTICS ( xi = 0.001 ) ],
			  cluster_labels ) :

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

			y     = i.fit_predict ( features_dict [ k ][ "data" ].loc [ features_dict [ k ][ "rows" ], features_dict [ k ][ "cols" ] ] )
			y_min = min ( y )
			y_max = max ( y )

			print ( f"\t\t{ y_min } - { y_max }" )

			data.loc [ data.loc [ features_dict [ k ][ "rows" ] ].loc [ y <  0 ].index, j0 ] = - 1
			data.loc [ data.loc [ features_dict [ k ][ "rows" ] ].loc [ y >= 0 ].index, j0 ] = offset_0 + y [ y >= 0 ]

			offset_0 += 1 + y_max

			print ( f"\t\t{ k } cluster bands" )

			y     = i.fit_predict ( features_dict [ k ][ "data" ].loc [ features_dict [ k ][ "rows" ], features_dict [ k ][ "cluster_cols" ] ] )
			y_min = min ( y )
			y_max = max ( y )

			print ( f"\t\t{ y_min } - { y_max }" )

			data.loc [ data.loc [ features_dict [ k ][ "rows" ] ].loc [ y <  0 ].index, j1 ] = - 1
			data.loc [ data.loc [ features_dict [ k ][ "rows" ] ].loc [ y >= 0 ].index, j1 ] = offset_1 + y [ y >= 0 ]

			offset_1 += 1 + y_max

		cluster_index.append ( len ( x_cols ) )
		x_cols.append ( j0 )
		cluster_index.append ( len ( x_cols ) )
		x_cols.append ( j1 )

	print ( "\tstoring" )
	data.loc [ :, x_cols ].iloc [ :, cluster_index ].to_csv ( clusters_fname )

# Sample
print ( "Sampling" )

pred_test      = np.zeros ( ( np.sum ( test_rows  ), n_labels ), dtype = float )
pred_valid     = np.zeros ( ( np.sum ( train_rows ), n_labels ), dtype = float )
test           = data.loc [ test_rows, x_cols ].to_numpy ( dtype = float )

print ( "pred_test" )
print ( pred_test.shape )
print ( "test" )
print ( test.shape )

samples        = []
valids         = []

for i in range ( n_folds ) :

	print ( f"\t{ i + 1 } / { n_folds }" )

	train = data.loc [ train_rows ].iloc [ train_indices [ i ] ] [ x_cols ].to_numpy ( dtype = float )
	valid = data.loc [ train_rows ].iloc [ valid_indices [ i ] ] [ x_cols ].to_numpy ( dtype = float )

	labst = data.loc [ train_rows ].iloc [ train_indices [ i ] ] [ y_col  ].to_numpy ( dtype = int )
	labsv = data.loc [ train_rows ].iloc [ valid_indices [ i ] ] [ y_col  ].to_numpy ( dtype = int )

	print ( f"\t\ttrain { train.shape }" )
	print ( f"\t\tvalid { valid.shape }" )
	print ( "\t\t\tsampling" )

	if len ( np.unique ( labst ) ) != len ( labels_text ) :

		print ( "Oops - have { len ( labels_text ) } labels and just { len ( np.unique ( labst ) ) } are in this sample" )
		quit ()

	sampler = SMOTENC ( cluster_index, random_state = rs + 31 * ( i + 61 ) )
	X, y = sampler.fit_resample ( train, labst )
	print ( f"\t\t\tsampled train { X.shape }" )

	samples.append ( ( X, y ) )
	valids.append ( ( valid, labsv ) )

# Train!
print ( "Training" )

imp_values = np.zeros ( len ( x_cols ), dtype = float )

for i in range ( n_folds ) :

	print ( f"\t{ i + 1 } / { n_folds }" )

	model_fname = os.path.join ( "models", f"{ model_basename }-{ i }.joblib" )
	valid, labsv = valids [ i ]

	if os.path.isfile ( model_fname ) :

		print ( "\t\t\tloading" )
		model = load ( model_fname )

	else :

		print ( "\t\t\tfitting" )
		model = RandomForestClassifier ( n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, random_state = rs + 17 * ( i + 19 ) )

		X, y = samples [ i ]
		model.fit ( X, y )

		print ( "\t\t\tsaving" )
		dump ( model, model_fname )

	pred_test += model.predict_proba ( test ) / n_folds
	pred_valid [ valid_indices [ i ] ] = model.predict_proba ( valid )
	imp_values += model.feature_importances_

	l = log_loss ( labsv, pred_valid [ valid_indices [ i ] ], labels = labels_known )
	print ( f"\t\t\t{ l }" )

	# Free up some space
	model = None
	samples [ i ] = None
	valids [ i ] = None

# Evaluate
print ( "Evaluating" )

l = log_loss ( data.loc [ train_rows, y_col ], pred_valid, labels = labels_known )
print ( f"\t{ l }" )

print ( "\fsaving oof predictions" )
oof = pd.DataFrame ( columns = [ "tile_id", "field_id" ] + labels_text )
oof.loc [ :, "tile_id"   ] = data.loc [ train_rows ].index.get_level_values ( "tile_id"  )
oof.loc [ :, "field_id"  ] = data.loc [ train_rows ].index.get_level_values ( "field_id" )
oof.loc [ :, labels_text ] = pred_valid
oof.to_csv ( ooft_fname )
oof = oof.drop ( "tile_id", axis = 1 ).groupby ( by = "field_id" ).mean ()
oof.to_csv ( oof_fname )

# Make submission file
print ( "Submitting" )

sub = pd.DataFrame ( columns = [ "field_id" ] + labels_text )
sub.loc [ :, "field_id"  ] = data.loc [ test_rows ].index.get_level_values ( "field_id" )
sub.loc [ :, labels_text ] = pred_test
sub = sub.groupby ( by = "field_id" ).mean ()
sub.to_csv ( sub_fname )

# Submission with weight based on field size
pix  = data.loc [ test_rows, "pixels" ].to_numpy ( dtype = float )
subw = pd.DataFrame ( columns = [ "field_id", "pixels" ] + labels_text )
subw.loc [ :, "field_id"  ] = data.loc [ test_rows ].index.get_level_values ( "field_id" )
subw.loc [ :, "pixels"    ] = pix
subw.loc [ :, labels_text ] = pred_test

for i in labels_text :

	subw [ i ] = subw [ i ] * subw [ "pixels" ]

subw = subw.groupby ( by = "field_id" ).sum ()

for i in labels_text :

	subw [ i ] = subw [ i ] / subw [ "pixels" ]

subw = subw.drop ( "pixels", axis = 1 )
subw.to_csv ( subw_fname )

# Write out feature importances
pd.Series ( index = x_cols, data = imp_values, name = "feature_importance" ).to_csv ( imp_fname )

# Done
print ( "All done"    )
print ( "MG Ferreira" )
print ( "2022"        )
