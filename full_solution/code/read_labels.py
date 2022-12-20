# This will write out the labels for each field
import glob
import json
import os
import rasterio

import numpy as np
import pandas as pd

from tqdm import tqdm

# Set parameters
input_dir              = "../input/ref_agrifieldnet_competition_v1"
cat_fname              = os.path.join ( input_dir, "catalog.json" )
labels_fname           = os.path.join ( "data", "labels.csv" )

# From data loading sample posted
collection_name        = "ref_agrifieldnet_competition_v1"
source_collection      = f"{collection_name}_source"
train_label_collection = f"{collection_name}_labels_train"
test_label_collection  = f"{collection_name}_labels_test"

train_paths            = os.listdir ( os.path.join ( input_dir, train_label_collection ) )
train_ids              = [ i.split ( "_" ) [ - 1 ] for i in train_paths if "labels_train" in i ]

field_paths            = [ os.path.join ( input_dir, train_label_collection, train_label_collection + "_" + i, "field_ids.tif"     ) for i in train_ids ]
label_paths            = [ os.path.join ( input_dir, train_label_collection, train_label_collection + "_" + i, "raster_labels.tif" ) for i in train_ids ]
source_paths           = [ os.path.join ( input_dir, source_collection,      source_collection      + "_" + i                      ) for i in train_ids ]

# Create a dictionary of field_id -> crop type
labels                 = dict ()

for i, j in tqdm ( zip ( field_paths, label_paths ) ) :

	with rasterio.open ( i ) as src :

		field_data = src.read () [ 0 ]

	with rasterio.open ( j ) as src :

		label_data = src.read () [ 0 ]

	for k in np.unique ( field_data ) :

		if k > 0 :

			label = np.unique ( label_data [ field_data == k ] )

			if len ( label ) > 1 :

				print ( "Multiple labels for { k } : { label }" )

			if k in labels :

				if labels [ k ] != label [ 0 ] :

					print ( "Different labels for { k } : { labels [ k ] } and { label [ 0 ] }" )

			else :

				labels [ k ] = label [ 0 ]

# Now save the labels
pd.DataFrame ( labels.items (), columns = [ "field_id", "label" ] ).to_csv ( labels_fname, index = False )

# Done
print ( "All done"    )
print ( "MG Ferreira" )
print ( "2022"        )
