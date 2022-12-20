# Extract data
import glob
import json
import os
import rasterio
import statistics

import numpy as np
import pandas as pd

from scipy.stats import cumfreq, kurtosis, skew

from skimage.measure import CircleModel, perimeter
from skimage.morphology import binary_erosion
from skimage.segmentation import expand_labels, find_boundaries, flood

from sklearn.preprocessing import QuantileTransformer

from tqdm import tqdm

# Parameters
base_name      = "013i"
rs             = 202213
qt_distro      = "normal"
qt_trans       = QuantileTransformer ( output_distribution = qt_distro, random_state = rs )
qu_distro      = "uniform"
qu_trans       = QuantileTransformer ( output_distribution = qu_distro, random_state = rs )
calc_stats     = [ np.min, np.max, np.mean, np.median, statistics.mode, np.std, skew, kurtosis ]
stats_labels   = [ "min", "max", "mean", "median", "mode", "std", "skew", "kurt" ]
bands          = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12" ]
derived_bands1 = [ "NDVI", "BNDVI", "GNDVI", "GBNDVI", "GRNDVI", "RBNDVI", "GARI", "NBR", "NDMI", "NPCRI", "AVI", "BSI" ]
derived_bands2 = [ "SI", "BRI", "MSAVI", "NDSI", "NDRE", "NGRDI", "RDVI", "SIPI", "GCI", "GBNDV2", "GRNDV2" ]
derived_bands3 = [ "REIP", "SLAVI", "TCARI", "TCI", "WDRVI", "ARI", "MYVI", "FE2", "CVI", "VARIG" ]
derived_bands4 = [ "EVI", "MI", "SAVI" ]
derived_bands  = derived_bands1 + derived_bands2 + derived_bands3 + derived_bands4
borders        = [ 0, 3, 20, 40 ]
border_labels  = [ "F0", "F+3", "F+20", "F+40" ]
transforms     = [ lambda x : x / 256, lambda x : qt_trans.fit_transform ( x.reshape ( - 1, 1 ) ), lambda x : qu_trans.fit_transform ( x.reshape ( - 1, 1 ) ) ]
trans_labels   = [ "none", "quantile", "uniform" ]
tile_shape     = [ 256, 256 ]
hist_numbins   = [ 3, 5, 7 ]

# Data
input_dir      = "/opt/radiant/docker-solution/data/input" # "infer/input"
workspace_dir  = "/opt/radiant/docker-solution/workspace"

tiles_fname    = os.path.join ( workspace_dir, "d" + base_name + "-tiles.csv"  )
fields_fname   = os.path.join ( workspace_dir, "d" + base_name + "-fields.csv" )
hist_fname     = os.path.join ( workspace_dir, "d" + base_name + "-hist.csv"   )
oth_fname      = os.path.join ( workspace_dir, "d" + base_name + "-oth.csv"    )

# Start reading
all_json_files  = glob.glob ( f"{ input_dir }/**/*.json", recursive = True )

all_tiff_files  = glob.glob ( f"{ input_dir }/**/*.tif", recursive = True )
b01_tiff_files  = [ i for i in all_tiff_files if "B01.tif" in i ]
field_ids_files = [ i for i in all_tiff_files if "field_ids.tif" in i ]

print ( f"All json files           : { len ( all_json_files  ):>5}" )
print ( f"All tiff files           : { len ( all_tiff_files  ):>5}" )
print ( f"B01 tiff files           : { len ( field_ids_files ):>5}" )
print ( f"Field ids tiff files     : { len ( b01_tiff_files  ):>5}" )

# Tiles
print ( "Reading tiles bbox" )

tiles_bbox = dict ()

for i in tqdm ( all_json_files ) :

	# Only process tiles
	if "collection" not in i and "catalog" not in i :

		# Basic info
		tile_id = i.split ( "_" ) [ - 1 ].split ( "." ) [ 0 ]

		# Only handle unknown ones
		if tile_id not in tiles_bbox :

			# bbox from json
			j = open ( i )
			k = json.load ( j )
			j.close ()

			tiles_bbox [ tile_id ] = k [ "bbox" ]

print ( f"\t{ len ( tiles_bbox ) } tiles" )

# Fields
def get_field_pixels ( field_id_file ) :

	with rasterio.open ( field_id_file ) as src :

		field_data = src.read () [ 0 ]

	field_ids = set ( list ( field_data [ field_data > 0 ] ) )

	field_pixels = dict ()

	for i in field_ids :

		field_pixels [ i ] = np.array ( field_data == i, dtype = int )

	return field_pixels

# Given an image, fill the holes in it
# and return the outside mask, boundary
# and inside mask and estimated
# perimeter length
# Note the boundary is 2 pixels wide
def outside_boundary_inside_perimeter ( img ) :

	x0 = 0
	y0 = 0

	if img [ y0, x0 ] :

		x0 = img.shape [ 1 ] - 1
		y0 = img.shape [ 1 ] - 1

	outside  = flood ( img, ( y0, x0 ) )
	inside   = ~outside
	boundary = find_boundaries ( inside )

	return outside, boundary, inside, perimeter ( inside )

# Given the border of an image, return the goodness of fit of a circle
def fit_circle ( boundary ) :

	total = np.count_nonzero ( boundary )

	if total > 10 :

		xy = np.zeros ( [ total, 2 ], dtype = int )
		i  = 0

		for x in range ( boundary.shape [ 1 ] ) :

			for y in range ( boundary.shape [ 0 ] ) :

				if boundary [ y, x ] :

					xy [ i, 0 ] = x
					xy [ i, 1 ] = y

					i = i + 1

		# Fit circle model
		circle = CircleModel ()

		if circle.estimate ( xy ) :

			return 1 / ( 1 + circle.residuals ( xy ).std () )

	return 0.0

# Read all fields and write pixel values to npy
tiles_done    = set ()

# Tile statistics
tile_out      = open ( tiles_fname, "wt" )

# Header
tile_out.write ( "tile_id,band,transformation" )

for i in stats_labels :

	tile_out.write ( "," )
	tile_out.write ( i )

tile_out.write ( "\n" )
tile_out.flush ()

# Field statistics
field_out     = open ( fields_fname, "wt" )

# Header
field_out.write ( "tile_id,field_id,band,transformation,mask" )

for i in [ "n", "x1", "x2", "x3", "x4" ] + stats_labels :

	field_out.write ( "," )
	field_out.write ( i )

field_out.write ( "\n" )
field_out.flush ()

# Other statistics
oth_out = open ( oth_fname, "wt" )

# Header
oth_out.write ( "tile_id,field_id,pixels,size,perimeter,perimeter_to_size,circular_measure\n" )
oth_out.flush ()

# Histogram
hist_out = open ( hist_fname, "wt" )

# Header
hist_out.write ( "tile_id,field_id,band,transformation,mask" )

for i in hist_numbins :

	hist_out.write ( ",lowerlimit_" )
	hist_out.write ( str ( i ) )
	hist_out.write ( ",binsize_" )
	hist_out.write ( str ( i ) )
	hist_out.write ( ",extrapoints_" )
	hist_out.write ( str ( i ) )

	for j in range ( i ) :

		hist_out.write ( ",bin_" )
		hist_out.write ( str ( j ) )
		hist_out.write ( "_" )
		hist_out.write ( str ( i ) )

hist_out.write ( "\n" )
hist_out.flush ()

# Given a mask, update the field statistics
def update_oth ( tile_id, w, h, field_id, mask ) :

	pixels = np.sum ( mask )
	size   = 100 * w * h * pixels

	# Some image processing
	o, b, i, p = outside_boundary_inside_perimeter ( mask )

	perimeter = p
	perimeter_to_size = p / pixels
	circular_measure = fit_circle ( b )

	# Now print results
	oth_out.write ( str ( tile_id ) )

	for i in [ field_id, pixels, size, perimeter, perimeter_to_size, circular_measure ] :

		oth_out.write ( "," )
		oth_out.write ( str ( i ) )

	oth_out.write ( "\n" )

# Bit crude but we do it just once ...
def safe_div ( a, b, b0 = 0 ) :

	return np.divide ( a, b, out = np.full ( b.shape, b0, dtype = float ), where = b != 0 )

def calc_band ( a, b ) :

	num  = a - b
	den  = a + b

	return safe_div ( a, b, 0 )

for i, I in enumerate ( b01_tiff_files ) :

	tile_id   = I.split ( "_" ) [ - 1 ].split ( "/" ) [ 0 ]

	if tile_id not in tiles_done :

		tiles_done.add ( tile_id )

		path           = I [ : I.rindex ( "/" ) + 1 ]
		tifs           = []
		tile_field_ids = [ j for j in field_ids_files if tile_id in j ]
		os.path.join ( input_dir, "ref_agrifieldnet_competition_v1_labels_train/ref_agrifieldnet_competition_v1_labels_train_" + tile_id + "/field_ids.tif" )

		print ( f"Now doing { tile_id }" )

		bbox       = tiles_bbox [ tile_id ]
		tile_w     = bbox [ 2 ] - bbox [ 0 ]
		tile_h     = bbox [ 3 ] - bbox [ 1 ]

		# Extract bands and update tile statistics
		for j in bands :

			print ( f"\t{ j }" )
			tif_fname = os.path.join ( path, j ) + ".tif"

			with rasterio.open ( tif_fname ) as tif :

				img = np.array ( tif.read () [ 0 ] )

			for k, K in enumerate ( transforms ) :

				x = K ( img.flatten () ).reshape ( img.shape )
				tifs.append ( x )
				x = x.flatten ()

				# Write output
				tile_out.write ( tile_id )

				for l in [ j, trans_labels [ k ] ] :

					tile_out.write ( "," )
					tile_out.write ( l )

				for l in calc_stats :

					tile_out.write ( "," )
					tile_out.write ( str ( l ( x ) ) )

				tile_out.write ( "\n" )
				tile_out.flush ()

		# Derived bands
		b2     = tifs [ bands.index ( "B02" ) * len ( transforms ) ].flatten ()
		b3     = tifs [ bands.index ( "B03" ) * len ( transforms ) ].flatten ()
		b4     = tifs [ bands.index ( "B04" ) * len ( transforms ) ].flatten ()
		b5     = tifs [ bands.index ( "B05" ) * len ( transforms ) ].flatten ()
		b6     = tifs [ bands.index ( "B06" ) * len ( transforms ) ].flatten ()
		b7     = tifs [ bands.index ( "B07" ) * len ( transforms ) ].flatten ()
		b8     = tifs [ bands.index ( "B08" ) * len ( transforms ) ].flatten ()
		b8a    = tifs [ bands.index ( "B8A" ) * len ( transforms ) ].flatten ()
		b11    = tifs [ bands.index ( "B11" ) * len ( transforms ) ].flatten ()
		b12    = tifs [ bands.index ( "B12" ) * len ( transforms ) ].flatten ()

		# ndvi, bndvi, gndvi, gbndvi, grndvi, rbndvi, gari, nbr, ndmi, npcri, avi, bsi
		ndvi   = calc_band ( b8, b4 )
		bndvi  = calc_band ( b8, b2 )
		gndvi  = calc_band ( b8, b3 )
		gbndvi = calc_band ( b8a, b2 + b3 )
		grndvi = calc_band ( b8a, b3 + b4 )
		rbndvi = calc_band ( b8a, b2 + b4 )
		gari   = calc_band ( b8 - b3, b4 - b2 )
		nbr    = calc_band ( b8, b12 )
		ndmi   = calc_band ( b8, b11 )
		npcri  = calc_band ( b4, b2 )

		t      = b8 * ( 256 - b4 ) * ( b8 - b4 )
		avi    = np.sign ( t ) * np.power ( np.abs ( t ), 1 / 3 )

		bsi    = calc_band ( b4 + b11, b2 + b8 )

		db1    = [ ndvi, bndvi, gndvi, gbndvi, grndvi, rbndvi, gari, nbr, ndmi, npcri, avi, bsi ]

		# si, bri, msavi, ndsi, ndre, ngrdi, rdvi, sipi, gci, gbndv2, grndv2
		t      = ( 256 - b4 ) * ( 256 - b3 ) * ( 256 - b2 )
		si     = np.sign ( t ) * np.power ( np.abs ( t ), 1 / 3 )

		bri    = safe_div ( safe_div ( 1, b3 ) - safe_div ( 1, b5 ), b6 )
		msavi  = ( 2 * b8 + 1 - np.sqrt ( np.power ( 2 * b8 + 1, 2 ) - 8 * ( b8 - b4 ) ) ) / 2
		ndsi   = calc_band ( b11, b12 )
		ndre   = calc_band ( b8a, b5 )
		ngrdi  = calc_band ( b3, b5 )
		rdvi   = safe_div ( b8 - b4, np.sqrt ( b8 + b4 ) )
		sipi   = safe_div ( b8 - b2, b8 + b4 )
		gci    = safe_div ( b8, b3 )
		gbndv2 = calc_band ( b3, b2 )
		grndv2 = calc_band ( b3, b4 )

		db2    = [ si, bri, msavi, ndsi, ndre, ngrdi, rdvi, sipi, gci, gbndv2, grndv2 ]

		# reip, slavi, tcari, tci, wdrvi, ari, myvi, fe2, cvi, varig
		reip   = 700 + 40 * safe_div ( ( b4 + b7 ) / 2 - b5, b6 - b5 )
		slavi  = safe_div ( b8, b4 + b12 )
		tcari  = 3 * ( b5 - b4 - 0.2 * ( b5 - b3 ) * safe_div ( b5, b4 ) )
		tci    = 1.2 * ( b5 - b3 ) - 1.5 * ( b4 - b3 ) * np.sqrt ( safe_div ( b5, b4 ) )
		wdrvi  = calc_band ( 0.1 * b8a, b5 )
		ari    = safe_div ( 1, b3 ) - safe_div ( 1, b5 )
		myvi   = 0.723 * b3 - 0.597 * b4 + 0.206 * b6 - 0.278 * b8a
		fe2    = safe_div ( b12, b8 ) + safe_div ( b3, b4 )
		cvi    = safe_div ( b8 * b4, np.power ( b3, 2 ) )
		varig  = safe_div ( b3 - b4, b3 + b4 - b2 )

		db3    = [ reip, slavi, tcari, tci, wdrvi, ari, myvi, fe2, cvi, varig ]

		# evi, mi, savi
		evi    = 2.5 * safe_div ( b8 - b4, b8 + 6 * b4 - 7.5 * b2 + 1 )
		mi     = safe_div ( b8a - b11, b4 + b8a )
		savi   = safe_div ( b8 - b4, 1.725 * ( b4 + b8 + 0.725 ) )

		db4    = [ evi, mi, savi ]

		for j, J in zip ( derived_bands, db1 + db2 + db3 + db4 ) :

			print ( f"\t{ j }" )

			for k, K in enumerate ( transforms ) :

				x = K ( J.flatten () ).reshape ( img.shape )
				tifs.append ( x )
				x = x.flatten ()

				# Write output
				tile_out.write ( tile_id )

				for l in [ j, trans_labels [ k ] ] :

					tile_out.write ( "," )
					tile_out.write ( l )

				for l in calc_stats :

					tile_out.write ( "," )
					tile_out.write ( str ( l ( x ) ) )

				tile_out.write ( "\n" )
				tile_out.flush ()

		# Fields in this tile
		print ( "\tFields" )
		for j, J in enumerate ( tile_field_ids ) :

			if os.path.isfile ( J ) :

				# Fields
				field_pixels = get_field_pixels ( J )

				for field_id, mask in field_pixels.items () :

					print ( f"\t\t{ field_id } with { mask.sum () } pixels" )

					print ( f"\t\t\tother info" )
					update_oth ( tile_id, tile_w, tile_h, field_id, mask )

					print ( f"\t\t\tmasks" )

					masks = []

					for b, B in zip ( borders, border_labels ) :

						print ( f"\t\t\t\t{ B }" )

						if b < 0 :

							# Shrink label
							x = mask.copy ()

							for k in range ( abs ( b ) ) :

								y = binary_erosion ( x, x )

								if y.sum () == 0 :

									y = x
									break

								else :

									x = y

							masks.append ( np.array ( x, dtype = bool ) )

						elif b == 0 :

							# No change
							masks.append ( np.array ( mask, dtype = bool ) )

						else :

							# Expand label
							x = mask.copy ()

							for k in range ( abs ( b ) ) :

								x = expand_labels ( x )

							masks.append ( np.array ( x - mask, dtype = bool ) )

					# Now apply the masks to all the bands
					print ( "\t\t\tapplying" )
					ti = 0

					for k in bands + derived_bands :

						for l in trans_labels :

							tif = tifs [ ti ]
							ti += 1

							for m, M in enumerate ( masks ) :

								x  = tif [ M ].flatten ()

								if len ( x ) != M.sum () :

									print ( "Mask not working" )
									print ( f"x { x.shape }" )

								n  = len ( x )
								x1 = np.sum ( x )
								x2 = np.sum ( x * x )
								x3 = np.sum ( x * x * x )
								x4 = np.sum ( x * x * x * x )
								r  = [ n, x1, x2, x3, x4 ]

								for f in calc_stats :

									r.append ( f ( x ) )

								# Fit histogram
								cf = [ cumfreq ( x, numbins = f ) for f in hist_numbins ]

								# Write output
								field_out.write ( tile_id )

								for x in [ field_id, k, l, border_labels [ m ] ] + r :

									field_out.write ( "," )
									field_out.write ( str ( x ) )

								field_out.write ( "\n" )
								field_out.flush ()

								# Write historgram
								hist_out.write ( tile_id )

								for x in [ field_id, k, l, border_labels [ m ] ] :

									hist_out.write ( "," )
									hist_out.write ( str ( x ) )

								for x in cf :

									for y in [ x.lowerlimit, x.binsize, x.extrapoints / n ] :

										hist_out.write ( "," )
										hist_out.write ( str ( y ) )

									# Difference the cumulative histogram
									z = 0

									for y in ( x.cumcount / n ).tolist () :

										hist_out.write ( "," )
										hist_out.write ( str ( y - z ) )
										z = y

								hist_out.write ( "\n" )
								hist_out.flush ()

					if ti != len ( tifs ) :

						print ( "Did not use all tifs" )

# Save results
print ( "Closing files" )
tile_out.close ()
field_out.close ()
oth_out.close ()
hist_out.close ()

print ( "All done"    )
print ( "MG Ferreira" )
print ( "2022"        )
