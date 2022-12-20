import os

import numpy as np
import pandas as pd

# Parameters
base_name      = "013"

tiles_fname    = os.path.join ( "data", "d" + base_name + "-tiles.csv"        )
fields_fname   = os.path.join ( "data", "d" + base_name + "-fields.csv"       )
oth_fname      = os.path.join ( "data", "d" + base_name + "-oth.csv"          )
out_fname      = os.path.join ( "data", "d" + base_name + "-fields+tiles.csv" )
groupby_fname  = os.path.join ( "data", "d" + base_name + "-fields-only.csv"  )

# Read and prepare inputs
data           = pd.read_csv ( fields_fname )
idxs           = [ "field_id", "band", "transformation", "mask" ]
cols           = [ "tile_id", "field_id", "band", "transformation", "mask", "min", "max", "mean", "median", "mode", "std", "skew", "kurt" ]

data [ cols ].to_csv ( out_fname, index = False )

# Calculate statistics per field id
# Thus a field that spans several tiles will now be aggregated across those tiles
# Index of these is idxs
mins           = data [ idxs + [ "min"    ] ].groupby ( by = idxs ).min  ()
maxs           = data [ idxs + [ "max"    ] ].groupby ( by = idxs ).max  ()
medians        = data [ idxs + [ "median" ] ].groupby ( by = idxs ).mean ()
sums           = data [ idxs + [ "n", "x1", "x2", "x3", "x4" ] ].groupby ( by = idxs ).sum ()

# Prepare dataframe with aggregation results
df             = pd.DataFrame ( index = mins.index, columns = [ "min", "max", "n", "mean", "median", "mode", "std", "skew", "kurt" ] )

df [ "min"    ] = mins    [ "min"    ]
df [ "max"    ] = maxs    [ "max"    ]
df [ "median" ] = medians [ "median" ]

# Use these to calculate aggregate statistics
n              = sums [ "n"  ].to_numpy ( dtype = float ).flatten ()
x1             = sums [ "x1" ].to_numpy ( dtype = float ).flatten ()
x2             = sums [ "x2" ].to_numpy ( dtype = float ).flatten ()
x3             = sums [ "x3" ].to_numpy ( dtype = float ).flatten ()
x4             = sums [ "x4" ].to_numpy ( dtype = float ).flatten ()
t              = x2 - x1 * x1 / n

# Aggregate - note scaling
df [ "n"    ]  = n
df [ "mean" ]  = x1 / n
df [ "std"  ]  = np.sqrt ( t / ( n - 1 ) )
df [ "skew" ]  = ( x3 * n - 3 * x1 * x2 + 2 * x1 * x1 * x1 / n ) / ( t * np.sqrt ( t * n ) )
df [ "kurt" ]  = n * ( x4 - 4 * x1 * x3 / n + 6 * x1 * x1 * x2 / ( n * n ) - 3 * x1 * x1 * x1 * x1 / ( n * n * n ) ) / ( t * t ) - 3

df.loc [ ~ np.isfinite ( df [ "std" ].to_numpy ( dtype = float ) ), [ "skew", "kurt" ] ] = np.nan
df.loc [ df [ "std" ] == 0, [ "skew", "kurt" ] ] = np.nan

print ( df )

# Write output
df.to_csv ( groupby_fname )

# Done
print ( "All done"    )
print ( "MG Ferreira" )
print ( "2022"        )
