# Create a submission file by ensembling a few others
import pandas as pd

print ( "Configuring" )

subs = [ "subs/sub-m013-37-w.csv",
	 "subs/sub-m013-35-w.csv",
	 "subs/sub-m013-12-w.csv",
	 "subs/sub-m013-8-w.csv",
	 "subs/sub-m013-36-w.csv",
	 "subs/sub-m013-14-w.csv",
	 "subs/sub-m013-38-w.csv",
	 "subs/sub-m013-21-w.csv" ]

# Weight based on local CV
wgts = [ 1 / i for i in [ 1.20928278481616,
			  1.20904773696184,
			  1.21314857913361,
			  1.23284698120985,
			  1.21363149393413,
			  1.21192870775733,
			  1.20811225571932,
			  1.23826407088706 ] ]
wtot = sum ( wgts )
sub_fname = "subs/sub-c033.csv"

# Same order as in sample submission
labels_text = [ "Wheat", "Mustard", "Lentil", "No Crop", "Green pea", "Sugarcane", "Garlic", "Maize", "Gram", "Coriander", "Potato", "Bersem", "Rice" ]

print ( "Opening submission files" )

inps = [ pd.read_csv ( i ).set_index ( "field_id" ) for i in subs ]

print ( "Calculating average and writing new sub" )

combo = pd.DataFrame ( index = inps [ 0 ].index, columns = labels_text, data = 0, dtype = float )

for i, j, k in zip ( inps, wgts, subs ) :

	print ( f"\t{ k } { j }/{ wtot }" )
	combo.loc [ :, labels_text ] += i.loc [ :, labels_text ] * j / wtot

print ( "Writing submission file" )

combo.to_csv ( sub_fname )

print ( "Done" )
print ( "MG Ferreira" )
print ( "2022" )
