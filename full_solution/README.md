# re-union: winning solution

This documents the following submission

 - ID: `ALpJXTBK`
 - Filename: `sub-c033.csv`
 - Comment: `—`
 - Submitted: `31 October 23:09`

This submission was mode on behalf of team `re-union` in the `AgriFieldNet India Challenge` and placed seventh on the private leaderboard, but eventually was upgraded to third place.

This solution was a team effort.

This README documents my (`skaak`) contribution.

It does not document my team member (`moto`) contribution, but does give instructions on how to include the output from his model into the final model here. It also includes his model and instructions on how to run it, as prepared by him.

## Environment

Below are details of the environment. The solution is fairly general and should give the same results on any recent Python and using recent versions of the mentioned packages.

### Python

```
python --version
Python 3.10.8
```

This was the latest version during the competition.

#### Packages

The relevant packages are shown below. Only relevant packages are shown.

```
imbalanced-learn==0.9.1
joblib==1.2.0
numpy==1.23.3
pandas==1.5.0
rasterio==1.3.2
scikit-image==0.19.3
scikit-learn==1.1.2
scipy==1.9.1
tqdm==4.64.1
```

These are typically the latest versions during the time of the competition. More recent versions will probably work, but may not reproduce the solution exactly.

## Setup

### Introduction

Note that what is described below is also reflected in this repository. If you clone it, it should create everything as described below, except for the competition data, which is not included.

### Directory structure

The solution has been placed in the

```
full_solution
```

directory of the repository. Consider that the home directory of the full solution, and then follow the instructions below.

The solution uses the following directory structure relatove to its home directory. You can also create this structure as shown next.

```
cd full_solution               # The root where you store the solution
mkdir code                     # All code files are stored here
mkdir code/data                # All processed data files are stored here
mkdir code/models              # All models are stored here
mkdir code/subs                # All submission files are stored here
mkdir input                    # The downloaded data files are stored here
```

Alternatively, this is what it should look like

```
full_solution
  |-- code
  |     |-- data
  |     |-- models
  |     |-- subs
  |-- input
  |-- verify
  |-- README.md
```

The repository contains an additional directory for reference, called

```
verify
```

that can be used to verify some of the interim (as well as the final) output of the model to speed up the verification process.

This is optional but could assist you a lot. Details on this follow below.

### Competition data

Copy all the competition data files into the

```
input
```

directory.

Here is what the first few files in this directory should look like when done.

```
full_solution
  |-- input
  |     |-- SampleSubmission.csv
  |     |-- ref_agrifieldnet_competition_v1
  |           |-- catalog.json
  |           |-- err_report.csv
  |           |-- mlhub_stac_assets.db
  ...
```

Note that the competition data is *not* included here. You *have to* copy it into this directory and you *have to* ensure it aligns with the structure as shown above. This is one reason why I give the first few files, to assist in confirming that this step was executed correctly.

### Moto data and pipeline

Note that I have already included `moto`s output file in this repository. If you do not want to create it yourself, you can skip to the next step.

The output of the other team member, `moto` should be copied into the

```
code/data
```

directory.

#### Rename the file!

Note that moto's output file must be renamed from

```
dict_predictions.pickle
```

to

```
dict_predictions_neighbours.pickle
```

and copied into the

```
code/data
```

directory.

When done, it should appear like this.

```
code/data/dict_predictions_neighbours.pickle
```

#### Moto pipeline

Note how the model described here uses only the output from `moto`'s pipeline. His model was run on kaggle and, together with instructions on how to run it, can be found in the

```
code/moto
```

directory. It contains the following notebooks that can be used to reproduce his input into the model.

```
radiant-how-to-read.ipynb
radiant-neighbour2images.ipynb
radiant-neighbours-features-extraction.ipynb
```

It also contains the output from his model

```
dict_predictions.pickle
```

which, as mentioned before, needs to be renamed and copied into place.

##### Moto environment

Note that moto used kaggle to run his model as also mentioned before and as described in his notebooks.

## Reproducing the solution

The steps below describe how to reproduce the rest of the solution, given moto's output.

### Preprocess the data

To preprocess the data, change into the `code` directory and execute the following

```
cd code
python read_labels.py                # Process the labels
python d013.py                       # Process the data
python d013-support.py               # Process the data
```

This will read and process the competition data and create output files in the

```
code/data
```

directory.

This step will take a few hours to complete.

### Run the models

The winning solution is an ensemble of eight models. They can be run next. Each model uses 10 folds and each fold takes around an hour to model. This means that this step will take roughly 80 hours to complete!

I used a modest configuration without a GPU. It may execute much faster in a different environment, e.g. with a GPU, but I am not sure it will reproduce the same results.

Below is the code to run to reproduce the eight models and their respective submission files. It is assumed you are still in the

```
code
```

directory from which these commands need to be executed.

```
python m013-37.py     # Best model on public LB
python m013-35.py     # Second best model
python m013-12.py     # 3rd best
python m013-8.py      # 4th best
python m013-36.py
python m013-14.py
python m013-38.py
python m013-21.py
```

### Ensemble

The models can now be ensembled. This can be done by executing the following, assuming you are still in the

```
code
```

directory.

```
python c033.py        # Ensemble top eight models
```

This will create an ensemble by using a weighted average of the eight models. The weight is the inverse of the local CV score.

### Final submission file

When done, the final submission file will be available, together with the interim submissions, in the

```
code/subs
```

directory. The final submission file or output of the model, is

```
code/subs/sub-c033.csv
```

### Suggestion

As mentioned, to run the whole model, will take a long time. An alternative is to run a random subset of the eight models and verify that the generated submission files, one of

```
code/subs/sub-m013-37-w.csv
code/subs/sub-m013-35-w.csv
code/subs/sub-m013-12-w.csv
code/subs/sub-m013-8-w.csv
code/subs/sub-m013-36-w.csv
code/subs/sub-m013-14-w.csv
code/subs/sub-m013-38-w.csv
code/subs/sub-m013-21-w.csv
```

agrees with what was submitted earlier, as each of these individual models were also submitted independently.

For ease of reference, these files, as well as the final submission `sub-c033.csv` are provided in the

```
verify
```

directory. Thus you can use any of these to verify a subset of the final model.

```
verify/sub-m013-37-w.csv
verify/sub-m013-35-w.csv
verify/sub-m013-12-w.csv
verify/sub-m013-8-w.csv
verify/sub-m013-36-w.csv
verify/sub-m013-14-w.csv
verify/sub-m013-38-w.csv
verify/sub-m013-21-w.csv
verify/sub-c033.csv
```

## Review of model

Here is a brief review of the model. The eight models are variations of the same basic model. It does the following.

It evaluates the pixels of the field itself as well as a series of mutually exclusive consentric borders around the field. The pixel values are evaluated as is as well as standardised using both the quantile and uniform transformations. A large number of bands are calculated and also evaluated.

The focus is on measures of locality, such as mean, median and mode. A selection of these features are clustered and the clusters are also provided as features to the final model.

`moto` my team member provided a classification for the area to the north, east, west and south of each field, as well as centered on the field. This classification is provided as inputs to the main model.

Additional locality based models, such as `linear discriminant` analysis, are used to classify the fields and these classifications are provided as inputs into the final model.

The final model is a random forest model that combines the features into a single classification.

### Concentric borders

The following sizes are used to construct the borders around each field.

 - `F0`: the field itself without any border
 - `F+3`: a border of width 3 around the field
 - `F+20`: a border of width 20 around the previous border
 - `F+40`: a border of width 40 around the previous border

### Bands

The following bands are used.

```
bands          = [ "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12" ]
derived_bands1 = [ "NDVI", "BNDVI", "GNDVI", "GBNDVI", "GRNDVI", "RBNDVI", "GARI", "NBR", "NDMI", "NPCRI", "AVI", "BSI" ]
derived_bands2 = [ "SI", "BRI", "MSAVI", "NDSI", "NDRE", "NGRDI", "RDVI", "SIPI", "GCI", "GBNDV2", "GRNDV2" ]
derived_bands3 = [ "REIP", "SLAVI", "TCARI", "TCI", "WDRVI", "ARI", "MYVI", "FE2", "CVI", "VARIG" ]
derived_bands4 = [ "EVI", "MI", "SAVI" ]
```

### Transformations

The quantile transformer is used to provide `normal` and `uniform` versions of the bands.

### Statistics

The following statistics are calculated for each field, band and transformation.

```
stats_labels   = [ "min", "max", "mean", "median", "mode", "std", "skew", "kurt" ]
```

Note that each of the eight models will use some selection of these to construct the final model.

### Clusters

A selection of the features are clustered in each model and the clusters are provided as additional features.

### Final model

The final model builds a random forest using the features pertaining to that model.

### Output

The model output is weighted by the number of pixels in each field and this weight is used to combine fields that span different tiles.

## Inference

The model was not created to easily perform inference, rather, it requires the full competition input data in order to produce the submission file as output. One difficulty presented when performing inference is that the model will cluster the data on the fly using e.g. OPTICS, that can not be applied afterwards to new data. It requires all data to cluster - an unfortunate side effect of the OPTICS clustering algorithm.

However, there is an additional file included in the

```
code
```

directory, called

```
code/m013-37f.py
```

that is similar to the

```
code/m013-37.py
```

in all respects, except that it will write out additional outputs for use in the model inference. These additional outputs are the cluster models that were fitted on the fly.

In order to perform inference using the docker, all the model files produced for model `m013-37` need to be copied from the

```
code/models
```

directory to the inference docker where they will be used by the inference process built up around a somewhat simplified version of the full `m013-37` model.

The files that have to be transferred to docker are all files of the form

```
code/models/m013-37*
```

and the full listing is provided below for reference.

```
code/models/m013-37-0.joblib
code/models/m013-37-1.joblib
code/models/m013-37-2.joblib
code/models/m013-37-3.joblib
code/models/m013-37-4.joblib
code/models/m013-37-5.joblib
code/models/m013-37-6.joblib
code/models/m013-37-7.joblib
code/models/m013-37-8.joblib
code/models/m013-37-9.joblib
code/models/m013-37-affinity_propagation-huge-cluster.joblib
code/models/m013-37-affinity_propagation-huge-selected.joblib
code/models/m013-37-affinity_propagation-large-cluster.joblib
code/models/m013-37-affinity_propagation-large-selected.joblib
code/models/m013-37-affinity_propagation-medium-cluster.joblib
code/models/m013-37-affinity_propagation-medium-selected.joblib
code/models/m013-37-affinity_propagation-small-cluster.joblib
code/models/m013-37-affinity_propagation-small-selected.joblib
code/models/m013-37-kmeans_10-huge-cluster.joblib
code/models/m013-37-kmeans_10-huge-selected.joblib
code/models/m013-37-kmeans_10-large-cluster.joblib
code/models/m013-37-kmeans_10-large-selected.joblib
code/models/m013-37-kmeans_10-medium-cluster.joblib
code/models/m013-37-kmeans_10-medium-selected.joblib
code/models/m013-37-kmeans_10-small-cluster.joblib
code/models/m013-37-kmeans_10-small-selected.joblib
code/models/m013-37-kmeans_20-huge-cluster.joblib
code/models/m013-37-kmeans_20-huge-selected.joblib
code/models/m013-37-kmeans_20-large-cluster.joblib
code/models/m013-37-kmeans_20-large-selected.joblib
code/models/m013-37-kmeans_20-medium-cluster.joblib
code/models/m013-37-kmeans_20-medium-selected.joblib
code/models/m013-37-kmeans_20-small-cluster.joblib
code/models/m013-37-kmeans_20-small-selected.joblib
code/models/m013-37-kmeans_40-huge-cluster.joblib
code/models/m013-37-kmeans_40-huge-selected.joblib
code/models/m013-37-kmeans_40-large-cluster.joblib
code/models/m013-37-kmeans_40-large-selected.joblib
code/models/m013-37-kmeans_40-medium-cluster.joblib
code/models/m013-37-kmeans_40-medium-selected.joblib
code/models/m013-37-kmeans_40-small-cluster.joblib
code/models/m013-37-kmeans_40-small-selected.joblib
code/models/m013-37-optics-huge-cluster.joblib
code/models/m013-37-optics-huge-selected.joblib
code/models/m013-37-optics-large-cluster.joblib
code/models/m013-37-optics-large-selected.joblib
code/models/m013-37-optics-medium-cluster.joblib
code/models/m013-37-optics-medium-selected.joblib
code/models/m013-37-optics-small-cluster.joblib
code/models/m013-37-optics-small-selected.joblib
```

These files need to be copied from the solution directory to the

```
/models/main
```

directory of the repository so that it can be available in docker.

### Inference using the full model

To fully reproduce the results as submitted to the competition requires to follow the steps outlined above. To perform inference on a small set of images is not as straightforward, mostly because of the clustering as explained earlier.

The `m013-37.py` model has been rewritten in the accompanying `m013-37i.py` model to specifically cater for inference, and serves as an example of how to transform any additional model scripts if required. More info on the process is also available in the inference docker.

Note that on the public leaderboard the `m013-37` model scored a little bit better than the full ensemble, so it should not make a big difference if only this model is used, but it will not reproduce the competition results exactly.
