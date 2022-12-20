# Crop Classification Models

Third place solution by the team `re-union` in the final round to classify crop types in agricultural fields across Northern India using multispectral observations from Sentinel-2 satellite. 

## ML Model Documentation

Please review the model architecture, license, applicable spatial and temporal extents
and other details in the [model documentation](/full_solution/README.md).

## System Requirements

* Git client
* [Docker](https://www.docker.com/) with
    [Compose](https://docs.docker.com/compose/) v1.28 or newer.

## Hardware Requirements

|Inferencing|Training|
|-----------|--------|
|12 GB RAM | 30 GB RAM|

## Get Started With Inferencing

1. Clone this Git repository, you need to use git lfs to get all the big files.

2. Prepare your input data in the data folder

The input data should follow the following convention. It should be placed in a directory named

```
xxx_<tile_id>
```

where `xxx` is arbitrary and `<tile_id>` represents the id of the tile stored in that directory.

Here is a sample for reference.

```
data/input/data_001c1
data/input/data_004fa
data/input/data_005fe
data/input/source_001c1
data/input/source_0023c
data/input/source_004fa
```

These directories will contain tiff files for three tiles (id `001c1`, `004fa` and `005fe`). It does not matter where the bands or field ids are, but note that the directory must split on `_` and the last portion must be the tile id. This is in accordance with the competition data.

3. Build the docker image

    ```bash
    cd docker-service
    docker build -t skamot/model_radiant:4 -f Dockerfile .
    ```

4. Run Model to generate the predictions

    ```bash
    # change paths to your actual input and output folders
    export INPUT_DATA="/opt/radiant/docker-solution/data/input"
    export OUTPUT_DATA="/opt/radiant/docker-solution/data/output"
    export MODELS_DIR="/opt/radiant/docker-solution/models"
    export WORKSPACE_DIR="/opt/radiant/docker-solution/workspace"

    docker-compose up model_radiant_3
    ```

4. Wait for the `docker compose` to finish running, then inspect the
`OUTPUT_DATA` folder for results.
