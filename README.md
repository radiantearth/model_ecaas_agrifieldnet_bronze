# Looking further: a crop type classification model for fields

This model classifies crop types for each field based on the field as well as
on its surroundings.

In the [Zindi AgriFieldNet competition](https://zindi.africa/competitions/agrifieldnet-india-challenge)
this was the third place solution by the team `re-union` in the final round to classify crop
types in agricultural fields across Northern India using multispectral
observations from Sentinel-2 satellite.

![model_ecaas_agrifieldnet_bronze_v1](https://radiantmlhub.blob.core.windows.net/frontend-ml-model-images/model_ecaas_agrifieldnet_bronze_v1.png)

MLHub model id: `model_ecaas_agrifieldnet_bronze_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/model_ecaas_agrifieldnet_bronze_v1).

## ML Model Documentation

Please review the model architecture, license, applicable spatial and temporal extents
and other details in the [model documentation](/docs/index.md).

## System Requirements

* Git client
* [Docker](https://www.docker.com/) with
    [Compose](https://docs.docker.com/compose/) v1.28 or newer.

## Hardware Requirements

|Inferencing|Training|
|-----------|--------|
|12 GB RAM | 30 GB RAM|

## Get Started With Inferencing

First clone this Git repository.

Please note: this repository uses
[Git Large File Support (LFS)](https://git-lfs.github.com/) to include the
model checkpoint file. Either install `git lfs` support for your git client,
use the official Mac or Windows GitHub client to clone this repository.

:zap: Shell commands have been tested with Linux and MacOS but will
differ on Windows, or depending on your environment.

```bash
git clone https://github.com/radiantearth/model_ecaas_agrifieldnet_bronze.git
cd model_ecaas_agrifieldnet_bronze/
```

After cloning the model repository, you can use the Docker Compose runtime
files as described below.

## Pull or Build the Docker Image

Pull pre-built image from Docker Hub (recommended):

```bash
docker pull docker.io/radiantearth/model_ecaas_agrifieldnet_bronze:1
```

Or build image from source:

```bash
cd docker-services/
docker build -t radiantearth/model_ecaas_agrifieldnet_bronze:1 .
```

## Run Model to Generate New Inferences

1. Prepare your input and output data folders:

    * The `data/input` folder in this repository contains some placeholder files to guide you.
    The input data should follow the following convention. It should be placed in a directory named
    `xxx_<tile_id>`,
    where `xxx` is arbitrary and `<tile_id>` represents the id of the tile stored in that directory.

    Here is a sample for reference.

    ```text
    data/input/data_001c1
    data/input/data_004fa
    data/input/data_005fe
    data/input/source_001c1
    data/input/source_0023c
    data/input/source_004fa
    ```

    These directories will contain tiff files for three tiles (id `001c1`,
    `004fa` and `005fe`). It does not matter where the bands or field ids are,
    but note that the directory must split on `_` and the last portion must be
    the tile id. This is in accordance with the competition data.

    * The `output/` folder is where the model will write inferencing results.

2. Set `INPUT_DATA` and `OUTPUT_DATA` environment variables corresponding with
    your input and output folders. These commands will vary depending on operating
    system and command-line shell:

    ```bash
    # change paths to your actual input and output folders
    export INPUT_DATA="/home/my_user/model_ecaas_agrifieldnet_bronze/data/input/"
    export OUTPUT_DATA="/home/my_user/model_ecaas_agrifieldnet_bronze/data/output/"
    export MODELS_DIR="/home/my_user/model_ecaas_agrifieldnet_bronze/models"
    export WORKSPACE_DIR="/home/my_user/model_ecaas_agrifieldnet_bronze/workspace"
    ```

3. Run the appropriate Docker Compose command for your system:

    ```bash
    docker compose up model_ecaas_agrifieldnet_bronze_v1
    ```

4. Wait for the `docker compose` to finish running, then inspect the
`OUTPUT_DATA` folder for results.

## Understanding Output Data

Please review the model output format and other technical details in the [model
documentation](/docs/index.md).
