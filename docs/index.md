# Looking further

This model classifies crop types for each field based on the field as well as on its surroundings.

![model_ecaas_agrifieldnet_bronze_v1](https://radiantmlhub.blob.core.windows.net/frontend-dataset-images/odk_sample_agricultural_dataset.png)

MLHub model id: `model_ecaas_agrifieldnet_bronze_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/model_ecaas_agrifieldnet_bronze_v1).

## Training Data

- [AgriFieldNet Competition Dataset - Source Imagery](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_source)
- [AgriFieldNet Competition Dataset - Test Labels](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_labels_train)

## Related MLHub Dataset

[AgriFieldNet Competition Dataset](https://mlhub.earth/data/ref_agrifieldnet_competition_v1)

## Citation

Ferreira, M.G. and Dung, T. "Looking further", Version 1.0, Radiant MLHub. [Date Accessed] Radiant MLHub

## License

[CC-BY-4.0](../LICENSE)

## Creators

MG Ferreira - Ferra Solutions www.ferrasolutions.com [LinkedIn](https://www.linkedin.com/in/mg-ferreira-35534)

Tien Dung

## Contact

Use LinkedIn to message.

## Applicable Spatial Extent

The applicable spatial extent, for new inferencing.

```geojson
{
    "type": "FeatureCollection",
    "features": [
        {
            "properties": {
                "id": "ref_agrifieldnet_competition_v1"
            },
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "bbox": [
                    76.2448,
                    18.9414,
                    88.046,
                    28.327
                ],
                "coordinates": [
                    [
                        [
                            [
                                88.046,
                                18.9414
                            ],
                            [
                                88.046,
                                28.327
                            ],
                            [
                                76.2448,
                                28.327
                            ],
                            [
                                76.2448,
                                18.9414
                            ],
                            [
                                88.046,
                                18.9414
                            ]
                        ]
                    ]
                ]
            }
        }
    ]
}
```

## Applicable Temporal Extent

The recommended start/end date of imagery for new inferencing.

| Start | End |
|-------|-----|
| 2022-01-01 | present |

## Learning Approach

- Supervised

## Prediction Type

- Classification

## Model Architecture

This will calculate descriptive statistics for each field as well as for a number of concentric borders around it. It will also classify the terrain type around the field. All of these features are then clustered, upsampled and finally fed into a random forest for classification.

## Training Operating System

- MacOS (darwin)

## Training Processor Type

- cpu

## Model Inferencing

Review the [GitHub repository README](../README.md) to get started running
this model for new inferencing.

## Methodology

### Training

The data is processed first. For each band and each field, a number of descriptive statistics such as mean, median, standard deviation and so on are calculated. Then a number of concentric borders are drawn around the field of varying thickness. The bands in each border is analysed in the same manner and contributes more descriptive statistic features. A pretrained EarthNet model is also used to classify the terrain to the north, east, west and south of the field and this classification is added to the features. Next the features are clustered and then upsampled to be more balanced.

### Model

The model itself is a random forest. A gradient booster does not work as well because the features are highly correlated.

### Structure of Output Data

Since a field can span multiple tiles, two outputs are prepared.

- A normal CSV file where the crop type probability prediction is averaged across tiles for each field.
- A CSV file ending in `-w.csv` where the probabilities are weighted according to the size of each field in each tile (this is the output used in the competition).
