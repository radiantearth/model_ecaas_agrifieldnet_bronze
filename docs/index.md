# Looking further: a crop type classification model for fields

This model classifies crop types for each field based on the field as well as on its surroundings.
In the [Zindi AgriFieldNet India Challenge](https://zindi.africa/competitions/agrifieldnet-india-challenge)
this was the third place solution by the team `re-union` in the final round to classify crop
types in agricultural fields across Northern India using multispectral
observations from Sentinel-2 satellite.

![model_ecaas_agrifieldnet_bronze_v1](https://radiantmlhub.blob.core.windows.net/frontend-ml-model-images/model_ecaas_agrifieldnet_bronze_v1.png)

MLHub model id: `model_ecaas_agrifieldnet_bronze_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/model_ecaas_agrifieldnet_bronze_v1).

## Training Data

- [AgriFieldNet Competition Dataset - Source Imagery](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_source)
- [AgriFieldNet Competition Dataset - Test Labels](https://api.radiant.earth/mlhub/v1/collections/ref_agrifieldnet_competition_v1_labels_train)

## Related MLHub Dataset

[AgriFieldNet Competition Dataset](https://mlhub.earth/data/ref_agrifieldnet_competition_v1)

## Citation

Ferreira, M.G. and Dung, T. "Looking further: a crop type classification model for fields", Version 1.0, Radiant MLHub. [Date Accessed] Radiant MLHub <https://doi.org/10.34911/rdnt.iaki1s>

## License

[CC-BY-4.0](../LICENSE)

## Creators

- MG Ferreira - Ferra Solutions www.ferrasolutions.com [LinkedIn](https://www.linkedin.com/in/mg-ferreira-35534)
- Tien Dung

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

This will calculate descriptive statistics for each field as well as for a
number of concentric borders around it. It will also classify the terrain type
around the field. All of these features are then clustered, up-sampled and
finally fed into a random forest for classification.

## Training Operating System

- MacOS (darwin)

## Training Processor Type

- CPU

## Model Inferencing

Review the [GitHub repository README](../README.md) to get started running
this model for new inferencing.

## Methodology and Development

### Background

A year ago our model got top score in the similar "Spot the Crop" competition.
There you had to classify field crop types based on Sentinel S1 and S2 and also
based on just S2 imagery (thus two streams).

That time, we used a gradient booster and just a few derivative bands and it
was enough.

For this competition, we started there. However, it soon became apparent that
this was a lot more difficult. The imagery was a lot more homogeneous and it
became really difficult to discriminate between crop types. The previous
approach simply did not work well this time.

### New ideas

Even in the previous challenge, we felt that we were not making use of the
surroundings enough. If a river or mountain lay next to a field, surely it
would influence the crop type! So this time we tried to also use the "dark"
pixels - those pixels that are outside of any field's boundary.

### Field borders

We did this by creating a thick border around each field, and then a border
around that border and so on. Each of these concentric borders were then
evaluated and added to the list of features for each field.

### Terrain classification

We also used a pre-trained model to classify the terrain to the north, east,
south and west of each field. We also applied the classification to the field
itself to try and squeeze a bit extra out of the information.

### Field distributions

On the very last day of the competition, we also added the distribution of crop
types of fields in the vicinity of the field being classified. This is not a
new idea, we used it in the "Spot the Crop" challenge, and here it did add a
bit to our score.

### Clusters

One of the best improvements we got in our score was when we started to cluster
the data and then to pass the clusters on as features to the model. This made a
huge difference to our position on the leaderboard and was one of our big
breakthroughs for this competition. I think the reason for this is that we used
a lot of derived bands and our features were very correlated, so that the
clustering helped reduce that and helped the model fit better.

### Oversampling

The dataset is highly imbalanced and we got a bit of an improvement in score when we oversampled the data to get better class representation.

### Model

This time around, a gradient booster did not work as well, and we ended up
using a random forest. The reason for this also, I think, is the correlation
between the features given to the model.

The model itself is a random forest. A gradient booster does not work as well
because the features are highly correlated.

### Training

The data is processed first. For each band and each field, a number of
descriptive statistics such as mean, median, standard deviation and so on are
calculated. Then a number of concentric borders are drawn around the field of
varying thickness. The bands in each border is analyzed in the same manner and
contributes more descriptive statistic features. A pre-trained EarthNet model
is also used to classify the terrain to the north, east, west and south of the
field and this classification is added to the features. Next the features are
clustered and then up-sampled to be more balanced.

### Structure of Output Data

Since a field can span multiple tiles, two outputs are prepared.

- A normal CSV file where the crop type probability prediction is averaged across tiles for each field.
- A CSV file ending in `-w.csv` where the probabilities are weighted according to the size of each field in each tile (this is the output used in the competition).

### Full Solution Codes

See also the [full_solution/](../full_solution/README.md) folder, which
documents the ensemble model used in the
[Zindi AgriFieldNet India Challenge](https://zindi.africa/competitions/agrifieldnet-india-challenge).
