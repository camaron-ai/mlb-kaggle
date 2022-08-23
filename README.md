# MLB Kaggle Competition

A repository for kaggle MLB Competition

download data from <https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting>

create a .env file with the ROOT_DIR variable pointing to the folder where the data is stored. Example: DATA_ROOT=data/

place the data into ROOT_DIR/raw/ directory and unzip it.

finally, do:

```bash
make ingest-data
```
