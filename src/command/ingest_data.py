from data import ingest_data
import click
from pathlib import Path
import os


@click.command()
def cli():
    """
    ingest new data from training
    """
    root_dir = Path(os.environ['ROOT_DIR'])
    raw_path = root_dir.joinpath('raw')
    output_path = root_dir.joinpath('processed')

    assert raw_path.exists(), \
           f'raw train path do not exists {raw_path}'
    train_data = ingest_data.ingest_train_data(raw_path / 'train_updated.csv',
                                               path_to_players_csv=raw_path / 'players.csv',
                                               path_to_season_csv=raw_path / 'seasons.csv')

    output_path.mkdir(exist_ok=True, parents=True)
    output_file = output_path / 'raw_data.csv'
    print(f'saving file {output_file}')
    train_data.to_csv(output_file, index=False)