from pathlib import Path
from typing import List
from typing import Optional
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbconvert import get_exporter
import nbformat
import os
import click


current_dir = os.path.dirname(__file__)

def nbuild(filenames: List[str]) -> nbformat.notebooknode.NotebookNode:
    """Create a Jupyter notebook from text files and Python scripts."""
    def read_code(file):
        raw_code = Path(file).read_text()
        lines = raw_code.split('\n')
        lines = [line for line in lines if not line.endswith('###no')]

        lines = [f'### {file}'] + lines
        code = '\n'.join(lines)
        return code


    nb = new_notebook()

    filenames = [os.path.join(current_dir, name)
                 for name in filenames]
    nb.cells = [
        # Create new code cells from files that end in .py
        new_code_cell(read_code(name)) 
        if name.endswith(".py")
        # Create new markdown cells from all other files
        else new_markdown_cell(Path(name).read_text()) 
        for name in filenames
    ]
    return nb


def submodule_files(submodule):
    return sorted([os.path.join(submodule, name)
                    for name in os.listdir(os.path.join(current_dir, submodule))
                    if name.endswith(".py")])

pipeline_modules = submodule_files('pipeline')
data_modules = submodule_files('data')
ingest_data_modules = submodule_files('data/ingest_data')
models_modules = submodule_files('models')
train_modules = submodule_files('train')
modules = (pipeline_modules + data_modules + ingest_data_modules +
           models_modules + train_modules)


@click.command()
@click.option('--output_file', type=str)
def cli(output_file: str):
    print(modules)
    nb = nbuild(modules)
    nbformat.write(nb, output_file)


if __name__ == '__main__':
    cli()

