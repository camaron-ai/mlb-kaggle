ingest-data:
	python src/run.py ingest_data
create-notebook:
	python src/build_nb.py --output_file notebook.ipynb
