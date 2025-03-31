import time  # Import the time module
import pre_processing.s3_download as s3_download
import src.pre_processing.split_db as split_db
import runpy
from sqlalchemy import create_engine


#db does not exist yet
database = 'src/pre_processing/stock_data/stock_data.db'
engine = create_engine(f'sqlite:///{database}')


if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    
    s3_download.download_data()
    
    split_db.process_and_insert_files(engine)
    split_db.process_database(engine)

    runpy.run_path('src/models/random_forest.ipynb')

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Script completed in {elapsed_time:.2f} seconds.")