import time  # Import the time module
import pre_processing.s3_download as s3_download
import pre_processing.data_transform as data_transform
from sqlalchemy import create_engine, inspect

#db does not exist yet
database = 'src/pre_processing/stock_data/stock_data.db'
engine = create_engine(f'sqlite:///{database}')


if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    
    #s3_download.download_data()
    
    data_transform.process_and_insert_files(engine)
    data_transform.process_database(engine)
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Script completed in {elapsed_time:.2f} seconds.")