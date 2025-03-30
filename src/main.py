import time  # Import the time module
import pre_processing.s3_download as s3_download
import pre_processing.data_transform as data_transform
import pre_processing.reformat_stock_splits as reformat_stock_splits
import pre_processing.technical_indicators as technical_indicators
from sqlalchemy import create_engine, inspect

#db does not exist yet
database = 'src/pre_processing/stock_data/stock_data.db'
engine = create_engine(f'sqlite:///{database}')
inspector = inspect(engine)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    
    s3_download.download_data()
    
    data_transform.process_and_insert_files(engine)
    data_transform.split_existing_database(engine)

    table_names = inspector.get_table_names()
    reformat_stock_splits.reformat_stock_splits(table_names, engine)

    technical_indicators.create_technical_indicators(engine,table_names)
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Script completed in {elapsed_time:.2f} seconds.")