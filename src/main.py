import time  # Import the time module
import pre_processing.s3_download as s3_download
import pre_processing.data_transform as data_transform
#import pre_processing.reformat_stock_splits as reformat_stock_splits

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    
    s3_download.download_data()
    
    data_transform.process_and_insert_files()
    data_transform.split_existing_database()

    #reformat_stock_splits.reformat_stock_splits()
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Script completed in {elapsed_time:.2f} seconds.")