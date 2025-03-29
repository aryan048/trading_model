import pre_processing.s3_download as s3_download
import pre_processing.data_transform as data_transform

if __name__ == "__main__":
    s3_download.download_data()
    data_transform.process_and_insert_files()