# Stock Trading ML Model

This repository contains a machine learning model that predicts stock price movements using historical stock price data. The model aims to predict short-term price movements for stocks in the S&P 500 index. It uses features like daily stock prices, volume, and various technical indicators.

## Requirements

Before you can run the code, make sure to install the following dependencies:

1. **Polygon.io Subscription**:
   - You need a Polygon.io subscription to access the S3 data. Visit [Polygon.io](https://polygon.io) to get your API access keys.

2. **MinIO Client**:
   - The MinIO client (`mc`) is used to download the historical stock data from the Polygon.io S3 storage.
   - Install MinIO using the following command:
     ```bash
     brew install minio/stable/mc
     ```

3. **Python Packages**:
   - Install the required Python dependencies using `pip` or `poetry`. For `pip`:
     ```bash
     pip install -r requirements.txt
     ```
   - Or, if using `poetry`:
     ```bash
     poetry install
     ```

## Setup

1. **Get Polygon.io Subscription**:
   - Sign up for Polygon.io and obtain your **API key** and **secret key**.

2. **Configure MinIO Client**:
   - Run the following command to set up the MinIO client:
     ```bash
     mc alias set s3polygon https://files.polygon.io <your-access-key> <your-secret-key>
     ```
   - Replace `<your-access-key>` and `<your-secret-key>` with the credentials you obtained from Polygon.io.

3. **Configure SMS**:
   - Set phone number in .env file to be notified upon data dump completion
   - Note: You must be running this on a mac for the sms notifier to work

3. **Download Data**:
   - Use the `mc` command to download specific stock data. For example:
     ```bash
     mc cp s3polygon/flatfiles/us_stocks_sip/day_aggs_v1/2024/12/2024-12-31.csv.gz ~/Downloads/
     ```
   - This will download a specific file for December 31, 2024, into the `Downloads` directory.
   - Alternatively, run main.py to start downloading data beginning 5 years ago, up until now

4. **Resolve pandas_ta error**
   - The squeeze_pro file has an incorrect import, head to this directory
   '''bash
   /Users/aryanhazra/Library/Caches/pypoetry/virtualenvs/trading-model-vnO__Cft-py3.13/lib/python3.13/site-packages/pandas_ta/momentum
   '''bash
   - and change 
   '''bash
   from numpy import NaN as npNaN
   '''bash
   - to
   '''bash
   from numpy import nan as npNaN
   '''bash

## Running the Model

Once the data is downloaded and stored locally, you can begin training the model. Here’s how to run the prediction script:

```bash
python run_model.py