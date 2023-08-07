import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.train_data = None
        self.test_data = None
        self.analog_train = None
        self.analog_test = None
        self.digital_train = None
        self.digital_test = None

    def preprocessing_df(self):
        # create a new column where 1 day before the timeframe as well as the timeframe indicated in the paper is labeled as unhealthy data i.e. 1
        # healthy data is set as 0
        self.df['is_anomaly'] = np.where(
            ((self.df['timestamp'] >= "2022-02-14 06:00:00") & (self.df['timestamp'] <= "2022-03-01 02:00:00")) |
            ((self.df['timestamp'] >= "2022-03-08 06:00:00") & (self.df['timestamp'] <= "2022-03-23 15:24:00")) |
            ((self.df['timestamp'] >= "2022-05-16 06:00:00") & (self.df['timestamp'] <= "2022-06-02 06:18:00")),
            1, 0
        )
        self.data_path = "C:/Users/SA_009/Documents/dataset_train_processed.csv" # change this path accordingly when you want to change the file location
        self.df.to_csv(self.data_path)
        self.df = pd.read_csv(self.data_path)

    def preprocessing_autoencoder(self):
        # self.train_data = self.df[
        #     (self.df['timestamp'] >= "2022-01-01 06:00:00") & (self.df['timestamp'] <= "2022-02-28 02:00:00")]
        # self.test_data = self.df[
        #     (self.df['timestamp'] >= "2022-02-28 06:00:00")]
        self.train_data = self.df[
            (self.df['timestamp'] >= "2022-01-01 06:00:00") & (self.df['timestamp'] < "2022-02-14 06:00:00")]
        self.test_data = self.df[
            (self.df['timestamp'] >= "2022-02-14 06:00:00")]
        self.train_data.drop(self.train_data.columns[0], axis=1,
                             inplace=True)  # there is an additional unnecessary column created at index 0 from preprocessing.df that needs to be removed if preprocessing_df is called
        self.test_data.drop(self.test_data.columns[0], axis=1, inplace=True)
        scaler = StandardScaler() # standardising only the analog data, leaving the digital data as is
        self.analog_train = pd.DataFrame(scaler.fit_transform(self.train_data.iloc[:, 1:9]))
        self.digital_train = self.train_data.iloc[:, 9:17]
        self.analog_test = pd.DataFrame(scaler.transform(self.test_data.iloc[:, 1:9]))
        self.digital_test = self.test_data.iloc[:, 9:17]

    def preprocessing_arima(self):
        df = self.df
        columns = ["timestamp", "TP2"]
        df = df[columns]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.resample("5T", on="timestamp").mean() # data can also be resampled to other timeframes: refer to pandas documentation for more info
        df = df.fillna(method="ffill")
        self.train_data = df[(df.index >= pd.to_datetime("2022-01-01 06:00:00")) & (df.index <= pd.to_datetime("2022-02-28 02:00:00"))]
        self.test_data = df[(df.index >= pd.to_datetime("2022-02-28 06:00:00")) & (df.index <= pd.to_datetime("2022-03-02 02:00:00"))] # change accordingly depending on your forecast timeframe

    def preprocessing_ma(self):
        columns = ["timestamp", "TP2"]
        self.df = self.df[columns]
        self.train_data = self.df[
            (self.df['timestamp'] >= "2022-01-01 06:00:00") & (self.df['timestamp'] <= "2022-02-28 02:00:00")]
        self.test_data = self.df[
            (self.df['timestamp'] >= "2022-02-28 06:00:00")]

def main():
    preprocessor = DataPreprocessor("C:/Users/SA_009/Documents/dataset_train.csv") # change this to the file path of the train dataset
    preprocessor.preprocessing_df()

if __name__ == "__main__":
    main()
