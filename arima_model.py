import pandas as pd
import pmdarima as pm
import preprocessing
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# pmd autoarima is not effective on this data; sarimax from statsmodels should be used but the correct parameters need to be identified

class ArimaModel:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def arima_predict(self):
        model = pm.auto_arima(self.train_data, trace=True, max_D=1, max_order=None, stepwise=False)
        forecast = pd.DataFrame(model.predict(n_periods=len(self.test_data)))
        # model = sm.sarimax.SARIMAX(self.train_data["TP2"],order=(12,1,12,2)).fit() # need to figure out the correct parmas for seasonal order/order
        # forecast = model.predict(start=self.test_data.index[0],end=self.test_data.index[-1])

        plt.plot(self.test_data.index, self.test_data, label="Actual", color="blue")
        plt.plot(self.test_data.index, forecast, label="Forecast", color="red")
        plt.legend()
        plt.show()

    def plot_correlation(self): # plot the autocorrelation and partial autocorrelation functions, but recommended to resample the data before calling this method
        fig, ax = plt.subplots(2, 1)
        plot_acf(self.train_data["TP2"], lags=60, ax=ax[0])
        plot_pacf(self.train_data["TP2"], lags=60, ax=ax[1])
        plt.show()

def main():
    preprocessor = preprocessing.DataPreprocessor("C:/Users/SA_009/Documents/dataset_train_processed.csv")
    preprocessor.preprocessing_arima()
    model = ArimaModel(preprocessor.train_data, preprocessor.test_data)
    model.arima_predict()
    # model.plot_correlation()

if __name__=="__main__":
    main()


