import matplotlib.pyplot as plt
import preprocessing

class TimeSeriesVisualizer:
    def __init__(self, df):
        self.df = df
        # normal condition
        self.plot_data_normal = df[(df['timestamp'] >= "2022-01-01 06:00:00") & (df['timestamp'] <= "2022-01-01 08:00:00")]
        # air leak 1 (lasted approximately 4 hours)
        self.plot_data_failure_one = df[(df['timestamp'] >= "2022-02-28 20:00:00") & (df['timestamp'] <= "2022-03-01 00:00:00")]
        # air leak 2 (lasted 1 hour)
        self.plot_data_failure_two = df[(df['timestamp'] >= "2022-03-23 13:00:00") & (df['timestamp'] <= "2022-03-23 17:00:00")]
        # oil leak (lasted approximately 42 hours)
        self.plot_data_failure_three = df[(df['timestamp'] >= "2022-05-30 06:00:00") & (df['timestamp'] <= "2022-05-31 12:00:00")]

    def plot_analog(self, plot_data):
        plt.subplot(4, 2, 1)
        plt.plot(plot_data['timestamp'], plot_data['TP2'])
        plt.title("TP2")

        plt.subplot(4, 2, 2)
        plt.plot(plot_data['timestamp'], plot_data['TP3'])
        plt.title("TP3")

        plt.subplot(4, 2, 3)
        plt.plot(plot_data['timestamp'], plot_data['H1'])
        plt.title("H1")

        plt.subplot(4, 2, 4)
        plt.plot(plot_data['timestamp'], plot_data['DV_pressure'])
        plt.title("DV_pressure")

        plt.subplot(4, 2, 5)
        plt.plot(plot_data['timestamp'], plot_data['Reservoirs'])
        plt.title("Reservoirs")

        plt.subplot(4, 2, 6)
        plt.plot(plot_data['timestamp'], plot_data['Oil_temperature'])
        plt.title("Oil_temperature")

        plt.subplot(4, 2, 7)
        plt.plot(plot_data['timestamp'], plot_data['Flowmeter'])
        plt.title("Flowmeter")

        plt.subplot(4, 2, 8)
        plt.plot(plot_data['timestamp'], plot_data['Motor_current'])
        plt.title("Motor_current")

        plt.tight_layout()
        plt.show()

    def plot_digital(self, plot_data):
        plt.subplot(4, 2, 1)
        plt.plot(plot_data['timestamp'], plot_data['COMP'])
        plt.title("COMP")

        plt.subplot(4, 2, 2)
        plt.plot(plot_data['timestamp'], plot_data['DV_eletric'])
        plt.title("DV_electric")

        plt.subplot(4, 2, 3)
        plt.plot(plot_data['timestamp'], plot_data['Towers'])
        plt.title("Towers")

        plt.subplot(4, 2, 4)
        plt.plot(plot_data['timestamp'], plot_data['MPG'])
        plt.title("MPG")

        plt.subplot(4, 2, 5)
        plt.plot(plot_data['timestamp'], plot_data['LPS'])
        plt.title("LPS")
        plt.ylim(-1, 1)

        plt.subplot(4, 2, 6)
        plt.plot(plot_data['timestamp'], plot_data['Pressure_switch'])
        plt.title("Pressure_switch")
        plt.ylim(-1, 1)

        plt.subplot(4, 2, 7)
        plt.plot(plot_data['timestamp'], plot_data['Oil_level'])
        plt.title("Oil_level")
        plt.ylim(-1, 1)

        plt.subplot(4, 2, 8)
        plt.plot(plot_data['timestamp'], plot_data['Caudal_impulses'])
        plt.title("Caudal_impulses")

        plt.tight_layout()
        plt.show()

def main(plot_type):
    preprocessor = preprocessing.DataPreprocessor("C:/Users/SA_009/Documents/dataset_train_processed.csv") # change file path as required
    visualizer = TimeSeriesVisualizer(preprocessor.df)
    if plot_type == "analog":
        # visualizer.plot_analog(visualizer.plot_data_normal)
        # visualizer.plot_analog(visualizer.plot_data_failure_one)
        # visualizer.plot_analog(visualizer.plot_data_failure_two)
        visualizer.plot_analog(visualizer.plot_data_failure_three)
    if plot_type == "digital":
        visualizer.plot_digital(visualizer.plot_data_normal)
        # visualizer.plot_digital(visualizer.plot_data_failure_one)
        # visualizer.plot_digital(visualizer.plot_data_failure_two)
        # visualizer.plot_digital(visualizer.plot_data_failure_three)

if __name__ == "__main__":
    #plot_type = "digital"
    plot_type = "analog"
    main(plot_type)
