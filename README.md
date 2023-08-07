# What is this project about?

From an operational metro train in Porto, Portugal, readings from analog sensors (pressure, temperature, and electric current), digital signals (control signals, discrete signals) and GPS information (latitude, longitude, and speed) were collected from a compressor's Air Production Unit (APU). The APU is a highly important element of the vehicle throughout the day and its failure is likely to result in the immediate removal of the train for repair. The failures are typically undetectable according to traditional condition-based maintenance criteria
(predefined thresholds). 

The MetroPT dataset is a real-world dataset where the ground truth of anomalies is revealed in the company's maintenance reports. The data was collected in 2022 and spans from 1 January 2022 till 2 June 2022 with 10773588 instances and 21 columns. It supports the development of predictive maintenance, anomaly detection, and remaining useful life prediction models using machine learning methods.


# How to run this project?

There are 5 different python files in this project, arranged in the order you should run them in:

1. preprocessing.py
- Processes the data to the format used in the respective autoencoder and arima models 
- Only needs to be run once to update the original dataset with a new column of labels as detailed by the ground truth

2. line_plots.py (optional)
- For visualizing the various analog and digital sensors operating during a normal time frame, as well as during the periods of failure

3. autoencoder_model.py
- 4 models are found here: SAE_digital, SAE_analog, VAE_digital, VAE_analog
- model architecture, plots of training/validation loss, and model predictions are available in this file
- model (h5 file) and its respective predictions (pickle file) can be saved in your desired file paths
- if done correctly, you should obtain 4 autoencoder models and 8 predictions, two for each model, as shown here:
![image](https://github.com/dsjh17/MetroPT/blob/main/image.png)

4. autoencoder_results.py
- calculation of reconstruction_error (training data), training_error, as well as threshold
- confusion matrix detailing precision, recall and f1 score can be found here

5. arima_model.py (optional)
- Not successful in forecasting future trends - likely that the parameters used in the model are incorrect

Rule of thumb:
Auto-regressive(p) : Maximum lag for which correlation is significant (i.e. lies outside the confidence interval)
Differencing (d) : Refers to order of differencing/no. of times differencing is applied
Moving Average (q): Lag which did not significantly reduce correlation down to zero 
