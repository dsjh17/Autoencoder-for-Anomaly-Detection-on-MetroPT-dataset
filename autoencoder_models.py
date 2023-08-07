import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import preprocessing
import keras.backend as K
from tensorflow import keras
from keras.models import Model, save_model
from keras.layers import Conv1D, Conv1DTranspose, Lambda, Reshape, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from tensorflow.python.framework.ops import disable_eager_execution

# Darren Sim Jun Hao

disable_eager_execution() # variational autoencoder will not work without this
# an alternative is to replace all instances of keras with tf.keras
class AutoencoderModels:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = None
        self.hist = None

    def leaky_relu(self, x):  # to be used as the activation function of the output layer for the analog models; greatly reduces the loss value
        return tf.nn.leaky_relu(x, alpha=0.01)

    def sae_analog(self):  # both analog and digital sensors have 8 features/columns
        # Encoder portion
        input_layer = keras.Input(shape=(1, self.train_data.shape[1])) # shape of input is (no. of timesteps = 1, no. of features = 8) - same for all models
        hidden_1 = Conv1D(filters=32, padding="same", kernel_size=3, activation='relu')(input_layer)
        hidden_2 = Conv1D(filters=16, padding="same", kernel_size=3, activation='relu')(hidden_1)
        bottleneck = Conv1D(filters=6, padding="same", kernel_size=3, activation="relu", activity_regularizer=keras.regularizers.l1_l2(l1=0.005, l2=0.1))(hidden_2)
        #  activity_regularizer=keras.regularizers.l1_l2(l1=0.005, l2=0.1)
        # Decoder portion
        hidden_3 = Conv1DTranspose(filters=16, padding="same", kernel_size=3, activation='relu')(bottleneck)
        hidden_4 = Conv1DTranspose(filters=32, padding="same", kernel_size=3, activation='relu')(hidden_3)
        output_layer = Conv1DTranspose(filters=self.train_data.shape[1], padding="same", kernel_size=3,
                              activation=self.leaky_relu)(
            hidden_4)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse') # loss function can also be set to RMSE, MAE etc. for parameter tuning
        callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        hist = model.fit(np.expand_dims(self.train_data, 1), np.expand_dims(self.train_data, 1), validation_split=0.2,
                         epochs=50, callbacks=[callback], batch_size=60)
        save_model(model, "C:/Users/SA_009/Documents/sae_analog.h5")
        self.model = model
        self.hist = hist

    def sae_digital(self):
        # Encoder portion
        input_layer = keras.Input(shape=(1, self.train_data.shape[1]))
        hidden_1 = Conv1D(filters=32, padding="same", kernel_size=3, activation='relu')(input_layer)
        hidden_2 = Conv1D(filters=16, padding="same", kernel_size=3, activation='relu')(hidden_1)
        bottleneck = Conv1D(filters=6, padding="same", kernel_size=3, activation="relu")(hidden_2)

        # Decoder Portion
        hidden_3 = Conv1DTranspose(filters=16, padding="same", kernel_size=3, activation='relu')(bottleneck)
        hidden_4 = Conv1DTranspose(filters=32, padding="same", kernel_size=3, activation='relu')(hidden_3)
        output_layer = Conv1DTranspose(filters=self.train_data.shape[1], padding="same", kernel_size=3, activation="sigmoid")(
            hidden_4)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        hist = model.fit(np.expand_dims(self.train_data, 1), np.expand_dims(self.train_data, 1), validation_split=0.2,
                         epochs=50, callbacks=[callback], batch_size=60)
        save_model(model, "C:/Users/SA_009/Documents/sae_digital.h5")
        self.model = model
        self.hist = hist

    def vae_analog(self):

        def sampling(args):  # reparameterization trick used in VAE to sample data points from the latent space
            z_mean, z_log_var = args  # log variance instead of variance is better for data that is not assumed to have a normal distribution
            batch_size = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch_size, dim)) # is a normal distribution with mean 0, variance 1
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        def vae_loss(x_actual, x_decoded):
            reconstruction_loss = K.mean(K.square(x_actual - x_decoded)) # measures the ability of the VAE to reconstruct the input data accurately
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) # measures the difference between the learned latent distribution and the distribution of the data
            return reconstruction_loss + kl_loss

        # Encoder portion
        input_layer = keras.Input(shape=(1, self.train_data.shape[1]))
        hidden_1 = Conv1D(filters=32, padding="same", kernel_size=3, activation='relu')(input_layer)
        hidden_2 = Conv1D(filters=16, padding="same", kernel_size=3, activation='relu')(hidden_1)
        z_mean = Conv1D(filters=6, padding="same", kernel_size=3, activation="linear")(hidden_2)
        z_log_var = Conv1D(filters=6, padding="same", kernel_size=3, activation="linear")(hidden_2)
        z = Lambda(sampling)([z_mean, z_log_var])  # this layer is the lower-dimensional representation of the input data learned from the sampling method i.e. the latent space
        # the combination of z_mean z_log_var and the sampling layer can be thought of as similar to the bottleneck layer in a traditional autoencoder

        # Decoder portion
        hidden_3 = Conv1DTranspose(filters=16, padding="same", kernel_size=3, activation='relu')(z)
        hidden_4 = Conv1DTranspose(filters=32, padding="same", kernel_size=3, activation='relu')(hidden_3)
        output_layer = Conv1DTranspose(filters=self.train_data.shape[1], padding="same", kernel_size=3,
                              activation=self.leaky_relu)(hidden_4)
        output_layer = GlobalAveragePooling1D()(output_layer)
        output_layer = Reshape((1, self.train_data.shape[1]))(output_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss=vae_loss)
        callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        hist = model.fit(np.expand_dims(self.train_data, 1), np.expand_dims(self.train_data, 1), validation_split=0.2,
                         epochs=1, callbacks=[callback], batch_size=60)
        save_model(model,"C:/Users/SA_009/Documents/vae_analog.h5")
        self.model = model
        self.hist = hist

    def vae_digital(self): # exact same architecture as vae_analog

        def sampling(args):
            z_mean, z_log_var = args
            batch_size = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch_size, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        def vae_loss(x_actual, x_decoded):
            reconstruction_loss = K.mean(K.square(x_actual - x_decoded)) # measures the ability of the VAE to reconstruct the input data accurately
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) # measures the difference between the learned latent distribution and the distribution of the data
            return reconstruction_loss + kl_loss

        # Encoder portion
        input_layer = keras.Input(shape=(1, self.train_data.shape[1]))
        hidden_1 = Conv1D(filters=32, padding="same", kernel_size=3, activation='relu')(input_layer)
        hidden_2 = Conv1D(filters=16, padding="same", kernel_size=3, activation='relu')(hidden_1)
        z_mean = Conv1D(filters=6, padding="same", kernel_size=3, activation="linear")(hidden_2)
        z_log_var = Conv1D(filters=6, padding="same", kernel_size=3, activation="linear")(hidden_2)
        z = Lambda(sampling)([z_mean, z_log_var])

        # Decoder portion
        hidden_3 = Conv1DTranspose(filters=16, padding="same", kernel_size=3, activation='relu')(z)
        hidden_4 = Conv1DTranspose(filters=32, padding="same", kernel_size=3, activation='relu')(hidden_3)
        output_layer = Conv1D(filters=self.train_data.shape[1], padding="same", kernel_size=3,
                              activation="sigmoid")(hidden_4)
        output_layer = GlobalAveragePooling1D()(output_layer)
        output_layer = Reshape((1, self.train_data.shape[1]))(output_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss=vae_loss)
        callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        hist = model.fit(np.expand_dims(self.train_data, 1), np.expand_dims(self.train_data, 1), validation_split=0.2,
                         epochs=50, callbacks=[callback], batch_size=60)
        save_model(model, "C:/Users/SA_009/Documents/vae_digital.h5")
        self.model = model
        self.hist = hist

    def autoencoder_predict(self):
        train_pred = self.model.predict(np.expand_dims(self.train_data, 1))
        train_pred = np.reshape(train_pred, (len(train_pred), 8))
        test_pred = self.model.predict(np.expand_dims(self.test_data, 1))
        test_pred = np.reshape(test_pred, (len(test_pred), 8))
        return train_pred, test_pred

    def plot_losses(self):
        plt.plot(self.hist.history["loss"])
        plt.plot(self.hist.history["val_loss"])
        plt.title("Training/Validation Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["training_loss", "validation_loss"])
        plt.show()

def main():
    preprocessor = preprocessing.DataPreprocessor("C:/Users/SA_009/Documents/dataset_train_processed.csv") # change according to where your data is stored
    preprocessor.preprocessing_autoencoder()
    autoencoder = AutoencoderModels(preprocessor.digital_train, preprocessor.digital_test) # comment out depending on whether digital/analog data is used
    # autoencoder = AutoencoderModels(preprocessor.analog_train, preprocessor.analog_test)
    # autoencoder.sae_analog() # comment out according to which method you call from the AutoencoderModels class
    autoencoder.vae_digital()
    # autoencoder.vae_analog()
    # autoencoder.vae_digital()
    train_pred, test_pred = autoencoder.autoencoder_predict()
    with open("C:/Users/SA_009/Documents/vae_digital_train_pred", "wb") as file: # name of file path can change depending on where you want to save it
        pickle.dump(train_pred, file)
    with open("C:/Users/SA_009/Documents/vae_digital_test_pred", "wb") as file: # similarly, this file path can be changed
        pickle.dump(test_pred, file)
    print(autoencoder.model.summary())
    autoencoder.plot_losses()

if __name__ == "__main__":
    main()
