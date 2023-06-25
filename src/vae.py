import logging
import tensorflow as tf
from tensorflow.keras import Model, optimizers, layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class VAE(tf.keras.Model):
    
    def __init__(self, input_dim, intermediate_dim, latent_dim, learning_rate, verbose=False, **kwargs):
        
        super(VAE, self).__init__(**kwargs)
        
        self.original_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.threshold = None

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model = self.build_vae()    
        
        self.logger = logging.getLogger(__name__)
        
    def build_encoder(self):
        
        inputs = layers.Input(shape=(self.original_dim,), name='encoder_input')
        x = layers.Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = layers.Lambda(self.sample, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
    def build_decoder(self):
        
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='z_sampling')
        x = layers.Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = layers.Dense(self.original_dim, activation='sigmoid')(x)
        return Model(latent_inputs, outputs, name='decoder')
    
    def build_vae(self):
        
        x = layers.Input(shape=(self.original_dim,), name='encoder_input')
        z_mean, z_log_var, z = self.encoder(x)
        
        x_decoded_mean = self.decoder(z)
        reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae = Model(x, x_decoded_mean)
        vae.add_loss(vae_loss)
                
        opt = optimizers.Adam(learning_rate=self.learning_rate)
        vae.compile(optimizer=opt)
        return vae    
    
    def sample(self, args):
        
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        #epsilon = K.ones_like(z_mean)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def fit(self, x_train, epochs, batch_size, validation_data=None, validation_split=None, verbose=0, save_dir=None):
        
        self.model.fit(x_train, 
                           None ,
                           shuffle=True,
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=validation_data,
                           validation_split=validation_split,
                           verbose=verbose
                      )
        if not (save_dir is None):
            
            "This block saves the model and weights as h5"            
            self.model.save(save_dir + '/' + self.name + '.h5')
            self.model.save_weights(save_dir + '/' + self.name + '-weights.h5') 
                    
        self.determine_classification_threshold(x_train)
        
    def load_model_weights(self, weights_path):
        
        self.model.load_weights(weights_path)
        
        if self.verbose:
            self.logger.info(f"Loaded model: {self.model.summary()}")
        

    def determine_classification_threshold(self, x_train):
        
        mae_train = self.calculate_reconstruction_error(x_train)
        self.threshold_max = np.max(mae_train)
        self.threshold = np.percentile(mae_train.values, 99)
        self.threshold_90 = np.percentile(mae_train.values, 90)
        
    def calculate_reconstruction_error(self, data):
                
        recon_data = self.model.predict(data)
        return np.mean(np.abs(data - recon_data), axis=1)
    
    def predict_anomaly(self, data):
        
        mae_data = self.calculate_reconstruction_error(data)
        
        pred = [1 if curr_mae > self.threshold else 0 for curr_mae in mae_data]
        
        return pred, mae_data
    
    
    def predict_anomaly_90(self, data):
        
        mae_data = self.calculate_reconstruction_error(data)
        
        pred = [1 if curr_mae > self.threshold_90 else 0 for curr_mae in mae_data]
        
        return pred, mae_data    