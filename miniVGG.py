import tensorflow as tf 
import numpy as np 


class MiniVGGNet: 

    @staticmethod 
    def build(width, height, depth, classes): 
        inputs = tf.keras.Input(shape=(width, height, depth)) 
        x = inputs
        # model 
        filters = [32, 64] 

        for f in filters: 
            x = MiniVGGNet._conv_block(x, f, 3, "maxpooling")
        
        # fc 
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x, training=True)
        
        # output 
        outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)

        # return model 
        model = tf.keras.Model(inputs=inputs, outputs=outputs) 
        return model

    @staticmethod
    def _conv_block(x, filters, kernel_size, pooling): 
        '''
            conv => act => bn => conv => act => bn => pool => dropout
        '''

        x = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, padding="same", activation="relu"
            )(x) 
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, padding="same", activation="relu"
            )(x) 
        x = tf.keras.layers.BatchNormalization()(x)

        # pooling 
        if pooling == "maxpooling": 
            x = tf.keras.layers.MaxPooling2D(padding="same")(x)
        elif pooling == "averagepooling": 
            x = tf.keras.layers.AveragePooling2D(padding="same")(x) 
        
        x = tf.keras.layers.Dropout(0.2)(x, training=True)

        return x
         
