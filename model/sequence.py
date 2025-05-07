import tensorflow as tf
from keras import layers, models

ckpt_dir = r'C:\Users\Benson\OPP\ML\checkpoint\sequence'

def network(input_size):

    input = tf.keras.layers.Input(input_size)
    
    # Feature extraction
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(input)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)


    output = tf.keras.layers.Dense(40, activation='softmax')(x)

    model = tf.keras.models.Model(inputs = input,  
                                outputs = output,  
                                name = 'Sequence') 

    return model

#RF in 500x500 : 40x40


def forward(model, x):
    prediction =  model.predict(x)
    return prediction
# probability of object 



def save_weights(model):
    model.save_weights(ckpt_dir)


def load_weights(model):
    model.load_weights(ckpt_dir)