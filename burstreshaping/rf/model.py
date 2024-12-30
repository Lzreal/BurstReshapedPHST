import keras
from keras.layers import Conv2D, MaxPool2D , Dropout, BatchNormalization, ReLU,Reshape,Conv1D,MaxPool1D,GlobalAveragePooling1D
from keras.models import Sequential
from keras.initializers import glorot_uniform
from keras.layers.core import Activation, Flatten, Dense, Dropout

class RF_Model:
    @staticmethod
    def build(input_shape,classes):
        model = Sequential()
        model.add(Conv2D(filters=32,kernel_size=(2,20), strides=1, padding="same" , input_shape=input_shape))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(Conv2D(filters=32,kernel_size=(2,20), strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(MaxPool2D(pool_size=(1,3)))
        model.add(Dropout(0.1))
        
        model.add(Conv2D(filters=64,kernel_size=(2,20), strides=1, padding="same" ))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(Conv2D(filters=64,kernel_size=(2,20), strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.1))
        
        # 
        model.add(Reshape((-1,32)))
        model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(MaxPool1D(pool_size=3))
        model.add(Dropout(0.3))
        
        model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(MaxPool1D(pool_size=3))
        model.add(Dropout(0.3))
        
        model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(Conv1D(filters=classes, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.1,epsilon=1e-05))
        model.add(ReLU())
        model.add(MaxPool1D(pool_size=3))
        model.add(Dropout(0.3))
        
        # model.add(GlobalAveragePooling1D())
        model.add(Flatten())
     
        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=1), name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc1_act'))

        model.add(Dropout(0.7, name='fc1_dropout'))

        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=1), name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc2_act'))

        model.add(Dropout(0.5, name='fc2_dropout'))

        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=1), name='fc3'))
        model.add(Activation('softmax', name="softmax"))
        
        # print(model.summary())
        return model
    

if __name__=="__main__":
    print(RF_Model().build_model(input_shape=(2,5000,1),classes = 103).summary())