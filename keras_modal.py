from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Layer, Conv2D, BatchNormalization, Input
from tensorflow.nn import relu
import tensorflow

# Doc:
# https://www.tensorflow.org/tutorials/customization/custom_layers
# https://www.tensorflow.org/guide/keras/custom_layers_and_models

class OutputLayer(Layer):
    """Returns 2 variables: the thruttle and the angle"""

    def __init__(self, out_dim, activation):
        super(OutputLayer, self).__init__()
        self.dense_thruttle = Dense(out_dim, activation=activation)
        self.dense_angle = Dense(out_dim, activation=activation)

    def call(self, x):
        thruttle = self.dense_thruttle(x)
        angle = self.dense_angle(x)

        return thruttle, angle


class DrivingModel(Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='driving',
                 **kwargs):
        super(DrivingModel, self).__init__(name=name, **kwargs)
        self.layer1 = Conv2D(1, (1, 1))
        self.layer2 = BatchNormalization()
        self.layer3 = Flatten()
        self.layer4 = Dense(16, activation='relu')
        self.layer5 = OutputLayer(output_dim, activation='linear')
        self.build(input_shape)


    def build(self, input_shape):
        # call it to build the graph of layer nodes to define an ibound and outbound node
        inputs = tensorflow.zeros(list(input_shape))
        # inputs = Input(shape=input_shape) # ValueError: Input 0 of layer conv2d is incompatible with the layer: expected ndim=4, found ndim=5. Full shape received: [None, 1, 120, 160, 3]
        print("Input shape:", input_shape)
        self.call(inputs)



    def call(self, inputs, training=False):
        x = self.layer1(inputs)
        x = self.layer2(x, training=training)
        x = relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


def get_model(input_shape, out_dim):
    model = DrivingModel(input_shape, out_dim)

    # print(model.summary())

    return model