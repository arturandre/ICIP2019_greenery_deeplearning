from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization
from keras.layers import Dropout, Conv2DTranspose
from keras.layers import add
from keras import Model

# encoder-decoder-skip implementation
# Based on: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/Encoder_Decoder.py
# And: https://arxiv.org/abs/1511.00561

def conv_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0, padding='same'):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    Dropout (if dropout_p > 0) on the inputs
    """
    conv = Convolution2D(n_filters, kernel_size, padding='same')(inputs)
    out = Activation('relu')(conv)
    out = BatchNormalization()(out)
    
    if dropout_p != 0.0:
        out = Dropout(rate=(1.0-dropout_p))(out)
    return out

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0, padding='same'):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    Dropout (if dropout_p > 0) on the inputs
    """
    conv = Conv2DTranspose(n_filters, kernel_size=kernel_size, strides=(2, 2), padding=padding)(inputs)
    out = Activation('relu')(conv)
    out = BatchNormalization()(out)

    if dropout_p != 0.0:
        out = Dropout(rate=(1.0-dropout_p))(out)
    return out

def build_encoder_decoder(inputs, num_classes, preset_model = "Encoder-Decoder", dropout_p=0.5, scope=None, maxPadding='valid'):
    """
    Builds the Encoder-Decoder model. Inspired by SegNet with some modifications
    Optionally includes skip connections
    Arguments:
      inputs: the input tensor
      n_classes: number of classes
      dropout_p: dropout rate applied after each convolution (0. for not using)
    Returns:
      Encoder-Decoder model
    """


    if preset_model == "Encoder-Decoder":
        has_skip = False
    elif preset_model == "Encoder-Decoder-Skip":
        has_skip = True
    else:
        raise ValueError("Unsupported Encoder-Decoder model '%s'. This function only supports Encoder-Decoder and Encoder-Decoder-Skip" % (preset_model))

    #####################
    # Downsampling path #
    #####################
    net = conv_block(inputs, 64)
    net = conv_block(net, 64)
    net = MaxPooling2D(padding=maxPadding)(net)
    skip_1 = net

    net = conv_block(net, 128)
    net = conv_block(net, 128)
    net = MaxPooling2D(padding=maxPadding)(net)
    skip_2 = net

    net = conv_block(net, 256)
    net = conv_block(net, 256)
    net = conv_block(net, 256)
    net = MaxPooling2D(padding=maxPadding)(net)
    skip_3 = net

    net = conv_block(net, 512)
    net = conv_block(net, 512)
    net = conv_block(net, 512)
    net = MaxPooling2D(padding=maxPadding)(net)
    skip_4 = net

    net = conv_block(net, 512)
    net = conv_block(net, 512)
    net = conv_block(net, 512)
    net = MaxPooling2D(padding=maxPadding)(net)


    #####################
    # Upsampling path #
    #####################
    net = conv_transpose_block(net, 512, kernel_size=[3,1], padding='valid')
    net = conv_block(net, 512)
    net = conv_block(net, 512)
    net = conv_block(net, 512)
    if has_skip:
        net = add([net, skip_4])

    net = conv_transpose_block(net, 512)
    net = conv_block(net, 512)
    net = conv_block(net, 512)
    net = conv_block(net, 256)
    if has_skip:
        net = add([net, skip_3])

    net = conv_transpose_block(net, 256)
    net = conv_block(net, 256)
    net = conv_block(net, 256)
    net = conv_block(net, 128)
    if has_skip:
        net = add([net, skip_2])

    net = conv_transpose_block(net, 128)
    net = conv_block(net, 128)
    net = conv_block(net, 64)
    if has_skip:
        net = add([net, skip_1])

    net = conv_transpose_block(net, 64)
    net = conv_block(net, 64)
    net = conv_block(net, 64)

    #####################
    #      Softmax      #
    #####################
    net = Convolution2D(filters=num_classes, kernel_size=[1, 1])(net)
    net = Activation('softmax')(net)
    
    return Model(inputs=inputs, outputs=net)
