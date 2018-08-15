import tensorflow as tf

def ResUnit3D(inputs, filters = 64, kernel_size = 3,
              strides = 1, bn=False, name=None):
    if name is None:
        name = ''
    with tf.name_scope("Residual_Unit_" + name) as res_unit:
        activation1 = tf.nn.relu(inputs, name='relu_1')
        conv1 = tf.layers.conv3d(
            inputs=activation1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name='conv_1'
        )
        activation2 = tf.nn.relu(conv1, name='relu_2')
        conv2 = tf.layers.conv3d(
            activation2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name='conv_2'
        )
        output = tf.add(inputs, conv2, name='add')
    return output
    

'''

def STResNet3D(
                c_conf=(3, 60, 20, 20),
                p_conf=(3, 60, 20, 20),
                t_conf=(3, 60, 20, 20),
                nb_resunit=3):
                
'''    