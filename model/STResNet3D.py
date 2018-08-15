import tensorflow as tf

def _shortcut(input, residual):
    return tf.add(input, residual)

def _bn_relu_conv3d(inputs,
                    filters,
                    kernel_size,
                    strides,
                    bn=False,
                    name=None):
    if name is None:
        name = ''
    sess = tf.Session()
    with tf.name_scope("Residual_Unit_" + name) as res_unit:
    
        print('input\n', sess.run(inputs)[-1][-1][-1])
        activation1 = tf.nn.relu(inputs, name='activation_1')
        print('av1', sess.run(activation1)[-1][-1][-1])
        print(filters, kernel_size, strides)
        conv1 = tf.layers.conv3d(
            inputs=activation1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name='conv_1'
        )
        print('conv1\n', sess.run(conv1))
        activation2 = tf.nn.relu(conv1, name='activation_2')
        print('av2\n', sess.run(activation2))
        conv2 = tf.layers.conv3d(
            activation2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name='conv_2'
        )
        print('conv2\n', sess.run(conv2))
        output = tf.add(inputs, conv2, name='add')
        print('add\n', sess.run(output))
    return output
    
'''    
    
    
def ResUnits3D(input,
               size=(None, 60, 20, 20),
               repetations=1,
               name=None,):
    
    


def STResNet3D(
                c_conf=(3, 60, 20, 20),
                p_conf=(3, 60, 20, 20),
                t_conf=(3, 60, 20, 20),
                nb_resunit=3):
                
'''    