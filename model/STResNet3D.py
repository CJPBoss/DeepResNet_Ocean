import tensorflow as tf

def _residual_unit_3d(inputs, filters=64, kernel_size=3,
                      strides=1, name=None):
    '''
    # a single residual unit,
    # inputs -> relu -> conv3d -> relu -> conv3d -> add -> outputs
    #    |                                           ^
    #    |___________________________________________|
    #
    # Arguments:
    #   inputs: a 5-D tensor, [batch, depth, height, width, channel]
    #   filters: Integer, the dimensionality of the output space.
    #   kernel_size: An integer or tuple/list of n integers, specifying the
    #       length of the convolution window.
    #   strides: An integer or tuple/list of n integers,
    #       specifying the stride length of the convolution.
    #       Specifying any stride value != 1 is incompatible with specifying
    #       any `dilation_rate` value != 1.
    #   name: A string, the name of the layer.
    #
    # Output:
    #   a 5-D tensor, which shape is the same as inputs'.
    '''
    if name is None:
        name = ''
    with tf.name_scope("Res_Unit_" + name) as res_unit: # res unit
        activation0 = tf.nn.relu(inputs, name='relu_0') # relu 0
        conv0 = tf.layers.conv3d( # conv3d 0
            inputs=activation0,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name='res_conv_'+name+'_0'
        ) 
        activation1 = tf.nn.relu(conv0, name='relu_1') # relu 1
        conv1 = tf.layers.conv3d( # conv3d 1
            activation1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name='res_conv_'+name+'_1'
        ) 
        output = tf.add(inputs, conv1, name='add') # add the inputs and conv3d
    return output
    
def res_units_3d(inputs, res_unit=_residual_unit_3d,
                 filters=64, kernel_size=3, strides=1,
                 name=None, num_res_units=1):
    '''
    # a sequence of residual units,
    #
    # Arguments:
    #   num_res_units: the number of residual units
    # 
    # Output:
    #   a 5-D tensor, which shape is the same as inputs'.
    '''
    if name is None:
        name=''
    with tf.name_scope("Res_Units_" + name) as res_units: # residual units
        for i in range(num_res_units): # generate resunits which amount is
                                       # the same as num_res_units
            inputs = res_unit(inputs, filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              name=name+'_'+str(i))
    return inputs


def STResNet3D(inputs,
               kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3)),
               filters=(64, 64, 64), 
               strides=((1, 1, 1), (1, 1, 1), (1, 1, 1)),
               num_res_units=(3, 3, 3),
               name=None):
    '''
    # Deep Spatio-Temporal Residual Networks.
    #
    # Arguments:
    #   input_c, input_p, input_t:
    #       a 5-D tensor, the input of Property closeness, period and trend.
    #       [batch, depth, height, width, channel]
    #       channel is the same as the length of one input's sequence
    #
    '''
    if name is None:
        name = 'STResNet3D'
    outputs = [] # output list of each property
    with tf.name_scope(name) as stresnet3d:
        for i in range(3): # 3 properties
            id = str(i)
            input = inputs[i]
            if input is None:
                continue
            with tf.name_scope("Property" + id) as process:
                preconv = tf.layers.conv3d( # use conv3d to make the shape of
                                            # tensor suit the res unit
                    inputs=input,
                    filters=filters[i],
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    padding='same',
                    name='Pre_conv_' + id
                )
                resunits = res_units_3d( # res units
                    inputs=preconv,
                    res_unit=_residual_unit_3d,
                    filters=filters[i],
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    name=id,
                    num_res_units=num_res_units[i]
                )
                activation = tf.nn.relu(resunits) # relu activation
                output = tf.layers.conv3d( # res unit's output
                    inputs=activation,
                    filters=1,
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    padding='same',
                    name='Res_' + id + 'out'
                )
            outputs.append(output)
        main_output = None
        
        if len(outputs) == 1:
            main_output = outputs[0]
            print('===============', main_output.shape)
        else:
            new_outputs = []
            for i in range(len(outputs)):
                newoutput = tf.layers.conv3d( # distribute weights for each output
                    outputs[i],
                    filters=1,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    name='weights' + str(i)
                )
                new_outputs.append(newoutput)
            main_output = sum(new_outputs) # merge all the outputs
    output = tf.nn.tanh(main_output, name=name + 'Output') # employ tanh to
                                                           # activate the output
    return output