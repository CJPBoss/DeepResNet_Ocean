import tensorflow as tf

def _residual_unit_3d(inputs, filters=64, kernel_size=3,
                      strides=1, name=None):
    if name is None:
        name = ''
    print(name)
    with tf.name_scope("Res_Unit_" + name) as res_unit:
        activation0 = tf.nn.relu(inputs, name='relu_0') 
        conv0 = tf.layers.conv3d(
            inputs=activation0,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name='res_conv_'+name+'_0'
        )
        activation1 = tf.nn.relu(conv0, name='relu_1')
        conv1 = tf.layers.conv3d(
            activation1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name='res_conv_'+name+'_1'
        )
        output = tf.add(inputs, conv1, name='add')
    return output
    
def res_units_3d(inputs, res_unit=_residual_unit_3d,
                 filters=64, kernel_size=3, strides=1,
                 name=None, repetations=1):
    if name is None:
        name=''
    with tf.name_scope("Res_Units_" + name) as res_units:
        for i in range(repetations):
            print(i)
            inputs = res_unit(inputs, filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              name=name+'_'+str(i))
    return inputs


def STResNet3D(input_c, input_p, input_t,
               kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3)),
               filters=(64, 64, 64), 
               strides=((1, 1, 1), (1, 1, 1), (1, 1, 1)),
               num_res_units=(3, 3, 3),
               name=None):
    if name is None:
        name = ''
    inputs = [input_c, input_p, input_t]
    outputs = []
    with tf.name_scope("ST_Res_Net_3D") as stresnet3d:
        for i in range(3):
            id = str(i)
            input = inputs[i]
            if input is None:
                continue
            #DHWC = input.shape[1:]  #depth, height, width, channel
            with tf.name_scope("Process" + str(i)) as process:
                preconv = tf.layers.conv3d(
                    inputs=input,
                    filters=filters[i],
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    padding='same',
                    name='Pre_conv_' + id
                )
                resunits = res_units_3d(
                    inputs=preconv,
                    res_unit=_residual_unit_3d,
                    filters=filters[i],
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    name='Pro_' + id,
                    repetations=num_res_units[i]
                )
                activation = tf.nn.relu(resunits)
                output = tf.layers.conv3d(
                    inputs=activation,
                    filters=1,
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    padding='same',
                    name='ResUnit_output_' + id
                )
            outputs.append(output)
        main_output = None
        
        if len(outputs) == 1:
            main_output = outputs[0]
        else:
            new_outputs = []
            for i in range(len(outputs)):
                newoutput = tf.layers.conv3d(
                    outputs[i],
                    filters=1,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    name='weights' + str(i)
                )
                new_outputs.append(newoutput)
            '''
            main_output = new_outputs[0]
            for i in range(len(new_outputs) - 1):
                main_output += new_outputs[i + 1]
            '''
            main_output = sum(new_outputs)
    output = tf.nn.tanh(main_output, name='OUTPUT')    
    return output
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    pass