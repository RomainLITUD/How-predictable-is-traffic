from custom_model.baselayers import *

def build_model(para):
    inp = Input(shape=(para['obs'],para['nb_nodes'],2))
    x = inp
    
    if para['output_form'] == 'beta':
        
        for i in range(para['nb_blocks']):
            x = STBlock(para)(x)
            if para['modelnorm'] == 'batch':
                x = BatchNormalization(epsilon=1e-6)(x)
            elif para['modelnorm'] == 'layer':
                x = LayerNormalization(epsilon=1e-6)(x)

        x = Conv2D(para['pred'], (1, 1),data_format='channels_first', activation=None)(x)
        x = Conv2D(2, (1, 1), activation=None)(x)
        w, k = Lambda(lambda x: tf.split(x,2,axis=-1))(x)
        w = Activation('sigmoid')(w)
        k = Lambda(lambda x: x-1)(k)
        k = Activation('exponential')(k)
        out = Concatenate(-1)([w,k])
        return Model(inp, out)
    
    elif para['output_form'] == 'histogram':
        for i in range(para['nb_blocks']):
            x = STBlock(para)(x)
            #if i!=para['nb_blocks']-1:
            if para['modelnorm'] == 'batch':
                x = BatchNormalization(epsilon=1e-6)(x)
            elif para['modelnorm'] == 'layer':
                x = LayerNormalization(epsilon=1e-6)(x)
        
        out = Conv2D(para['gc_units'], (para['obs']-para['pred']+1, 1))(x)
        out = Dropout(0.1)(out)
        #out = Conv2D(para['nb_classes'], (1, 1), activation='sigmoid')(out)
        out = CrossAttention(para, activation='sigmoid')(out)
        out = AveragePooling2D((1,3), strides=(1,1), padding='same',data_format='channels_first')(out)

        return Model(inp, out)
    
def build_crossmodel(para):
    obs = para['obs']
    inp = Input(shape=(para['obs'],para['nb_nodes'],2))
    x = inp
    
    STlayers = [STBlock(para) for i in range(para['nb_blocks'])]
    normlayers = [LayerNormalization(epsilon=1e-6) for i in range(para['nb_blocks'])]
    
    if para['output_form'] == 'beta':
        convlayers = [Conv2D(1, (1, 1),data_format='channels_first', activation=None) for i in range(para['nb_blocks'])]
        crosslayers = [Conv2D(2, (1, 1), activation=None) for i in range(para['nb_blocks'])]

    if para['output_form'] == 'histogram':
        crosslayers = CrossAttention(para,activation='sigmoid')
        convlayers = [Conv2D(para['gc_units'], (para['nb_blocks'], 1), activation=None) for i in range(para['nb_blocks'])]
    
    hidden = []
    
    for i in range(para['nb_blocks']):
        x = STlayers[i](x)
        x = normlayers[i](x)
        hidden.append(x)
        
    output = []

    for i in range(para['pred']):
        xin = Concatenate(axis=-3)([hidden[j][:,obs-j-1:obs-j] if j>i-1 else hidden[i][:,obs-j-1:obs-j] for j in range(para['nb_blocks'])])
        if para['output_form'] == 'histogram':
            out = convlayers[i](xin)
            out = Dropout(0.1)(out)
            out = crosslayers(out)
            output.append(out)
        
        if para['output_form'] == 'beta':
            x = convlayers[i](xin)
            x = crosslayers[i](x)
            w, k = Lambda(lambda x: tf.split(x,2,axis=-1))(x)
            w = Activation('sigmoid')(w)
            k = Lambda(lambda x: x-1)(k)
            k = Activation('exponential')(k)
            out = Concatenate(-1)([w,k])
            output.append(out)
            
    out = Concatenate(axis=-3)(output)
    return Model(inp, out)