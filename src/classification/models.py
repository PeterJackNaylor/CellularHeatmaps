
from keras.layers.core import Dropout, Flatten, Dense, Lambda
from keras.layers import Conv2D
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, SGD

def load_model(name, classes=2, dropout=0.5, batch_size=6):
    if name == 'resnet50':
        basemodel = ResNet50(include_top=False, weights='imagenet', pooling='avg', classes=classes)
        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = Dense(512, activation="relu")(basemodel.output)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(classes, activation="softmax")(headModel)
        model = Model(inputs=basemodel.input, outputs=headModel)
    elif name == 'resnet50conv':

        basemodel = ResNet50(include_top=False, weights='imagenet', pooling=None, classes=classes)
        headModel = Conv2D(512, (7,7), activation='relu')(basemodel.output)
        if dropout:
            headModel = Dropout(dropout)(headModel)
        headModel = Conv2D(512, (1,1), activation='relu')(headModel)
        if dropout:
            headModel = Dropout(dropout)(headModel)
        headModel = Conv2D(classes, (1,1), activation='softmax')(headModel)
        model = Model(inputs=basemodel.input, outputs=headModel)

    elif name == 'resnet50convflat':

        basemodel = ResNet50(input_shape=(),    
                             include_top=False, 
                             weights='imagenet',
                             pooling=None, 
                             classes=classes)
        headModel = Conv2D(512, (7,7), activation='relu')(basemodel.output)
        if dropout:
            headModel = Dropout(dropout)(headModel)
        headModel = Conv2D(512, (1,1), activation='relu')(headModel)
        if dropout:
            headModel = Dropout(dropout)(headModel)
        headModel = Conv2D(classes, (1,1), activation='softmax')(headModel)
        # headModel = Flatten()(headModel)
        headModel = Lambda(lambda x: x[:,0,0,:], input_shape=(batch_size, classes))(headModel)
        model = Model(inputs=basemodel.input, outputs=headModel)
        
    elif name == 'resnet50convflatfreeze':

        basemodel = ResNet50(input_shape=(),    
                             include_top=False, 
                             weights='imagenet',
                             pooling=None, 
                             classes=classes)
        headModel = Conv2D(512, (7,7), activation='relu')(basemodel.output)
        if dropout:
            headModel = Dropout(dropout)(headModel)
        headModel = Conv2D(512, (1,1), activation='relu')(headModel)
        if dropout:
            headModel = Dropout(dropout)(headModel)
        headModel = Conv2D(classes, (1,1), activation='softmax')(headModel)
        # headModel = Flatten()(headModel)
        headModel = Lambda(lambda x: x[:,0,0,:], input_shape=(batch_size, classes))(headModel)
        model = Model(inputs=basemodel.input, outputs=headModel)
        for layer in model.layers[4:-4]:
            layer.trainable = False
    elif name == 'custom_simple':
        pass

    return model 

def load_optimizer(stri, lr):
    if stri == 'adam':
        return Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif stri == "sgd":
        return SGD(lr=lr, momentum=0.9)
    else:
        return 'sgd'

def call_backs_load(stri, filepath):
    if stri:
        if stri == "version1":
            lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=5,
                                           mode='auto')  
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                                         save_best_only=True, 
                                         save_weights_only=True,
                                         mode='auto', period=1)
            out = [lr_reducer, checkpoint]
    else:
        out = None

    return out