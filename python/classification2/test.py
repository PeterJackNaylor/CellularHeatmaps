import pandas as pd
from keras.models import Sequential#Import from keras_preprocessing not from keras.preprocessingfrom keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
from keras.applications.resnet import ResNet50

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model


df=pd.read_csv("/Users/naylorpeter/tmptmp/nature/test_jpg.csv")
df['path'] = df.apply(lambda row: row['path'].replace('.jpg', '.png'), axis=1)
df['path'] = df.apply(lambda row: row['path'].split('/')[-1], axis=1)

y_name = "Residual"
df[y_name] = df[y_name].astype(str)
datagen=ImageDataGenerator(rescale=1./255)
n = df.shape[0]
import pdb; pdb.set_trace()
train_generator=datagen.flow_from_dataframe(dataframe=df.ix[:n//2], directory="/Users/naylorpeter/tmptmp/nature/test_jpg", x_col="Biopsy", y_col=y_name, 
                                            class_mode="categorical", target_size=(224,224), batch_size=8, validate_filenames=False)
valid_generator=datagen.flow_from_dataframe(dataframe=df.ix[n//2:], directory=None, x_col="Biopsy", y_col=y_name, 
                                            class_mode="categorical", target_size=(224,224), batch_size=8)#, validate_filenames=False)



basemodel = ResNet50(include_top=False, weights='imagenet', pooling='avg', classes=2)
# construct the head of the model that will be placed on top of the
# the base model
headModel = Dense(512, activation="relu")(basemodel.output)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=basemodel.input, outputs=headModel)

model.compile(optimizers.rmsprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10)