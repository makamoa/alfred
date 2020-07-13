import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
if tf.__version__ != '1.12.0':
    import efficientnet.tfkeras as efn

class ResearchModels():
    def __init__(self, model, npoints=20, saved_model=None, pretrained_weights=None, input_shape=[100,100,1]):
        # Now compile the network.
        self.input_shape = input_shape
        self.nb_of_points = npoints
        self.saved_model = saved_model
        self.pretrained_weights = pretrained_weights
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model=='alexnet':
            print("Loading alexnet model.")
            self.model = self.alexnet()
        elif model=='primanet':
            print("Loading prima model.")
            self.model = self.primanet()
        elif model=='primanet2':
            print("Loading prima 2 model.")
            self.model = self.primanet2()
        elif model=='vgg':
            print("Loading vgg model.")
            self.model = self.primanet_vgg()
        elif model=='vgg2':
            print("Loading vgg model.")
            self.model = self.primanet_vgg2()
        elif model=='resnet18':
            print("Loading resnet model.")
            self.model = self.resnet18()
        elif model=='resnet18_v2':
            print("Loading resnet model.")
            self.model = self.resnet18_v2()
        elif model=='resnet18_112_112':
            print("Loading resnet 112X112 model.")
            self.model = self.resnet18_112_112()
        elif model=='effnet':
            self.model = self.effnet()
        elif model=='effnet_b1':
            self.model = self.effnet_b1()
        else:
            print("Unknown network.")
            sys.exit()

        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='mse', optimizer=optimizer)
        print(self.model.summary())

    def alexnet(self):
        # (3) Create a sequential model
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(250, 250, 3), kernel_size=(11, 11), \
                         strides=(4, 4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(512))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(512))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation

        # Output Layer
        model.add(Dense(self.nb_of_points))
        model.add(Activation('sigmoid'))
        return model

    def primanet(self):
        model = Sequential()
        model.add(Conv2D(32, (11, 11), activation='relu',
                         input_shape=(250, 250, 1)))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(64, (11, 11), activation='relu'))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_of_points, activation='sigmoid'))
        return model

    def primanet2(self):
        model = Sequential()
        model.add(Conv2D(64, (11, 11), activation='relu',
                         input_shape=(250, 250, 3)))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(128, (11, 11), activation='relu'))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_of_points, activation='sigmoid'))
        return model

    def primanet_vgg(self):
        from primanet_fcnn import nn_base
        input_shape_img = (224, 224, 1)
        img_input = Input(shape=input_shape_img, name='ImgInput')
        #x = BatchNormalization(name='BatchNormal_0')(img_input)
        x = nn_base(img_input, trainable=True)
        x = Flatten(name='flat')(x)
        x = Dropout(0.5, name='drop1')(x)
        x = Dense(1024, activation='relu', name='dense1')(x)
        x = Dropout(0.5, name='drop2')(x)
        x = Dense(512, activation='relu', name='dense2')(x)
        x = Dense(self.nb_of_points, activation='sigmoid',name='dense3')(x)
        model = Model(img_input, x)
        return model

    def primanet_vgg2(self):
        from primanet_fcnn import nn_base
        input_shape_img = (224, 224, 3)
        img_input = Input(shape=input_shape_img, name='ImgInput')
        #x = BatchNormalization(name='BatchNormal_0')(img_input)                
        x = nn_base(img_input, trainable=True,dim=3)
        x = Flatten(name='flat')(x)
        x = Dropout(0.5, name='drop1')(x)
        x = Dense(1024, activation='relu', name='dense1')(x)
        x = Dropout(0.5, name='drop2')(x)
        x = Dense(512, activation='relu', name='dense2')(x)
        x = Dense(self.nb_of_points, activation='sigmoid',name='dense3')(x)
        model = Model(img_input, x)
        return model

    def resnet18(self):
        from classification_models.keras import Classifiers
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        conv_base = ResNet18(input_shape=self.input_shape, include_top=False)
        x = Flatten(name='flat')(conv_base.output)
        x = Dropout(0.5, name='drop1')(x)
        x = Dense(1024, activation='relu', name='dense1')(x)
        x = Dropout(0.5, name='drop2')(x)
        x = Dense(512, activation='relu', name='dense2')(x)
        x = Dense(self.nb_of_points, activation='sigmoid',name='dense3')(x)
        model = Model(conv_base.input, x)
        return model

    def resnet18_v2(self):
        from classification_models.keras import Classifiers
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        conv_base = ResNet18(input_shape=(224,224,1), include_top=False)
        x = GlobalAveragePooling2D(name='GlobalAveragePooling')(conv_base.output)
        x = Dropout(0.5, name='drop1')(x)
        x = Dense(1024, activation='relu', name='dense1')(x)
        x = Dropout(0.5, name='drop2')(x)
        x = Dense(512, activation='relu', name='dense2')(x)
        x = Dense(self.nb_of_points, activation='sigmoid',name='dense3')(x)
        model = Model(conv_base.input, x)
        return model

    def resnet18_112_112(self):
        from classification_models.keras import Classifiers
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        conv_base = ResNet18(input_shape=(128,128,1), include_top=False)
        #layer_name = 'stage4_unit1_relu1'
        #conv_base = Model(inputs=rm.input, outputs=rm.get_layer(layer_name).output)
        x = Flatten(name='flat')(conv_base.output)
        x = Dropout(0.5, name='drop1')(x)
        x = Dense(1024, activation='relu', name='dense1')(x)
        x = Dropout(0.5, name='drop2')(x)
        x = Dense(512, activation='relu', name='dense2')(x)
        x = Dense(self.nb_of_points, activation='sigmoid',name='dense3')(x)
        model = Model(conv_base.input, x)
        return model

    def effnet(self):
        conv_base = efn.EfficientNetB0(weights=None, input_shape=self.input_shape, include_top=False,
                                       pooling='avg')  # or weights='noisy-student'
        x = Flatten(name='flat')(conv_base.output)
        x = Dropout(0.5, name='drop1')(x)
        x = Dense(1024, activation='relu', name='dense1')(x)
        x = Dropout(0.5, name='drop2')(x)
        x = Dense(512, activation='relu', name='dense2')(x)
        x = Dense(self.nb_of_points, activation='sigmoid',name='dense3')(x)
        model = Model(conv_base.input, x)
        return model

    def effnet_b1(self):
        conv_base = efn.EfficientNetB1(weights=None, input_shape=self.input_shape, include_top=False,
                                       pooling='avg')  # or weights='noisy-student'
        x = Flatten(name='flat')(conv_base.output)
        x = Dropout(0.5, name='drop1')(x)
        x = Dense(1024, activation='relu', name='dense1')(x)
        x = Dropout(0.5, name='drop2')(x)
        x = Dense(512, activation='relu', name='dense2')(x)
        x = Dense(self.nb_of_points, activation='sigmoid',name='dense3')(x)
        model = Model(conv_base.input, x)
        return model

class MultipleModels():
    def __init__(self, model_files, input_shapes=[100,100,1]):
        self.nmodels = len(model_files)
        self.model_files = model_files
        if np.array(input_shapes).shape.__len__() == 1:
            #same shapes for all inputs
            input_shapes = [input_shapes] * self.nmodels
        self.input_shapes=input_shapes
        self.model = self.build_model()
        optimizer = Adam(lr=1e-5, decay=1e-6)
        losses= ['mse'] * self.nmodels
        self.model.compile(loss=losses, optimizer=optimizer)


    def build_model(self):
        inputs = [Input(input_shape) for input_shape in self.input_shapes]
        models = [MultipleModels.rename_layers(load_model(file),i) for i,file in enumerate(self.model_files)]
        outputs = []
        for i in range(self.nmodels):
            output = models[i](inputs[i])
            outputs.append(output)
        return Model(inputs=inputs, outputs=outputs)
    @staticmethod
    def rename_layers(model,i):
        model._name = model._name + '_' + str(i)
        for layer in model.layers:
            layer._name = layer.name + '_' + str(i)
        return model
