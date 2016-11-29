mentor_conv.h5
  50 epochs, batch size 100, final accuracy: 99.21
  Model spec:
      def build_mentor_model_conv(load=False):
          mentor_model = Sequential()
          # This method of adding the first layer is required to synchronize the
          # Keras and TensorFlow representations
          # See https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
          # for additional information
          l1 = Convolution2D(20, 5, 5, subsample=(1, 1), border_mode='same', activation='relu', input_shape=(28, 28, 1))
          l1.set_input(img_conv)
          mentor_model.add(l1)
          mentor_model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
          mentor_model.add(Convolution2D(50, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
          mentor_model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
          mentor_model.add(Flatten())
          mentor_model.add(Dense(64, activation='relu'))
          mentor_model.add(Dense(64, activation='relu'))
          mentor_model.add(Dense(10))
          mentor_model.add(Activation('softmax'))
          if load:
              print ("Loading saved model")
              mentor_model.load_weights('mentor_conv.h5')
          return mentor_model
