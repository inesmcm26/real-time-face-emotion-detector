from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

def train_model(model, data, checkpoint_path, rotation_range = 10, width_shift_range = 0.2, height_shift_range = 0.2, zoom_range = 0.2, horizontal_flip = True):
    """
     Train the model using data augmentation
    """
    [X_train, X_val, y_train, y_val] = data
    
    # data augmentation generator
    datagen = ImageDataGenerator(
            rotation_range=10, # rotation
            width_shift_range=0.2, # horizontal shift
            height_shift_range=0.2, # vertical shift
            zoom_range=0.2, # zoom
            horizontal_flip=True) # horizontal flip
    
    # checkpoint
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')

    # fit the model
    history = model.fit(datagen.flow(X_train,y_train,
                                        batch_size=32, 
                                        seed=27,
                                        shuffle=False),
                        epochs=30,
                        steps_per_epoch=X_train.shape[0] // 32,
                        validation_data=(X_val,y_val),
                        callbacks = [checkpoint],
                        verbose = True)
    
    return history