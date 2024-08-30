import numpy as np
import tensorflow as tf
import keras_tuner as kt

class KillWhenLowAcc(tf.keras.callbacks.Callback):
    def __init__(self, minaccuracy, epoch):
        super(KillWhenLowAcc, self).__init__()
        self.minaccuracy = minaccuracy
        self.epoch = epoch
        self.maxaccuracyyet = 0.0
        self.epochindex = 0

    def on_epoch_end(self, epoch, logs=None):

        accuracy = logs.get('val_accuracy')

        if accuracy > self.maxaccuracyyet:
            self.maxaccuracyyet = accuracy

        if self.epochindex < len(self.epoch) and epoch == self.epoch[self.epochindex]:

            if self.maxaccuracyyet < self.minaccuracy[self.epochindex]:
                self.model.stop_training = True

            self.epochindex += 1


class HyperNet(kt.HyperModel):

    def __init__(self, **kwargs):
        self.ninputs = kwargs.get('ninputs')
        self.noutputs = kwargs.get('noutputs')
        super().__init__()

    def build(self, hp):
        nsteps = hp.Int('nsteps', min_value=3, max_value=5, step=1)
        nlengthconv = hp.Int('nlengthconv', min_value=8, max_value=16, step=1)
        kernelsize = hp.Int('kernelsize', min_value=2, max_value=5, step=1)
        startkernelsize = hp.Int('startkernelsize', min_value=4, max_value=8, step=1)
        filters = hp.Int('filters', min_value=80, max_value=120, step=2)
        scale = hp.Boolean('scale')
        filterscale = hp.Float('filterscale', min_value=2.0, max_value=4.0, step=0.2)
        padding = hp.Choice('padding', ['valid', 'same'])
        strides = 1

        pool = hp.Boolean('pool')
        psize = hp.Int('psize', min_value=2, max_value=4)

        nlength = hp.Int('nlength', min_value=8, max_value=16, step=2)
        nwidth = hp.Int('nwidth', min_value=150, max_value=250, step=5)

        input_ = tf.keras.layers.Input(shape=self.ninputs)

        rf = tf.keras.layers.RandomFlip("horizontal")
        rr = tf.keras.layers.RandomRotation(0.1)
        rz = tf.keras.layers.RandomZoom(0.1)

        input_ = rz(rr(rf(input_)))

        rsc = tf.keras.layers.Rescaling(scale=1 / 255)
        resc = rsc(input_)

        for n in range(0, nsteps):
            for n2 in range(0, nlengthconv):
                bn = tf.keras.layers.BatchNormalization()

                if n + n2 == 0:
                    conv = tf.keras.layers.SeparableConv2D(activation='relu', depthwise_initializer='he_normal', pointwise_initializer='he_normal', filters=filters, kernel_size=startkernelsize, padding=padding, strides=(strides, strides))
                    midconv = conv(bn(resc))
                else:
                    conv = tf.keras.layers.SeparableConv2D(activation='relu', depthwise_initializer='he_normal', pointwise_initializer='he_normal', filters=filters, kernel_size=kernelsize, padding=padding, strides=(strides, strides))
                    midconv = conv(bn(midconv))

            if scale:
                filters = round(filterscale * filters)

            if pool:
                pool = tf.keras.layers.MaxPool2D(pool_size=psize)
                midconv = pool(midconv)

        fla = tf.keras.layers.Flatten()
        midconv = fla(midconv)

        for n in range(0, nlength):
            bn = tf.keras.layers.BatchNormalization()
            midlayer = tf.keras.layers.Dense(nwidth, activation='leaky_relu', kernel_initializer='he_normal', use_bias=False)
            do = tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.3, step=0.05))

            if n == 0:
                mid = do(midlayer(bn(midconv)))
            elif n != nlength-1:
                mid = do(midlayer(bn(mid)))
            else:
                mid = midlayer(bn(mid))

        lastdo = tf.keras.layers.Dropout(rate=hp.Float('lastdropout_rate', min_value=0.3, max_value=0.6, step=0.05))
        outputlayer = tf.keras.layers.Dense(self.noutputs, activation='softmax')

        output = outputlayer(lastdo(mid))

        model = tf.keras.Model(inputs=[input_], outputs=[output])
        opt = tf.keras.optimizers.Nadam(learning_rate=hp.Float('learning_rate', min_value=0.01, max_value=0.1, step=0.01))

        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model


def main():

    if tf.config.list_physical_devices('GPU'):
        print("GPU is available.")
    else:
        print("GPU is not available.")

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    trials = 50

    tuner = kt.BayesianOptimization(hypermodel=HyperNet(ninputs=X_train[0].shape, noutputs=len(np.unique(y_train))),
                                    objective='val_accuracy',
                                    max_trials=trials,
                                    max_retries_per_trial=0,
                                    max_consecutive_failed_trials=trials,
                                    overwrite=True,
                                    project_name='Convo',
                                    seed=47)

    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
    es = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    kwa = KillWhenLowAcc(minaccuracy=[0.2, 0.5, 0.75], epoch=[2, 14, 29])

    tuner.search(X_train, y_train, batch_size=64, validation_data=(X_test, y_test), callbacks=[es, rlr, kwa], epochs=1000)

    bestmodel = tuner.get_best_models(num_models=1)[0]

    best_hp = tuner.get_best_hyperparameters()[0]

    print(best_hp.values)

    bestmodel.save('convmodel.keras')


if __name__ == '__main__':
    main()
