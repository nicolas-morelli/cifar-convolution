import functools
import numpy as np
import tensorflow as tf
import keras_tuner as kt


def modelbuilder(hp, ninputs, noutputs):

    nsteps = hp.Int('nsteps', min_value=1, max_value=5, step=1)
    nlengthconv = hp.Int('nlengthconv', min_value=1, max_value=5, step=1)
    kernelsize = hp.Int('kernelsize', min_value=2, max_value=8, step=1)
    startkernelsize = hp.Int('startkernelsize', min_value=6, max_value=20, step=1)
    filters = hp.Int('filters', min_value=50, max_value=250, step=2)
    filterscale = hp.Float('filterscale', min_value=1.0, max_value=4.0, step=0.1)
    padding = hp.Choice('padding', ['valid', 'same'])
    strides = 1

    pool = True  # hp.Boolean('pool')
    psize = hp.Int('psize', min_value=2, max_value=3)

    nlength = hp.Int('nlength', min_value=2, max_value=6, step=1)
    nwidth = hp.Int('nwidth', min_value=100, max_value=1000, step=5)

    input_ = tf.keras.layers.Input(shape=ninputs)

    rsc = tf.keras.layers.Rescaling(scale=1 / 255)
    resc = rsc(input_)

    for n in range(0, nsteps):
        for n2 in range(0, nlengthconv):
            bn = tf.keras.layers.BatchNormalization()

            if n + n2 == 0:
                conv = tf.keras.layers.Conv2D(activation='leaky_relu', kernel_initializer='he_normal', filters=filters, kernel_size=startkernelsize, padding=padding, strides=strides)
                midconv = conv(bn(resc))
            else:
                conv = tf.keras.layers.Conv2D(activation='leaky_relu', kernel_initializer='he_normal', filters=filters, kernel_size=kernelsize, padding=padding, strides=strides)
                midconv = conv(bn(midconv))

        filters = round(filterscale * filters)

        if pool:
            pool = tf.keras.layers.MaxPool2D(pool_size=psize)
            midconv = pool(midconv)

    fla = tf.keras.layers.Flatten()
    midconv = fla(midconv)

    for n in range(0, nlength):
        bn = tf.keras.layers.BatchNormalization()
        midlayer = tf.keras.layers.Dense(nwidth, activation='leaky_relu', kernel_initializer='he_normal', use_bias=False)
        do = tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.6, step=0.05))

        if n == 0:
            mid = do(midlayer(bn(midconv)))
        else:
            mid = do(midlayer(bn(mid)))

    outputlayer = tf.keras.layers.Dense(noutputs, activation='softmax')

    output = outputlayer(mid)

    model = tf.keras.Model(inputs=[input_], outputs=[output])
    opt = tf.keras.optimizers.Nadam(learning_rate=hp.Float('learning_rate', min_value=0.0001, max_value=0.01, step=0.00005))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def main():

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    modelbuilderparc = functools.partial(modelbuilder, ninputs=X_train[0].shape, noutputs=len(np.unique(y_train)))

    trials = 15

    tuner = kt.Hyperband(hypermodel=modelbuilderparc,
                         max_epochs=500,
                         factor=3,
                         hyperband_iterations=1,
                         objective='val_accuracy',
                         overwrite=True,
                         executions_per_trial=trials,
                         max_consecutive_failed_trials=trials,
                         project_name='Convo',
                         seed=47)

    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    tuner.search(X_train, y_train, batch_size=64, validation_data=(X_test, y_test), callbacks=[es], epochs=500, verbose=2)

    bestmodel = tuner.get_best_models(num_models=1)[0]

    best_hp = tuner.get_best_hyperparameters()[0]

    bestmodel.save('convmodel.keras')


if __name__ == '__main__':
    main()
