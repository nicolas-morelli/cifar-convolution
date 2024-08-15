import functools
import numpy as np
import tensorflow as tf
import keras_tuner as kt


def conv_output_size(input_size, kernel_size, strides=1, padding="valid"):
    if padding == "valid":
        z = input_size - kernel_size + strides
        output_size = z // strides
        num_ignored = z % strides
        return output_size, num_ignored
    else:
        output_size = (input_size - 1) // strides + 1
        num_padded = (output_size - 1) * strides + kernel_size - input_size
        return output_size, num_padded


def modelbuilder(hp, ninputs, noutputs):

    nsteps = hp.Int('nsteps', min_value=1, max_value=5, step=1)
    nlengthconv = hp.Int('nlengthconv', min_value=1, max_value=5, step=1)
    # kernelsize = hp.Int('kernelsize', min_value=6, max_value=12, step=1)
    filters = hp.Int('filters', min_value=42, max_value=82, step=2)
    filterscale = hp.Float('filterscale', min_value=1.0, max_value=3.0, step=0.1)
    padding = hp.Choice('padding', ['valid', 'same'])
    strides = 1

    pool = hp.Boolean('pool')
    # psize = hp.Int('psize', min_value=2, max_value=2)

    nlength = hp.Int('nlength', min_value=2, max_value=6, step=1)
    nwidth = hp.Int('nwidth', min_value=60, max_value=300, step=5)

    input_ = tf.keras.layers.Input(shape=ninputs)

    rsc = tf.keras.layers.Rescaling(scale=1 / 255)
    resc = rsc(input_)

    for n in range(0, nsteps):
        for n2 in range(0, nlengthconv):
            bn = tf.keras.layers.BatchNormalization()
            conv = tf.keras.layers.Conv2D(activation='leaky_relu', kernel_initializer='he_normal', filters=filters, kernel_size=3, padding=padding, strides=strides)

            if n + n2 == 0:
                bn = tf.keras.layers.BatchNormalization()
                conv = tf.keras.layers.Conv2D(activation='leaky_relu', kernel_initializer='he_normal', filters=filters, kernel_size=7, padding=padding, strides=strides)
                midconv = conv(bn(resc))
            else:
                midconv = conv(bn(midconv))

        filters = filterscale * filters

        if pool:
            pool = tf.keras.layers.MaxPool2D(pool_size=2)
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
    opt = tf.keras.optimizers.Nadam(learning_rate=hp.Float('learning_rate', min_value=0.00001, max_value=0.01, step=0.000005))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def main():

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    modelbuilderparc = functools.partial(modelbuilder, ninputs=X_train[0].shape, noutputs=len(np.unique(y_train)))

    random_search_tuner = kt.BayesianOptimization(modelbuilderparc, objective='val_accuracy', overwrite=True, max_trials=25, seed=47)
    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    random_search_tuner.search(X_train, y_train, batch_size=64, validation_data=(X_test, y_test), callbacks=[es], epochs=500)

    bestmodel = random_search_tuner.get_best_models(num_models=1)[0]

    best_hp = random_search_tuner.get_best_hyperparameters()[0]


if __name__ == '__main__':
    main()
