import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt

# create model

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")
        ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    return model_clf

# create unique path
def get_unique_filename(filename): # try to use ascitime that is best
    unique_filename= time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


#save model
def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)


# save plot
# create unique path
def get_unique_filePlotname(filename):
    unique_filename= time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_plot(df, plot_name, plots_dir ):
    unique_filename = get_unique_filePlotname(plot_name)
    path_to_plot = os.path.join(plots_dir, unique_filename)
    df.plot(figsize=(10, 7))
    plt.grid(True)
    #plt.show()
    plt.savefig(path_to_plot)


