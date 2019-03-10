import tensorflow as tf

# The export path contains the name and the version of the model
model = tf.keras.models.load_model('./cnnModel.h5')
export_path = '../my_digit_classifier/1'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})