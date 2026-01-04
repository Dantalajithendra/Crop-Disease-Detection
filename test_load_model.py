import tensorflow as tf
print("TensorFlow version:", tf.__version__)

print("Loading model...")
model = tf.keras.models.load_model("crop_disease_model.h5")
print("✅ Model loaded successfully")
