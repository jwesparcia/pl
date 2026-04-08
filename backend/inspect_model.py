import tensorflow as tf
import os

MODEL_PATH = "e:/movie-recommendation/model/recommender.keras"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
    for layer in model.layers:
        print(f"Layer: {layer.name}, Type: {layer.__class__.__name__}")
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            if weights:
                print(f"  Weights shape: {[w.shape for w in weights]}")
else:
    print("Model not found.")
