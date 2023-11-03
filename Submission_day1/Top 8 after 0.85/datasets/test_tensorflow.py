import numpy as np
import tensorflow as tf

# Ensure TensorFlow is working
print(f"TensorFlow Version: {tf.__version__}")

# Define a simple linear model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

# Dummy data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=1000)

# Predict using the trained model
print(model.predict([10.0]))
