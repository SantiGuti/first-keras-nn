import numpy as np
import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# WIP
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])



model.compile(
  #optimizer='adam',
  optimizer=Adam(learning_rate=0.0005),
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

history = model.fit(
  train_images, # training data
  to_categorical(train_labels), # training targets
  epochs=20,
  batch_size=64,
  validation_data=(test_images, to_categorical(test_labels))
)

model.evaluate(
  test_images,
  to_categorical(test_labels)
)

# # Predict on the first 5 test images.
# predictions = model.predict(test_images[:5])
# # Print our model's predictions.
# print("ML Predictions: ", end='')
# print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]
# # Check our predictions against the ground truths.
# print("Real values: ", end='')
# print(test_labels[:5]) # [7, 2, 1, 0, 4]

# # Save the model to disk.
model.save_weights('model.h5')

# # Load the model from disk later using:
model.load_weights('model.h5')

# access accuracy and loss for every epoch
accuracy_history = history.history['val_accuracy']
loss_history = history.history['val_loss']

# Create ploth graph with matplotlib showing changes in accuracy and loss over epochs
plt.plot(accuracy_history, label="accuracy", linewidth=2, marker='o', markersize=7)
plt.plot(loss_history, label="loss", linewidth=2, marker='o', markersize=7)
plt.ylim(0,1)
plt.xlim(0,len(accuracy_history)-1)
plt.xlabel("Epoch")
plt.ylabel("Accuracy/Loss value")
plt.title("Results")
plt.legend()
plt.show()