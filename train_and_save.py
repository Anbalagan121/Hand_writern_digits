import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print("Preprocessing data...")
    x_train_cnn = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test_cnn = x_test.reshape(-1, 28, 28, 1) / 255.0

    print("Building CNN model...")
    cnn = Sequential()
    cnn.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    cnn.add(MaxPooling2D(2,2))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dense(10, activation='softmax'))

    cnn.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training CNN model...")
    history = cnn.fit(x_train_cnn, y_train, epochs=5, validation_data=(x_test_cnn, y_test))

    print("Evaluating model...")
    loss, acc_cnn = cnn.evaluate(x_test_cnn, y_test)
    print(f'CNN Accuracy: {acc_cnn:.4f}')

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("Saving model to models/cnn_model.keras...")
    cnn.save('models/cnn_model.keras')

    # Save history and metrics
    print("Saving history and metrics to data/...")
    history_dict = history.history
    # Convert numpy types to python lists for json serialization
    for key in history_dict:
        history_dict[key] = [float(val) for val in history_dict[key]]
    
    with open('data/history.json', 'w') as f:
        json.dump(history_dict, f, indent=4)

    # Compute confusion matrix on test data
    y_pred = cnn.predict(x_test_cnn)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    
    metrics_dict = {
        'test_loss': float(loss),
        'test_accuracy': float(acc_cnn),
        'confusion_matrix': cm.tolist()
    }
    with open('data/metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print("Training and saving completed successfully!")

if __name__ == "__main__":
    main()
