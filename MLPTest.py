import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parameters ---
IMG_SIZE = 100  # 100x100 images
CHANNELS = 3     # RGB images
INPUT_DIM = IMG_SIZE * IMG_SIZE * CHANNELS  # 100*100*3 = 30,000

# --- Build Classifier ---
def build_classifier(input_dim=INPUT_DIM):
    input_img = keras.Input(shape=(input_dim,))
    
    # Feature extraction layers (similar to encoder)
    x = layers.Dense(1024, activation="relu")(input_img)
    x = layers.Dropout(0.3)(x)  # Add dropout for regularization
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid")(x)  # Binary classification
    
    classifier = keras.Model(input_img, output, name="classifier")
    classifier.compile(
        optimizer=keras.optimizers.Adam(1e-3), 
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return classifier

classifier = build_classifier()
classifier.summary()

IMG_SIZE = (100, 100)   # target resize
BATCH_SIZE = 32

# --- Load training set ---
train_ds = keras.utils.image_dataset_from_directory(
    "dataset/train",
    labels="inferred",       # infer labels from subfolders
    label_mode="int",        # "int" = numeric labels, "categorical" for one-hot
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# --- Load test set ---
test_ds = keras.utils.image_dataset_from_directory(
    "dataset/test",
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Normalize pixel values to [0,1]
normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds  = test_ds.map(lambda x, y: (normalization_layer(x), y))

def flatten_map(x, y):
    # flatten images into vectors (batch, 100*100*3)
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    return x, y   # keep original labels for classification

train_flat = train_ds.map(flatten_map)
test_flat = test_ds.map(flatten_map)

# Train the classifier
history = classifier.fit(train_flat, validation_data=test_flat, epochs=10)

# Evaluate the model
test_loss, test_accuracy = classifier.evaluate(test_flat)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions for confusion matrix
print("\nGenerating predictions for confusion matrix...")
y_pred_proba = classifier.predict(test_flat)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Extract true labels from test dataset
y_true = []
for images, labels in test_flat:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Display confusion matrix with labels
print(f"\nConfusion Matrix (with labels):")
print("                 Predicted")
print("                 No Flower  Flower")
print(f"Actual No Flower    {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"Actual Flower       {cm[1,0]:4d}     {cm[1,1]:4d}")

# Calculate and display additional metrics
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nDetailed Metrics:")
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Create and display confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Flower', 'Flower'], 
            yticklabels=['No Flower', 'Flower'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Print classification report
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['No Flower', 'Flower']))