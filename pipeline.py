import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Step 1: Generate dataset ---
def generate_lightcurve(has_transit):
    time = np.linspace(0, 15, 150)
    flux = np.ones_like(time)
    if has_transit:
        flux[70:80] -= 0.01  # transit dip
    flux += np.random.normal(0, 0.0005, size=len(time))  # noise
    return flux

X, y = [], []
for i in range(300):   # 300 samples (fast)
    label = np.random.choice([0, 1])
    X.append(generate_lightcurve(label))
    y.append(label)

X = np.array(X).reshape((300,150,1))
y = to_categorical(y, num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- Step 2: Build CNN model ---
model = Sequential([
    Conv1D(8, 5, activation='relu', input_shape=(150,1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Step 3: Train model ---
history = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test,y_test), verbose=1)

# --- Step 4: Evaluate ---
loss, acc = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", acc)

# --- Step 5: Plot Training/Validation Accuracy ---
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CNN Training Progress")
plt.legend()
plt.show()

# --- Step 6: Show example predictions ---
preds = model.predict(X_test)

# Pick 5 random examples from test set
indices = np.random.choice(len(X_test), 5, replace=False)
for i in indices:
    flux = X_test[i].flatten()
    label = np.argmax(y_test[i])
    pred = np.argmax(preds[i])
    
    plt.figure(figsize=(5,3))
    plt.plot(np.linspace(0,15,150), flux, label="Light Curve")
    plt.title(f"True: {'Transit' if label==1 else 'No Transit'} | Predicted: {'Transit' if pred==1 else 'No Transit'}")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.legend()
    plt.show()                                                                                                                     
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # for smoothing

# --- Function to create synthetic light curve ---
def generate_lightcurve(has_transit):
    time = np.linspace(0, 15, 150)
    flux = np.ones_like(time)
    if has_transit:
        flux[70:80] -= 0.01  # transit dip
    flux += np.random.normal(0, 0.0005, size=len(time))  # noise
    return flux

# --- Function to flatten a light curve ---
def flatten_lightcurve(flux, window=21, poly=2):
    # Apply Savitzky-Golay smoothing filter
    trend = savgol_filter(flux, window, poly)
    flattened = flux / trend
    return flattened, trend

# --- Example: Show original vs trend vs flattened ---
flux = generate_lightcurve(has_transit=True)
flat_flux, trend = flatten_lightcurve(flux)

plt.figure(figsize=(8,5))
plt.plot(np.linspace(0,15,150), flux, label="Original Light Curve", alpha=0.7)
plt.plot(np.linspace(0,15,150), trend, label="Fitted Trend", color="orange", linewidth=2)
plt.plot(np.linspace(0,15,150), flat_flux, label="Flattened Light Curve", color="green")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Light Curve Flattening Example")
plt.legend()
plt.show()
