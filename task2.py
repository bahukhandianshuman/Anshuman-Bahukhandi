import os
import numpy as np
import librosa
import tensorflow as tf
#from tensorflow.kera import layers, models
from keras import layers, models
from sklearn.model_selection import train_test_split

# --- SET YOUR FILE PATH HERE ---
# Example for Mac: '/Users/yourname/Downloads/RAVDESS_Dataset'
# Example for Windows: 'C:/Users/yourname/Desktop/RAVDESS'
dataset_path = 'Audio_Speech_Actors_01-24' 

# 1. Feature Extraction (Phase 1: Preprocessing)
def extract_features(file_path, duration=3, sr=22050):
    audio, _ = librosa.load(file_path, duration=duration, res_type='kaiser_fast')
    # Audio Cleaning: Remove dead air
    audio, _ = librosa.effects.trim(audio)
    
    # Uniform Padding (Ensures all images are the same size)
    expected_length = duration * sr
    if len(audio) < expected_length:
        audio = np.pad(audio, (0, expected_length - len(audio)), 'constant')
    else:
        audio = audio[:expected_length]

    # Convert to Log-Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec[..., np.newaxis]

# 2. Loading the Dataset (Phase 1: EDA & Prep)
X, y = [], []
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            # RAVDESS naming: 03-01- emotion -... .wav
            parts = file.split('-')
            if len(parts) > 2:
                emotion = int(parts[2]) - 1 # 0-indexed for 8 classes
                full_path = os.path.join(subdir, file)
                X.append(extract_features(full_path))
                y.append(emotion)

X = np.array(X)
y = np.array(y)

# 3. Model Building (Phase 2: Architecture)
# Split: 80% Train, 20% for Val/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = models.Sequential([
    layers.Input(shape=(128, 130, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(), # Required for Phase 2
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3), # Regularization to prevent memorizing voices
    layers.GlobalAveragePooling2D(), 
    layers.Dense(8, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Training
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# 5. Save Model
model.save('ser_model.keras')
print("Model trained and saved as ser_model.keras")
