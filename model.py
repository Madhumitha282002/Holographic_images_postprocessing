import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Kirchhoff-Fresnel Diffraction Kernel for 3D Reconstruction
def kirchhoff_fresnel_reconstruction(hologram, wavelength, pixel_size, z_distance):
    """
    Reconstruct 3D hologram using Kirchhoff-Fresnel diffraction integral.

    Parameters:
    hologram: 2D numpy array, the input hologram image
    wavelength: float, wavelength of the light in meters
    pixel_size: float, size of a single pixel in meters
    z_distance: float, distance from hologram to reconstruction plane in meters

    Returns:
    reconstructed_image: 2D numpy array, reconstructed image
    """
    k = 2 * np.pi / wavelength  # Wavenumber
    ny, nx = hologram.shape
    fx = np.fft.fftfreq(nx, d=pixel_size)
    fy = np.fft.fftfreq(ny, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)

    # Transfer function of the Kirchhoff-Fresnel kernel
    H = np.exp(1j * k * z_distance) * np.exp(-1j * np.pi * wavelength * z_distance * (FX**2 + FY**2))

    # Fourier transform of the hologram
    Hologram_fft = np.fft.fft2(hologram)

    # Applying the transfer function
    reconstructed_fft = Hologram_fft * H

    # Inverse Fourier transform to get the reconstructed image
    reconstructed_image = np.abs(np.fft.ifft2(reconstructed_fft))

    return reconstructed_image

# Preprocess and Dataset Preparation
def preprocess_data(dataset, wavelength, pixel_size, z_distance):
    reconstructed_images = []
    for hologram in dataset:
        reconstructed = kirchhoff_fresnel_reconstruction(hologram, wavelength, pixel_size, z_distance)
        reconstructed_images.append(reconstructed)
    return np.array(reconstructed_images)

data_3d_holograms = np.load('3d_hologram_dataset.npy')
wavelength = 532e-9 
pixel_size = 1.12e-6  
z_distance = 0.01  

# Preprocess the dataset
processed_data = preprocess_data(data_3d_holograms, wavelength, pixel_size, z_distance)
processed_data = processed_data[..., np.newaxis]  # Add channel dimension

# Customize ResNet50 for Particle Size Determination
# Using pre-trained ResNet50 as the base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(1, activation='linear')(x)  # Particle size prediction as a regression task

model = Model(inputs=base_model.input, outputs=output)


for layer in base_model.layers:
    layer.trainable = False

# Compile the model with a tuned learning rate
learning_rate = 1e-4  # Adjusted learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Data Augmentation and Model Training
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

data_train = data_gen.flow(processed_data, batch_size=32, subset='training')
data_val = data_gen.flow(processed_data, batch_size=32, subset='validation')

# Train the model
history = model.fit(data_train, validation_data=data_val, epochs=10, steps_per_epoch=100, validation_steps=20)

# Save the trained model
model.save('resnet50_particle_size.h5')

# Evaluate Model Accuracy
def evaluate_model_accuracy(model, data_val):
    """Evaluate the model's accuracy on validation data."""
    evaluation = model.evaluate(data_val)
    print(f"Validation Loss: {evaluation[0]:.4f}")
    print(f"Validation MAE: {evaluation[1]:.4f}")

# Evaluate the model
evaluate_model_accuracy(model, data_val)

# Calculate Average Particle Size
def calculate_average_particle_size(model, dataset):
    """Calculate the average particle size predictions on the dataset."""
    predictions = model.predict(dataset)
    average_size = np.mean(predictions)
    print(f"Average Predicted Particle Size: {average_size:.4f}")
    return average_size

#Calculate average particle size
average_particle_size = calculate_average_particle_size(model, data_val)
