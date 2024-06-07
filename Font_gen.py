import os
import numpy as np
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import svgwrite

def load_system_fonts():
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    return font_files

def draw_glyph(font_path, char, image_size=(64, 64)):
    font = ImageFont.truetype(font_path, image_size[1])
    image = Image.new("L", image_size, 255)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), char, font=font, fill=0)
    return np.array(image)

def create_dataset(chars, font_files, image_size=(64, 64)):
    images = []
    labels = []
    for i, font_file in enumerate(font_files):
        for char in chars:
            image = draw_glyph(font_file, char, image_size)
            images.append(image)
            labels.append(i)
    return np.array(images), np.array(labels)

chars = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ']  # Example Hindi characters
font_files = load_system_fonts()
images, labels = create_dataset(chars, font_files)
images = images / 255.0  # Normalize images
images = np.expand_dims(images, axis=-1)  # Add channel dimension

def build_autoencoder(input_shape):
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(encoder_input, decoded)
    return autoencoder

input_shape = (64, 64, 1)
autoencoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(images, images, epochs=50, batch_size=32, validation_split=0.2)

def save_as_svg(image, filename, threshold=128):
    dwg = svgwrite.Drawing(filename, profile='tiny', size=(image.shape[1], image.shape[0]))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] < threshold:  # Threshold to convert grayscale to binary
                dwg.add(dwg.rect((x, y), (1, 1), fill=svgwrite.rgb(0, 0, 0, '%')))
    dwg.save()

def generate_and_save_svgs(autoencoder, images, n=5, output_dir='output_svgs'):
    decoded_imgs = autoencoder.predict(images[:n])
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n):
        filename = os.path.join(output_dir, f'reconstructed_{i}.svg')
        save_as_svg(decoded_imgs[i].reshape(64, 64), filename)

generate_and_save_svgs(autoencoder, images)
