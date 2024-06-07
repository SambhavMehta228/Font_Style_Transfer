Font Style Transfer Model

This project aims to create a simple model that takes hand-drawn fonts as input and generates glyphs in a similar style using existing system fonts. The model is trained using a convolutional autoencoder, and the generated glyphs are saved as SVG vector files.
Table of Contents

    Prerequisites
    Installation
    Dataset Preparation
    Model Training
    Generating SVGs
    Usage
    Contributing
    License

Prerequisites

Ensure you have the following libraries installed:

    TensorFlow
    NumPy
    Matplotlib
    Pillow
    svgwrite
    matplotlib.font_manager

You can install these using pip:

bash

pip install tensorflow numpy matplotlib pillow svgwrite

Installation

Clone the repository:

bash

git clone https://github.com/SambhavMehta228/Font_Style_Transfer.git

cd Font_Style_Transfer

Dataset Preparation

This project uses existing system fonts to create a dataset of glyphs. Define the characters you want to use and create a dataset by drawing glyphs from the fonts. Normalize and reshape the images as needed.
Model Training

The model is a simple convolutional autoencoder. Build the autoencoder, compile it, and train it on the prepared dataset.
Generating SVGs

Convert the output images from the autoencoder to SVG files. Save the generated glyphs as SVG files in the specified directory.
Usage

    Load System Fonts
    Create and Preprocess Dataset
    Build and Train the Model
    Generate and Save SVGs

Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
License

This project is licensed under the MIT License.
