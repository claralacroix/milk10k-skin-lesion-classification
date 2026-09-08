# MILK10k Skin Lesion Classification

## Project Overview

This project focuses on skin lesion classification using the MILK10k dataset.

The main task is a three-class classification problem:

- Benign
- Malignant
- Indeterminate

## Clinical Task

The goal is to develop a computer vision model that classifies skin lesion images into broad diagnostic categories.

The three target categories are Benign, Malignant, and Indeterminate.

This project is a machine learning classification study and is not intended to replace clinical diagnosis.

## Dataset

The MILK10k dataset used in this project contains:

- 10,480 images
- 5,240 unique lesions
- 2 images per lesion

The dataset includes image metadata and ground-truth diagnostic labels.

## Exploratory Data Analysis

The first session focused on:

- Loading `metadata.csv` and `training_gt.csv`
- Exploring the dataset structure
- Inspecting sample images
- Converting images to NumPy arrays
- Investigating class imbalance
- Examining metadata and missing values
- Checking image dimensions
- Verifying the two-images-per-lesion structure

### Main EDA Results

The distribution of the three main diagnostic categories is:

| Diagnosis | Number of images |
|-----------|-----------------:|
| Malignant | 7,268 |
| Benign | 2,966 |
| Indeterminate | 246 |

The average image size in a sample of 200 images was:

- Width: 600 pixels
- Height: 450 pixels

Every lesion in the dataset has exactly two images.

## Project Structure

```text
milk10k-skin-lesion-classification/

├── data/
│   └── milk10k.zip
├── notebooks/
│   └── Computer_vision_n1.ipynb
├── src/
├── .gitignore
└── README.md
