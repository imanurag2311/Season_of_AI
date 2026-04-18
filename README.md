# Season of AI

A curated collection of hands-on AI and Deep Learning assignments covering Computer Vision, Natural Language Processing (NLP), and Generative Models.

This repository contains Jupyter notebooks developed as part of a learning journey through core ML/DL concepts and practical model building.

## Table of Contents

- [About the Project](#about-the-project)
- [Project Highlights](#project-highlights)
- [Repository Structure](#repository-structure)
- [Assignments Included](#assignments-included)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [How to Run](#how-to-run)
- [Datasets](#datasets)
- [Learning Outcomes](#learning-outcomes)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About the Project

This project showcases multiple AI assignments implemented in notebook format. The notebooks are focused on:

- Building and training deep learning models from scratch
- Understanding core architectures like CNNs and GANs
- Applying NLP pipelines for sentiment analysis
- Working with standard benchmark datasets

The main goal is educational: to demonstrate practical implementation and experimentation with modern AI workflows.

## Project Highlights

- End-to-end notebook-based experiments
- Computer vision tasks with MNIST and CIFAR-10
- Sentiment analysis tasks using IMDB and Twitter text data
- Introductory GAN implementation for image generation
- Reproducible workflow suitable for Google Colab and local Jupyter environments

## Repository Structure

```text
Season_of_AI/
|-- README.md
|-- requirements.txt
|-- LICENSE
|-- Week_2_Assignment_1_Handwritte_Digit_Recognition_System.ipynb
|-- Week_2_Assignment_2_Image_classification_using_CNN_(CIFAR10_dataset).ipynb
|-- Week_4_Assignment_1_IMDB_Reviews.ipynb
|-- Week_4_Assignment_2_Twitter_Sentiment_Analysis_using_NLP.ipynb
|-- Week_6_first_GAN_simple_project.ipynb
```

## Assignments Included

| Week | Notebook | Topic | Summary |
|------|----------|-------|---------|
| 2 | `Week_2_Assignment_1_Handwritte_Digit_Recognition_System.ipynb` | Handwritten Digit Recognition | Train a model to classify handwritten digits (MNIST). |
| 2 | `Week_2_Assignment_2_Image_classification_using_CNN_(CIFAR10_dataset).ipynb` | Image Classification with CNN | Build a CNN for CIFAR-10 image classification. |
| 4 | `Week_4_Assignment_1_IMDB_Reviews.ipynb` | IMDB Sentiment Analysis | Predict movie review sentiment using NLP techniques. |
| 4 | `Week_4_Assignment_2_Twitter_Sentiment_Analysis_using_NLP.ipynb` | Twitter Sentiment Analysis | Perform tweet-level sentiment classification. |
| 6 | `Week_6_first_GAN_simple_project.ipynb` | Simple GAN Project | Basic GAN setup for synthetic image generation. |

## Tech Stack

- Language: Python
- Notebook Environment: Jupyter Notebook / Google Colab
- Deep Learning Libraries: TensorFlow, Keras, PyTorch (assignment dependent)
- Data Processing: NumPy, Pandas, scikit-learn
- Visualization: Matplotlib, Seaborn

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda
- Jupyter Notebook or JupyterLab

### Clone the Repository

```bash
git clone https://github.com/imanurag2311/Season_of_AI.git
cd Season_of_AI
```

### Install Dependencies

Since this repository contains multiple notebooks, install common dependencies first:

```bash
pip install -r requirements.txt
```

If any notebook requires extra libraries, install them as prompted in notebook cells.

## How to Run

### Option 1: Local Jupyter

```bash
jupyter notebook
```

Open and run any notebook from the repository root.

### Option 2: Google Colab

1. Upload the notebook to Google Colab.
2. Enable GPU from Runtime settings if required.
3. Run cells sequentially.

## Datasets

Datasets used across assignments include:

- MNIST
- CIFAR-10
- IMDB Reviews Dataset
- Twitter text dataset (as used in the corresponding notebook)

Most datasets are either loaded directly through framework utilities or prepared within notebook workflows.

## Learning Outcomes

By completing these notebooks, you can expect to strengthen skills in:

- Neural network design and training
- CNN-based image classification
- Text preprocessing and sentiment modeling
- GAN fundamentals (generator-discriminator training)
- Model evaluation and experimentation


## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for full details.

## Acknowledgments

- MLSC Amity Noida for learning support and community
- Open-source ecosystem: TensorFlow, PyTorch, Keras, scikit-learn
- Dataset providers and research communities


