# Bird Classifier

This project is a bird classification system that distinguishes between images of birds and non-bird images. It leverages a pre-trained deep learning model, ResNet-18, fine-tuned on a small dataset of bird and forest images.

## Features

- Automatically searches, downloads, and resizes images based on search terms.
- Trains a bird classifier model using transfer learning.
- Supports predictions on new images.
- Saves and loads the trained model for future use.

## Requirements

This project uses `conda` and `pip` for environment setup. The primary dependencies include `fastai`, `torch`, `torchvision`, `fastdownload`, `pillow`, and `duckduckgo_search`.

## Installation

1. **Create and Activate Conda Environment**

   ```bash
   conda create -n bird_classifier python=3.9
   conda activate bird_classifier
   ```

2. **Install Dependencies**

   ```bash
   pip install fastai torch torchvision fastdownload pillow duckduckgo_search
   ```

3. **Clone the Repository and Set Up Project Structure**

   ```bash
   git clone <repository-url>
   cd bird_classifier
   mkdir -p data bird_classifier/models
   ```

## Usage

1. **Run `bird_classifier.py`**

   This script sets up directories, downloads images, trains a classifier, and saves the model for future use.

   ```bash
   python bird_classifier.py
   ```

2. **Testing a New Image**

   Place a test image (e.g., `test.jpg`) in the `data` folder. Run the script, which will output the predicted class and the probability.

   ### Code Overview

   Below are key functions in `bird_classifier.py` and their roles.

## Code Walkthrough

### 1. Search for Images

```python
def search_images(term, max_images=30):
    ddgs = DDGS()
    return [img['image'] for img in ddgs.images(
        term,
        max_results=max_images
    )]
```

This function uses DuckDuckGo to search for images based on a search term. It returns a list of URLs for the specified number of images.

### 2. Download Images

```python
def download_images(path, urls):
    for i, url in enumerate(urls):
        try:
            download_url(url, f"{path}/{i}.jpg", show_progress=False)
        except Exception as e:
            print(f'Error downloading {url}: {e}')
```

Downloads each image from the provided URLs and saves them to the specified path. Errors in downloading are caught and printed.

### 3. Set Up Directories and Process Images

```python
def setup_directories():
    path = Path('data/bird_or_not')
    searches = ['forest', 'bird']

    for o in searches:
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True)
        urls = search_images(f'{o} photo')
        download_images(dest, urls)
        time.sleep(5)
        resize_images(path/o, max_size=400, dest=path/o)

    return path
```

This function creates the necessary folder structure and searches, downloads, and resizes images for each category ("forest" and "bird").

### 4. Train Model

```python
def train_model(path):
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)
    return learn
```

Defines a `DataBlock` to process and load the dataset, then trains a ResNet-18 model on it using transfer learning.

### 5. Save and Load Model

```python
def save_model(learn, model_path='models'):
    """Save the trained model to disk"""
    path = Path(model_path)
    path.mkdir(exist_ok=True)
    learn.export(path/'bird_classifier.pkl')

def load_model(path='models/bird_classifier.pkl'):
    """Load a previously trained model"""
    if Path(path).exists():
        return load_learner(path)
    return None
```

These functions save and load the trained model, so it can be reused without retraining.

### 6. Predict on a New Image

```python
def predict_image(learn, image_path):
    img = PILImage.create(image_path)
    pred_class,_,probs = learn.predict(img)
    return pred_class, probs[0]
```

Loads an image, runs it through the model, and returns the predicted class and probability.

## Main Script Execution

```python
if __name__ == "__main__":
    model_path = 'models/bird_classifier.pkl'

    # Try to load existing model first
    learn = load_model(model_path)

    # If no model exists, train a new one
    if learn is None:
        # Setup and download images
        path = setup_directories()

        # Clean failed downloads
        failed = verify_images(get_image_files(path))
        failed.map(Path.unlink)

        # Train model
        learn = train_model(path)

        # Save the trained model
        save_model(learn)

    # Test prediction
    test_image = "data/test.jpg"
    if Path(test_image).exists():
        pred_class, prob = predict_image(learn, test_image)
        print(f"Prediction: {pred_class}")
        print(f"Probability: {prob:.4f}")
```

This section:

1. Loads an existing model, or trains and saves a new model if none exists.
2. Uses the model to make predictions on a test image, displaying the result.

---

## License

This project is licensed under the MIT License.
