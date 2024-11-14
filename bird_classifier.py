from fastai.vision.all import *
from fastdownload import download_url
from duckduckgo_search import DDGS
import time
from pathlib import Path

def search_images(term, max_images=30):
    ddgs = DDGS()
    return [img['image'] for img in ddgs.images(term, max_results=max_images)]

def download_images(path, urls):
    for i, url in enumerate(urls):
        try:
            download_url(url, f"{path}/{i}.jpg", show_progress=False)
        except Exception as e:
            print(f'Error downloading {url}: {e}')

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
        
        # Debugging: Print the number of images downloaded
        print(f"Downloaded {len(list(dest.glob('*.jpg')))} images to {dest}")
    
    return path

def train_model(path):
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)
    
    # Debugging: Check if DataLoaders has valid data
    if len(dls.train) == 0 or len(dls.valid) == 0:
        raise ValueError("DataLoaders are empty. Ensure images are in the correct path and loaded properly.")
    
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(3)
    return learn

def predict_image(learn, image_path):
    img = PILImage.create(image_path)
    pred_class, _, probs = learn.predict(img)
    return pred_class, probs[0]

def save_model(learn, model_path='models'):
    path = Path(model_path)
    path.mkdir(exist_ok=True)
    learn.export(path/'bird_classifier.pkl')

def load_model(path='models/bird_classifier.pkl'):
    if Path(path).exists():
        return load_learner(path)
    return None

if __name__ == "__main__":
    model_path = 'models/bird_classifier.pkl'
    
    # Attempt to load an existing model
    learn = load_model(model_path)
   
    # If no saved model exists, train a new one
    if learn is None:
        path = setup_directories()
        
        # Clean up any corrupted downloads
        failed = verify_images(get_image_files(path))
        failed.map(Path.unlink)
        
        # Train model and save
        learn = train_model(path)
        save_model(learn)
    
    # Verify the model and calculate accuracy if available
    if learn is not None:
        # Check if validation data is available before computing accuracy
        if len(learn.dls.valid) > 0:
            accuracy = learn.validate()[1].item()  # Accuracy is the second item
            print(f"Model accuracy on validation set: {accuracy:.4f}")
        else:
            print("No validation data found.")

    # Test prediction on a sample image
    test_image = "data/test.jpg"
    if Path(test_image).exists():
        pred_class, prob = predict_image(learn, test_image)
        print(f"Prediction: {pred_class}")
        print(f"Probability: {prob:.4f}")
