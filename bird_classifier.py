# Updated bird_classifier.py
from fastai.vision.all import *
from fastdownload import download_url
from duckduckgo_search import DDGS  # Updated import
import time
from pathlib import Path

def search_images(term, max_images=30):
    ddgs = DDGS()
    return [img['image'] for img in ddgs.images(
        term, 
        max_results=max_images
    )]

# Rest of the code remains same
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
    
    return path


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

def predict_image(learn, image_path):
    img = PILImage.create(image_path)
    pred_class,_,probs = learn.predict(img)
    return pred_class, probs[0]

# Add these functions to bird_classifier.py

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

# Update the main execution block
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