# Classifier

A simple neural network that can very accurately determine the subject of a photo to be apples, bananas, oranges, and their rotten counterparts.

## Setup
---
### Get the data
You need to ensure there is a dataset. Go to this url: https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification and download the data. Extract it into the root of this repo. Rename the `dataset` SUBDIR to `fruit`.

The resulting file structure should be `dataset` -> `fruit` -> `test` -> `train` with files inside the two test/train directories.

### Environment
- Install `pipenv` if you have not already (Google is your best friend)
- run `pipenv install` from the root of this repo
- run `pipenv shell`


## Running
---
### Trainer
At this point you should be inside the virtual environment. If you have a better GPU than me, you might be able to play around with the batch sizes for more performance; I was limited on memory (GTX 1060, 6GB). You can now run the trainer via `python trainer.py`

Depending on your system this could take a long time. Once it finishes, you'll see two new directories:
- `fruit_model`
- `fruit_model_tweaked_base`

The logs will inform you of which one performs the best. Open `logs/LATEST.log` and see the bottom; it'll show which of the two models performs the best. Also, if there is an old model left over, it'll compare these results with the old model.

### Predictor
If you want to use one of the new models, delete (if it exists) the `_model` directory and rename the desired new model to `_model`. Then simply run `python predictor.py <filename>` where `<filename>` is a path to an image of a fruit (a valid test you'll want to download something off the internet). This should not take long to run at all. It'll first display the image; once you close that window it'll classify it print the prediction to the terminal.