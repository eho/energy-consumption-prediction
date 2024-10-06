<h1 align="center">
  Machine Learning Project Template
  <br />
  <a href="https://github.com/eho/ml-project-template/issues">
    <img
      alt="GitHub issues"
      src="https://img.shields.io/github/issues/eho/ml-project-template?logo=git&style=plastic"
    />
  </a>
  <a href="https://github.com/eho/ml-project-template/network">
    <img
      alt="GitHub forks"
      src="https://img.shields.io/github/forks/eho/ml-project-template?style=plastic&logo=github"
    />
  </a>
  <a href="https://github.com/eho/ml-project-template/stargazers">
    <img
      alt="GitHub stars"
      src="https://img.shields.io/github/stars/eho/ml-project-template?style=plastic&logo=github"
    />
  </a>
</h1>

## Energy Consumption Prediction

This project is to create a machine learning model to predict energy consumption usage.

I am training the model using power usage data obtained from a Tesla Powerwall. Data points are at 5 mins interval. I am using Long Short Team Model (LSTM) as this seems to be best model for time-series data prediction.

This is my first time building a machine learning model. I have gone through serval iterations of the implementation. I initially built the model using PyTorch's LSTM module (well, that was the easy part.), I tried to use [fastai](https://docs.fast.ai)'s [Learner](https://docs.fast.ai/learner.html) for the training logic. I figured, why implement my own training loop if I can reuse an exsiting framework. So why fastai? That's probably because I had been watching fastai's deep learning course and that was the first framework I was exposed to. But soon, I was hitting with roadblocks after roadblocks with the fastai's Learner. fastai is an incredible framework. It can do a lot of things. But, the framework is just too complicated and built in too much magic to my liking.

Next, I implemented my own Learner class. After all, the training loop is not complicated. AI can do all the heavy lifting, right? I had a working Learner implementation within a couple of days. I was able to use it to train the model and got some reasonble results.

Then, I came across PyTorch [Lightning](https://lightning.ai/docs/pytorch/stable/). It is a framework for building ML models. Just like fastai, it has the training logic, able to track training and validation losses, model checkpoints (saving/loading models), and more. Lightning has a much cleaner design which is much easier to follow. I can take my pyTorch model and data processing logic, and wrap that with Lightning's Module and DataModule, and they just works with Lightning's Trainer. Unfortuniately, this is something I couldn't figure out using fastai.

I spent the next day porting my code over to use Lightning. I ditched my own Learner implementation, implemented the LSTM as LightningModule and moved the data preparation logic in a DataModule. After running the Lightning's Trainer with a few epochs, as advertised, I can visualise the train/val loss with [tensorboard](https://www.tensorflow.org/tensorboard), not to mention that the trained model is automatically saved to a local directory. I could just load the model for inference. Without Lightning, I would have had to write my own model serialization code!

The code for prediction was a bit more trickier than anticipated. Perhaps it was because of the time-series data prodiction, I needed to prepare the input data in a sequence of "previous" time steps and I had to normalise the input using the same "fitted" scalers I used for training. Conveniently, I was able to save the fitted scalers as part of the model checkpoints so I didn't need to implement and manage another file. All the data I need for inference are in one checkpoint file.

Visualising the time-series data was also a fun exercise. `matplotlib` and `seaborn` plot the graphs with ease. But there are tons of other bells and whistles and I barely scratched the surface!

Check out the notebook on how I develop and train the model.
