# Energy Consumption Prediction

This project is to create a machine learning model to predict energy consumption usage using historical data. 

The dataset used for this project is obtained from a Tesla Powerwall. Data points are at 5 mins interval. I am using Long Short Team Model (LSTM) as this seems to be best model for time-series data prediction.

This is my first time building a machine learning model. I have gone through serval iterations of the implementation. I initially built the model using PyTorch's LSTM module (well, that was the easy part.), I tried to use [fastai](https://docs.fast.ai)'s [Learner](https://docs.fast.ai/learner.html) for the training logic. I figured, why implement my own training loop if I can reuse an exsiting framework. So why fastai? That's probably because I had been watching fastai's deep learning course and that was the first framework I was exposed to. But soon, I was hitting with roadblocks after roadblocks with the fastai's Learner. fastai is an incredible framework. It can do a lot of things. But, the framework is just too complicated and built in too much magic to my liking.

Next, I implemented my own Learner class. After all, the training loop is not complicated. AI can do all the heavy lifting, right? I had a working Learner implementation within a couple of days. I was able to use it to train the model and got some reasonble results.

Then, I came across PyTorch [Lightning](https://lightning.ai/docs/pytorch/stable/). It is a framework for building ML models. Just like fastai, it has the training logic, able to track training and validation losses, model checkpoints (saving/loading models), and more. Lightning has a much cleaner design which is much easier to follow. I can take my pyTorch model and data processing logic, and wrap that with Lightning's Module and DataModule, and they just works with Lightning's Trainer. Unfortuniately, this is something I couldn't figure out using fastai.

I spent the next day porting my code over to use Lightning. I ditched my own Learner implementation, implemented the LSTM as LightningModule and moved the data preparation logic in a DataModule. After running the Lightning's Trainer with a few epochs, as advertised, I can visualise the train/val loss with [tensorboard](https://www.tensorflow.org/tensorboard), not to mention that the trained model is automatically saved to a local directory. I could just load the model for inference. Without Lightning, I would have had to write my own model serialization code!

The code for prediction was a bit more trickier than anticipated. Perhaps it was because of the time-series data prodiction, I needed to prepare the input data in a sequence of "previous" time steps and I had to normalise the input using the same "fitted" scalers I used for training. Conveniently, I was able to save the fitted scalers as part of the model checkpoints so I didn't need to implement and manage another file. All the data I need for inference are in one checkpoint file.

Visualising the time-series data was also a fun exercise. `matplotlib` and `seaborn` plot the graphs with ease. But there are tons of other bells and whistles and I barely scratched the surface!

Check out the notebook on how I develop and train the model.


## Usage

### Data Preparation

The input data is expected to be in a csv file `data/processed/time_series_data.csv`. The path is specified in the `config.yml` (among other configuration). The data is expected to have columns on `timestamp`, `month`, `day_of_week`, `time_of_day`, `total_home_usage`.

When training the model, we can specify how many features we are going to use for training. If we have additional data such as weather temperature, humidity, etc, we can add to the data file, specific the additionl feature columns in `config.yml`, the training code will use the additional features for training.

If we change the number features, we also need to change `input_size` to match.

`mins_per_time_step` is the time resolution between data points (time steps) in the time series data. The data I obtained from Powerwall is 5min interval.

`sequence_length` is the number of time steps to use for training. I used 24 hours worth of tiime steps, ie 24 * 60 / 5 == 288 steps.


### Configuration/Hyperparameters

This project uses [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html). It is very flexible in configuring how to run the trainer. The configurations that were used to train to model are in `config.yml`. To train using the config file, run this command line.

```bash
python -m src.trainer fit --config src/config.yml
```

The best checkpoint file is saved to `./lightning_logs/version_3/checkpoints/`. This can be loaded into the model for inference.

To predict the next 7 days' consumption from a start date, use the following command. The predictiion is saved to a plot file.  

```bash

CHECKPOINT=./lightning_logs/version_8/checkpoints/best-checkpoint-epoch=17-val_loss=0.18.ckpt
python -m src.predict --model-checkpoint-path $CHECKPOINT --start-date "2024-10-01" --predict-days 7

```

