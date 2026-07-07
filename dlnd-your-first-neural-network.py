import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Your first neural network

    In this project, you'll build your first neural network and use it to predict daily bike rental ridership. We've provided some of the code, but left the implementation of the neural network up to you (for the most part). After you've submitted this project, feel free to explore the data and the model more.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = 'retina'

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and prepare the data

    A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, we've written the code to load and prepare the data. You'll learn more about this soon!
    """)
    return


@app.cell
def _(pd):
    data_path = 'Bike-Sharing-Dataset/hour.csv'

    rides = pd.read_csv(data_path)
    return data_path, rides


@app.cell
def _(rides):
    rides.tail()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking out the data

    This dataset has the number of riders for each hour of each day from January 1 2011 to December 31 2012. The number of riders is split between casual and registered, summed up in the `cnt` column. You can see the first few rows of the data above.

    Below is a plot showing the number of bike riders over the first 10 days in the data set. You can see the hourly rentals here. This data is pretty complicated! The weekends have lower over all ridership and there are spikes when people are biking to and from work during the week. Looking at the data above, we also have information about temperature, humidity, and windspeed, all of these likely affecting the number of riders. You'll be trying to capture all this with your model.
    """)
    return


@app.cell
def _(rides):
    rides[:24*10].plot(x='dteday', y='cnt')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dummy variables
    Here we have some categorical variables like season, weather, month. To include these in our model, we'll need to make binary dummy variables. This is simple to do with Pandas thanks to `get_dummies()`.
    """)
    return


@app.cell
def _(pd, rides):
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for _each in dummy_fields:
        dummies = pd.get_dummies(rides[_each], prefix=_each, drop_first=False)
        rides_1 = pd.concat([rides, dummies], axis=1)
    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides_1.drop(fields_to_drop, axis=1)
    data.head()
    return data, rides_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scaling target variables
    To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.

    The scaling factors are saved so we can go backwards when we use the network for predictions.
    """)
    return


@app.cell
def _(data):
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # Ensure these columns are float so assigning scaled values (floats) will succeed
    data[quant_features] = data[quant_features].astype('float64')
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for _each in quant_features:
        _mean, _std = (data[_each].mean(), data[_each].std())
        scaled_features[_each] = [_mean, _std]
        data.loc[:, _each] = (data[_each] - _mean) / _std
    return (scaled_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Splitting the data into training, testing, and validation sets

    We'll save the last 21 days of the data to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual number of riders.
    """)
    return


@app.cell
def _(data):
    # Save the last 21 days 
    test_data = data[-21 * 24:]
    data_1 = data[:-21 * 24]
    target_fields = ['cnt', 'casual', 'registered']
    # Separate the data into features and targets
    features, targets = (data_1.drop(target_fields, axis=1), data_1[target_fields])
    test_features, test_targets = (test_data.drop(target_fields, axis=1), test_data[target_fields])
    return features, targets, test_data, test_features, test_targets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll split the data into two sets, one for training and one for validating as the network is being trained. Since this is time series data, we'll train on historical data, then try to predict on future data (the validation set).
    """)
    return


@app.cell
def _(features, targets):
    # Hold out the last 60 days of the remaining data as a validation set
    train_features, train_targets = features[:-60*24], targets[:-60*24]
    val_features, val_targets = features[-60*24:], targets[-60*24:]
    return train_features, train_targets, val_features, val_targets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time to build the network

    Below you'll build your network. We've built out the structure and the backwards pass. You'll implement the forward pass through the network. You'll also set the hyperparameters: the learning rate, the number of hidden units, and the number of training passes.

    The network has two layers, a hidden layer and an output layer. The hidden layer will use the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. That is, the activation function is $f(x)=x$. A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called *forward propagation*.

    We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called *backpropagation*.

    > **Hint:** You'll need the derivative of the output activation function ($f(x) = x$) for the backpropagation implementation. If you aren't familiar with calculus, this function is equivalent to the equation $y = x$. What is the slope of that equation? That is the derivative of $f(x)$.

    Below, you have these tasks:
    1. Implement the sigmoid function to use as the activation function. Set `self.activation_function` in `__init__` to your sigmoid function.
    2. Implement the forward pass in the `train` method.
    3. Implement the backpropagation algorithm in the `train` method, including calculating the output error.
    4. Implement the forward pass in the `run` method.
    """)
    return


@app.cell
def _(np):
    class NeuralNetwork(object):

        def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):  # Set number of nodes in input, hidden and output layers.
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes
            self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** (-0.5), (self.hidden_nodes, self.input_nodes))
            self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** (-0.5), (self.output_nodes, self.hidden_nodes))  # Initialize weights
            self.lr = learning_rate
            self.activation_function = self.sigmoid

        def sigmoid(self, x):
            """
            Numerically-safe sigmoid: coerce x to a numpy array of floats first.
            """
            x = np.asarray(x, dtype=float)
            return 1.0 / (1.0 + np.exp(-x))  # Activation function (can be swapped if you want)

        def train(self, inputs_list, targets_list):
            inputs = np.array(inputs_list, ndmin=2).T.astype(float)
            targets = np.array(targets_list, ndmin=2).T.astype(float)
            hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
            final_outputs = final_inputs
            output_errors = targets - final_outputs
            hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
            hidden_grad = hidden_outputs * (1.0 - hidden_outputs)  # Convert inputs/targets to 2d numpy arrays of float dtype
            self.weights_hidden_to_output = self.weights_hidden_to_output + self.lr * np.dot(output_errors, hidden_outputs.T)
            self.weights_input_to_hidden = self.weights_input_to_hidden + self.lr * np.dot(hidden_errors * hidden_grad, inputs.T)

        def run(self, inputs_list):  # Forward pass
            inputs = np.array(inputs_list, ndmin=2).T.astype(float)
            hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
            final_outputs = final_inputs  # linear output (identity)
            return final_outputs  # Backward pass  # propagate errors to hidden layer  # note: weights_hidden_to_output shape is (output_nodes, hidden_nodes)  # Update the weights  # (use correct transposes so shapes align)  # Convert inputs to column vector of floats

    return (NeuralNetwork,)


@app.cell
def _(np):
    def MSE(y, Y):
        return np.mean((y-Y)**2)

    return (MSE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the network

    Here you'll set the hyperparameters for the network. The strategy here is to find hyperparameters such that the error on the training set is low, but you're not overfitting to the data. If you train the network too long or have too many hidden nodes, it can become overly specific to the training set and will fail to generalize to the validation set. That is, the loss on the validation set will start increasing as the training set loss drops.

    You'll also be using a method know as Stochastic Gradient Descent (SGD) to train the network. The idea is that for each training pass, you grab a random sample of the data instead of using the whole data set. You use many more training passes than with normal gradient descent, but each pass is much faster. This ends up training the network more efficiently. You'll learn more about SGD later.

    ### Choose the number of epochs
    This is the number of times the dataset will pass through the network, each time updating the weights. As the number of epochs increases, the network becomes better and better at predicting the targets in the training set. You'll need to choose enough epochs to train the network well but not too many or you'll be overfitting.

    ### Choose the learning rate
    This scales the size of weight updates. If this is too big, the weights tend to explode and the network fails to fit the data. A good choice to start at is 0.1. If the network has problems fitting the data, try reducing the learning rate. Note that the lower the learning rate, the smaller the steps are in the weight updates and the longer it takes for the neural network to converge.

    ### Choose the number of hidden nodes
    The more hidden nodes you have, the more accurate predictions the model will make. Try a few different numbers and see how it affects the performance. You can look at the losses dictionary for a metric of the network performance. If the number of hidden units is too low, then the model won't have enough space to learn and if it is too high there are too many options for the direction that the learning can take. The trick here is to find the right balance in number of hidden units you choose.
    """)
    return


@app.cell
def _(
    MSE,
    NeuralNetwork,
    train_features,
    train_targets,
    val_features,
    val_targets,
):
    import sys

    ### Set the hyperparameters here ###
    epochs = 100 * 6
    learning_rate = 0.0001
    hidden_nodes = 2*12
    output_nodes = 1

    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

    losses = {'train':[], 'validation':[]}
    for e in range(epochs):

        # Use DataFrame.sample to get a random batch (cleaner and less error-prone)
        # sample a random batch of 128 rows
        batch_df = train_features.sample(n=128)
        batch_targets = train_targets.loc[batch_df.index, 'cnt'].values

        # iterate the sampled rows directly
        for record, target in zip(batch_df.values, batch_targets):
            network.train(record, target)


        # Printing out the training progress
        train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
        sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                         + "% ... Training loss: " + str(train_loss)[:5] \
                         + " ... Validation loss: " + str(val_loss)[:5])

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)
    return losses, network


@app.cell
def _(losses, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    # ensure numeric numpy arrays (helps avoid silent failures)
    train_losses = np.asarray(losses['train'], dtype=float)
    val_losses = np.asarray(losses['validation'], dtype=float)
    if train_losses.size == 0 or val_losses.size == 0:
        raise ValueError('One of the loss arrays is empty — check training loop/appends')
    _ax.plot(train_losses, label='Training loss', linewidth=2, alpha=0.9)
    _ax.plot(val_losses, label='Validation loss', linewidth=2, alpha=0.9)
    _ax.set_xlabel('Epoch')
    _ax.set_ylabel('Loss (MSE)')
    _ax.set_title('Training and Validation Loss')
    _ax.legend(loc='upper right')
    _ax.grid(True, linestyle='--', alpha=0.4)
    _fig.tight_layout()
    # ax.set_ylim(top=0.5)      # same as plt.ylim(ymax=0.5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check out your predictions

    Here, use the test data to view how well your network is modeling the data. If something is completely wrong here, make sure each step in your network is implemented correctly.
    """)
    return


@app.cell
def _(
    network,
    np,
    pd,
    plt,
    rides_1,
    scaled_features,
    test_data,
    test_features,
    test_targets,
):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _mean, _std = scaled_features['cnt']
    predictions = network.run(test_features) * _std + _mean
    _ax.plot(predictions[0], label='Prediction')
    _ax.plot((test_targets['cnt'] * _std + _mean).values, label='Data')
    _ax.set_xlim(right=len(predictions))
    _ax.legend()
    dates = pd.to_datetime(rides_1.iloc[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    _ax.set_xticks(np.arange(len(dates))[12::24])
    _ = _ax.set_xticklabels(dates[12::24], rotation=45)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Thinking about your results

    Answer these questions about your results. How well does the model predict the data? Where does it fail? Why does it fail where it does?

    > **Note:** You can edit the text in this cell by double clicking on it. When you want to render the text, press control + enter

    #### Your answer below
    I think that the model makes a fare prediction. It does not improve when training it with to large number of epochs. It seems that the training dataset have overfitted the model. So when the graph show the later days of december the predictions gets a little to optimistic.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unit tests

    Run these unit tests to check the correctness of your network implementation. These tests must all be successful to pass the project.
    """)
    return


@app.cell
def _(NeuralNetwork, data_path, np, pd, rides_1):
    import unittest
    inputs = [0.5, -0.2, 0.1]
    targets_1 = [0.4]
    test_w_i_h = np.array([[0.1, 0.4, -0.3], [-0.2, 0.5, 0.2]])
    test_w_h_o = np.array([[0.3, -0.1]])

    class TestMethods(unittest.TestCase):

        def test_data_path(self):
            self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
      ##########
        def test_data_loaded(self):  # Unit tests for data loading
            self.assertTrue(isinstance(rides_1, pd.DataFrame))  ##########

        def test_activation(self):
            network = NeuralNetwork(3, 2, 1, 0.5)  # Test that file path to dataset has been unaltered
            self.assertTrue(np.all(network.activation_function(0.5) == 1 / (1 + np.exp(-0.5))))

        def test_train(self):
            network = NeuralNetwork(3, 2, 1, 0.5)  # Test that data frame loaded
            network.weights_input_to_hidden = test_w_i_h.copy()
            network.weights_hidden_to_output = test_w_h_o.copy()
            network.train(inputs, targets_1)  ##########
            self.assertTrue(np.allclose(network.weights_hidden_to_output, np.array([[0.37275328, -0.03172939]])))  # Unit tests for network functionality
            self.assertTrue(np.allclose(network.weights_input_to_hidden, np.array([[0.10562014, 0.39775194, -0.29887597], [-0.20185996, 0.50074398, 0.19962801]])))  ##########

        def test_run(self):
            network = NeuralNetwork(3, 2, 1, 0.5)
            network.weights_input_to_hidden = test_w_i_h.copy()  # Test that the activation function is a sigmoid
            network.weights_hidden_to_output = test_w_h_o.copy()
            self.assertTrue(np.allclose(network.run(inputs), 0.09998924))
    suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
    unittest.TextTestRunner().run(suite)  # Test that weights are updated correctly on training  # Test correctness of run method
    return


if __name__ == "__main__":
    app.run()
