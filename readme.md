# CANDLE: Classification And Noise Detection With Local Embedding Approximations

This repository contains a demo implementation of the CANDLE classifier from the paper with above title.
The code provided here consists of a small collection of plotting functions based on `plotly`, an example implementation of the CANDLE classifier, and a demo application `curves_demo.py` that generates a demo data set of two curves with added noise (similar to the example in the paper), runs the CANDLE and reference classifiers, and generates the following output:

- Accuracy values of the CANDLE and reference classifiers on the demo data set.
- An interactive scatter plot that shows the original dataset colored by their respective class labels.
- An interactive scatter plot that shows the CANDLE classifications including Noise- and Undecided-Labels.

The parameters of all classifiers as well as the data set generator can be easily changed in code with little to no knowledge of Python.

## Setup

Simply clone the repository to your local drive and install the required packages via

`python3 -m pip install numpy tqdm scipy sklearn plotly`

or whatever you are using for package installation (conda etc.).
Afterwards, you can run the demo experiment with

`python3 curves_demo.py`

which will run the experiment, giving you a very rough progress report, and plot the results in your browser with `plotly`.

This is by no means optimized code.
It is all written in Python and rewriting codes or JIT compiling the heavy-load functions using, e.g., numba, can certainly improve on the performance largely.
This is a proof-of-concept implementation that can be used as a reference implementation or to run small experiments and must not be used for speed comparisons.
If you find bugs, feel free to contact me.
Changing parameters requires extremely little knowledge of Python and should be very accessible to anyone, who has access to the paper.
Changing the data set used in the experiments requires a bit of Python knowledge but not a lot.

Have fun experimenting.

