## deep-learning-from-scratch-c
Implement the sample sources in deep-learning-from-scratch ([ゼロから作る Deep Learning](https://github.com/oreilly-japan/deep-learning-from-scratch)) in C.

## Background

[ゼロから作る Deep Learning](https://github.com/oreilly-japan/deep-learning-from-scratch) is a great book, but it uses too useful libraries such as NumPy.
Therefore, there is a possibility that it will only make us feel like we completely understand Deep Learning, and actually don't.

So I attempted to implement Deep Learning from a little closer to "zero" by implementing the sample code in the book in C.

In order to make it easier to understand, I have implemented it in a straightforward manner, without aiming for high speed.

## Requirements
You need gnuplot to draw graphs and GoogleTest to run tests.

## Build and run
Go to the folder for each chapter and execute `make` , and run binary.

Example:

```
$ cd ch05
$ make
$ ./train_neuralnet 
train acc, test acc | 0.129067, 0.132400
train acc, test acc | 0.901400, 0.906300
train acc, test acc | 0.921300, 0.925100
train acc, test acc | 0.938067, 0.938500
train acc, test acc | 0.945933, 0.944400
train acc, test acc | 0.951583, 0.949100
train acc, test acc | 0.957733, 0.952500
...
```
