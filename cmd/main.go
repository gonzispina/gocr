package main

import (
	"fmt"
	"github.com/gonzispina/gocr/internal/mnist"
	"github.com/gonzispina/gocr/internal/network"
	"github.com/gonzispina/gocr/kit/vector"
	"os"
	"path"
)

func main() {
	wd, _ := os.Getwd()
	dataPath := path.Join(wd, "data")

	data, err := mnist.ReadTrainSet(dataPath)
	if err != nil {
		panic(err)
	}

	net := network.NewRandom([]int{784, 30, 10})

	// Train the net
	trainingData := data[0:50000]
	testData := data[50001:]
	epochs := 30

	fmt.Printf("Training... \n")
	for k := 0; k < epochs; k++ {
		/*
			The idea is that for every epoch we run, we shuffle the data and create the batches for
			the stochastic gradient descent. For every batch we process we are going to propagate backwards
			the delta in the gradient and adjust every weight and bias.
		*/

		net.StochasticGradientDescent(trainingData, 10, 3.0)
		correct := 0
		for _, i := range testData {
			res, _ := net.FeedForward(i.Input)
			if vector.MaxKey(res) == vector.MaxKey(i.Expected) {
				correct++
			}
		}

		fmt.Printf("After %v epochs: %v / %v answers are correct \n", k+1, correct, len(testData))
	}

}
