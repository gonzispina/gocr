package main

import (
	"github.com/gonzispina/gocr/internal/mnist"
	"github.com/gonzispina/gocr/internal/network"
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

	net := network.New([]int{728, 30, 10})
	net.StochasticGradientDescent(data[0:50000], 30, 10, 3, data[50001:])
}
