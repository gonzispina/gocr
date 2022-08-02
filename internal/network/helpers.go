package network

import (
	"github.com/gonzispina/gocr/kit/vector"
	"math"
	"math/rand"
)

// s(x) = 1 / (1 + e^(-x))
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// the derivative of the sigmoid function: s'(x) = s(x) * (1 - s(x))
func sigmoidPrime(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

type Input struct {
	Input    vector.Vector
	Expected vector.Vector
}

// Batch of training data
type Batch []*Input

// Batches of training data
type Batches []Batch

func splitIntoBatches(trainingData []*Input, batchSize int) Batches {
	amount := len(trainingData) / (batchSize)
	remainder := len(trainingData) % batchSize
	res := make(Batches, int(math.Floor(float64(amount))))
	for i := 0; i < amount; i++ {
		from := batchSize * i
		to := batchSize * (i + 1)
		if i+1 == amount {
			to += remainder
		}
		res[i] = trainingData[from:to]
	}
	return res
}

func shuffleSlice[T any](slice []T) []T {
	for i := range slice {
		j := rand.Intn(i + 1)
		slice[i], slice[j] = slice[j], slice[i]
	}
	return slice
}

func initializeZeroWeightsAndBiases(sizes []int) ([]vector.Vector, [][]vector.Vector) {
	biases := make([]vector.Vector, len(sizes)-1)
	weights := make([][]vector.Vector, len(sizes)-1)
	for i := 1; i < len(sizes); i++ {
		biases[i-1] = vector.CreateZero(sizes[i])
		weights[i-1] = make([]vector.Vector, sizes[i])
		for j := 0; j < sizes[i]; j++ {
			weights[i-1][j] = vector.CreateZero(sizes[i-1])
		}
	}
	return biases, weights
}
