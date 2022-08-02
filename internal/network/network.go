package network

import (
	"errors"
	"fmt"
	"github.com/gonzispina/gocr/kit/vector"
)

// NewRandom constructor
func NewRandom(sizes []int) *Network {
	return &Network{
		layers: len(sizes) - 1,
		sizes:  sizes,
		biases: vector.CreateManyNormalRandom(sizes[1:]),
		weights: func() [][]vector.Vector {
			res := make([][]vector.Vector, len(sizes)-1)
			for i := 1; i < len(sizes); i++ {
				res[i-1] = vector.CreateManyFixedSizeNormalRandom(sizes[i], sizes[i-1])
			}
			return res
		}(),
	}
}

// Network implementation
type Network struct {
	layers  int
	sizes   []int
	biases  []vector.Vector
	weights [][]vector.Vector
}

func (n *Network) FeedForward(input vector.Vector) (vector.Vector, error) {
	if len(input) != n.sizes[0] {
		return nil, errors.New(fmt.Sprintf("invalid input size %v, expected %v", len(input), n.sizes[0]))
	}

	for i := 0; i < n.layers; i++ {
		// layer
		a := make(vector.Vector, n.sizes[i+1])
		for j := 0; j < n.sizes[i+1]; j++ {
			a[j] = sigmoid(vector.Dot(input, n.weights[i][j]) + n.biases[i][j])
		}

		if i+1 == n.layers {
			// We reached the last layer so we return its output
			return a, nil
		} else {
			// The output of the last layer becomes the Input of the next one
			input = a
		}
	}

	// Unreachable
	return nil, nil
}

// StochasticGradientDescent for training the network
func (n *Network) StochasticGradientDescent(trainingData []*Input, batchSize int, learningRate float64) {
	trainingData = shuffleSlice(trainingData)
	batches := splitIntoBatches(trainingData, batchSize)

	// Process every batch
	for _, batch := range batches {
		// We set M the batch size we are going to use to compute deltas
		M := float64(len(batch))
		sdgRatio := -learningRate / M

		biasDeltas := make([]vector.Vector, len(n.sizes)-1)
		weightDeltas := make([][]vector.Vector, len(n.sizes)-1)
		for i := 1; i < len(n.sizes); i++ {
			biasDeltas[i-1] = vector.CreateZero(n.sizes[i])
			weightDeltas[i-1] = make([]vector.Vector, n.sizes[i])
			for j := 0; j < n.sizes[i]; j++ {
				weightDeltas[i-1][j] = vector.CreateZero(n.sizes[i-1])
			}
		}

		for _, trainingInput := range batch {
			input := trainingInput.Input
			expected := trainingInput.Expected

			// First we compute the Z values and the activations of every neuron of the network

			// The activation values after computing the activation function of all the layers
			var activations []vector.Vector
			activations = append(activations, input)

			// The values of the layers without computing the activation function
			zs := make([]vector.Vector, n.layers)
			for i := 0; i < n.layers; i++ {
				z := make(vector.Vector, n.sizes[i+1])
				activation := make(vector.Vector, n.sizes[i+1])

				for j := 0; j < n.sizes[i+1]; j++ {
					z[j] = vector.Dot(activations[i], n.weights[i][j]) + n.biases[i][j]
					activation[j] = sigmoid(z[j])
				}

				zs[i] = z
				activations = append(activations, activation)
			}

			// last layer index
			lli := n.layers - 1

			// Now we calculate the cost derivative with respect to every weight and bias in the network and
			// subtract that from every weight and bias of the network.
			cost := vector.Substract(expected, activations[n.layers])
			sp := vector.Apply(zs[lli], sigmoidPrime)
			deltas := vector.Hadamard(cost, sp)

			biasDeltas[lli] = vector.Add(biasDeltas[lli], deltas)
			for j := 0; j < n.sizes[n.layers]; j++ {
				neuronDeltas := vector.Scale(activations[lli], deltas[j])
				weightDeltas[lli][j] = vector.Add(weightDeltas[lli][j], neuronDeltas)
			}

			// From the last hidden layer the first hidden one, 'cause the first one is the input one
			for i := lli - 1; i >= 0; i-- {
				z := zs[i]
				sp = vector.Apply(z, sigmoidPrime)

				// We transpose the next layer weights and calculate the new deltas
				newDeltas := make(vector.Vector, n.sizes[i+1])
				for j := 0; j < n.sizes[i+1]; j++ {
					for k := 0; k < n.sizes[i+2]; k++ {
						newDeltas[j] += deltas[k] * n.weights[i+1][k][j] * sp[j]
					}

					neuronDeltas := vector.Scale(activations[i], newDeltas[j])
					weightDeltas[i][j] = vector.Add(weightDeltas[i][j], neuronDeltas)
				}

				deltas = newDeltas
				biasDeltas[i] = vector.Add(biasDeltas[i], deltas)
			}
		}

		for i := 0; i < n.layers; i++ {
			n.biases[i] = vector.Substract(n.biases[i], vector.Scale(biasDeltas[i], sdgRatio))

			for j := 0; j < n.sizes[i+1]; j++ {
				n.weights[i][j] = vector.Substract(n.weights[i][j], vector.Scale(weightDeltas[i][j], sdgRatio))
			}
		}
	}
}
