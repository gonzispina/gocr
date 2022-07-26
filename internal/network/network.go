package network

import (
	"fmt"
	"github.com/gonzispina/gocr/kit/vector"
)

// New constructor
func New(sizes []int) *Network {
	return &Network{
		layersCount: len(sizes),
		sizes:       sizes,
		layers: func() []Layer {
			res := make([]Layer, len(sizes))
			for i, size := range sizes {
				res[i] = make(Layer, size)

				// Create the biases randomly
				biases := vector.CreateNormalRandom(size)

				// Create the weights randomly
				for j := 0; j < size; j++ {
					res[i][j] = NewSigmoidNeuron(biases[j], vector.CreateNormalRandom(size))
				}
			}
			return res
		}(),
	}
}

// Network implementation
type Network struct {
	layersCount int
	sizes       []int
	layers      []Layer
}

func (n *Network) FeedForward(input vector.Vector) vector.Vector {
	for i, layer := range n.layers {
		z := layer.Activate(input)
		if i+1 == n.layersCount {
			// We reached the last layer so we return its output
			return z
		} else {
			// The output of the last layer becomes the Input of the next one
			input = z
		}
	}

	// Unreachable
	return nil
}

// StochasticGradientDescent for training the network
func (n *Network) StochasticGradientDescent(trainingData []*Input, epochs int, batchSize int, learningRate float64, testData []*Input) {
	fmt.Printf("Training...")
	for k := 0; k < epochs; k++ {
		/*
			The idea is that for every epoch we run, we shuffle the data and create the batches for
			the stochastic gradient descent. For every batch we process we are going to propagate backwards
			the delta in the gradient and adjust every weight and bias.
		*/

		trainingData = shuffleSlice(trainingData)
		batches := splitIntoBatches(trainingData, batchSize)

		// In the following vectors we are going to store the values of the derivatives of the cost
		// function with respect to the biases and the weights respectively (nabla is the notation used for gradients)
		nablaB := vector.CreateManyZeroVector(n.sizes)
		nablaW := vector.CreateManyZeroVector(n.sizes)

		// Process every batch
		for _, batch := range batches {
			// We set M the batch size we are going to use to compute deltas
			M := float64(len(batch))
			deltaNablaB := vector.CreateManyZeroVector(n.sizes)
			deltaNablaW := vector.CreateManyZeroVector(n.sizes)

			for _, trainingInput := range batch {
				input := trainingInput.Input
				expected := trainingInput.Expected

				/*
					First we need to obtain the deltas to apply to the weights and the biases:
					deltaC = nablaW * deltaW + nablaB * deltaB
						   = (nablaW ; nablaB) . (deltaW ; deltaB)

					Where (nablaW ; nablaB) = nablaC the gradient of our cost function
					But we want to minimize, so the step we are going to take is in the opposite direction of the gradient of the cost function, so:
					(deltaW ; deltaB) = mu * -(nablaW ; nablaB)

					We also have:
					deltaW = (W - W0) where W is the new weight and W0 is the current weight
					deltaB = (B - WV) where B is the new bias and B0 is the current bias

					Then:
					(W - W0) = -mu * nablaW <=> W = W0 -mu * nablaW
					(B - B0) = -mu * nablaB <=> B = B0 -mu * nablaB
				*/

				// The activations of all the layers
				var activations []vector.Vector
				activations = append(activations, input)

				// The values of the layers without computing the activation function
				zs := make([]vector.Vector, n.layersCount)
				for i, layer := range n.layers {
					z := make(vector.Vector, len(layer))
					activation := make(vector.Vector, len(layer))
					for j, neuron := range layer {
						z[j] = vector.Dot(neuron.Weights(), activations[i]) + neuron.Bias()
						activation[j] = sigmoid(z[j])
					}

					zs[i] = z
					activations = append(activations, activation)
				}

				/*
					We know nablaW = (C_w11, C_w12, ..., C_w1n, ..., C_wl1, ..., C_wlm) where every C_w is the derivative of the
					cost function with respect to wji where j is the layer number and i is the index of the neuron in the given layer.

					Every C_wji = - (mu / m) * SUM {(sigmoid(z) - Expected) * sigmoid_prime(z) * wij} FROM 1 to
					Respectively C_bji = - (mu / m) * SUM {(Expected - sigmoid(z)) * sigmoid_prime(z) * 1} FROM 1 to m

					So the first thing we are going to compute is

					cost = -(Expected - sigmoid(z))
					sp = sigmoid_prime(z)

					And with that we are going to build our nablaW and nablaB to compute the deltas
				*/

				// From the last layer to the first one
				for i := n.layersCount - 1; i > 0; i-- {
					for j := range n.layers[i] {
						z := zs[i][j]
						wij := n.layers[i][j].Weights()[j]

						// We already computed sigmoid(z) and we saved it in the activations slice
						// So now we compute the deltas
						ps := sigmoidPrime(z)
						delta := -(expected[i] - activations[i][j]) * ps
						deltaNablaB[i][j] = delta
						deltaNablaW[i][j] = delta * wij
					}
				}

				// We change the previous values we stored in the nabla arrays
				for i := 0; i < len(deltaNablaB); i++ {
					nablaB[i] = vector.Add(nablaB[i], deltaNablaB[i])
					nablaW[i] = vector.Add(nablaW[i], deltaNablaW[i])
				}
			}

			// We update the weights
			for i := 0; i < n.layersCount; i++ {
				for j := 0; j < len(nablaB); j++ {
					bij := n.layers[i][j].Bias()
					n.layers[i][j].SetBias(bij - (learningRate/M)*nablaB[i][j])

					for _, wij := range n.layers[i][j].Weights() {
						n.layers[i][j].SetWeight(j, wij-(learningRate/M)*nablaW[i][j])
					}
				}
			}
		}

		correct := 0
		for _, i := range testData {
			res := n.FeedForward(i.Input)
			if vector.MaxKey(res) == vector.MaxKey(i.Expected) {
				correct++
			}
		}

		fmt.Printf("After %v epochs: %v / %v answers are correct", k+1, correct, len(testData))
	}
}
