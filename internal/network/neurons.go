package network

import (
	"github.com/cpmech/gosl/la"
	"github.com/gonzispina/gocr/kit/vector"
)

// Layer representation
type Layer []Neuron

// Activate the layer
func (l Layer) Activate(input vector.Vector) la.Vector {
	res := make(vector.Vector, len(l))
	for i, n := range l {
		res[i] = n.Activate(input)
	}
	return res
}

// Neuron contract for different types of neurons
type Neuron interface {
	Activate(input vector.Vector) float64
	Weights() vector.Vector
	Bias() float64
	SetWeight(index int, value float64)
	SetBias(value float64)
}

type baseNeuron struct {
	bias    float64
	weights vector.Vector
}

// Weights getter
func (n *baseNeuron) Weights() vector.Vector {
	return n.weights
}

// Bias getter
func (n *baseNeuron) Bias() float64 {
	return n.bias
}

// SetWeight the neuron weights and biases
func (n *baseNeuron) SetWeight(index int, value float64) {
	n.weights[index] = value
}

// SetBias the neuron weights and biases
func (n *baseNeuron) SetBias(value float64) {
	n.bias = value
}

func NewSigmoidNeuron(b float64, w vector.Vector) Neuron {
	return &SigmoidNeuron{&baseNeuron{
		bias:    b,
		weights: w,
	}}
}

// SigmoidNeuron representation
type SigmoidNeuron struct {
	*baseNeuron
}

// Activate calls the sigmoid function.
func (n *SigmoidNeuron) Activate(input vector.Vector) float64 {
	return sigmoid(vector.Dot(n.weights, input) + n.bias)
}
