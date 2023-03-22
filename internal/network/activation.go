package network

import "math"

type ActivationFunc interface {
	Compute(x float64) float64
	Derivative(x float64) float64
}

type Sigmoid struct{}

// Compute the value s(x) = 1 / (1 + e^(-x))
func (s *Sigmoid) Compute(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative computes the derivative of the sigmoid function: s'(x) = s(x) * (1 - s(x))
func (s *Sigmoid) Derivative(x float64) float64 {
	x = s.Compute(x)
	return x * (1 - x)
}
