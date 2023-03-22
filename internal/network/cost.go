package network

import "github.com/gonzispina/gocr/kit/vector"

type CostFunction interface {
	Delta(activation, expected vector.Vector) vector.Vector
}

/*
	Quadratic cost function
*/

func NewQuadraticCostFunc() QuadraticCost {
	return QuadraticCost{}
}

type QuadraticCost struct{}

func (c *QuadraticCost) Delta(activation, expected vector.Vector) vector.Vector {
	return vector.Substract(activation, expected)
}

/*
	Cross entropy cost function
*/

func NewCrossEntropyCostFunc() CrossEntropyCost {
	return CrossEntropyCost{sigmoid: Sigmoid{}}
}

type CrossEntropyCost struct {
	sigmoid Sigmoid
}

func (c *CrossEntropyCost) Delta(activation, expected vector.Vector) vector.Vector {
	return (activation - expected) / c.sigmoid
}
