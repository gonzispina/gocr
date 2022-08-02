package network

import (
	"github.com/gonzispina/gocr/kit/vector"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNetwork_StochasticGradientDescent(t *testing.T) {
	sizes := []int{2, 3, 1}

	trainningTest := []*Input{
		{
			Input:    vector.Vector{0.9, 0.9},
			Expected: vector.Vector{0},
		},
	}

	net := NewRandom(sizes)

	value := 1.0
	for i := 0; i < 100; i++ {
		net.StochasticGradientDescent(trainningTest, 1, 5.0)
		res, _ := net.FeedForward(trainningTest[0].Input)

		assert.True(t, res[0] < value)
		value = res[0]
	}
}
