package vector

import (
	"github.com/cpmech/gosl/la"
	"math/rand"
)

// Vector shadowing, this way it is easier to change the lib in the future if necessary
type Vector = la.Vector

// MaxKey returns the key corresponding to the maximum value
func MaxKey(v Vector) int {
	max := v[0]
	key := 0
	for i := 1; i < len(v); i++ {
		if v[i] > max {
			max = v[i]
			key = i
		}
	}
	return key
}

// Dot product
func Dot(u Vector, v Vector) float64 {
	return la.VecDot(u, v)
}

// Add returns the difference between two vectors
func Add(u Vector, v Vector) Vector {
	res := la.NewVector(len(u))
	la.VecAdd(res, 1, u, 1, v)
	return res
}

// Substract returns the difference between two vectors
func Substract(u Vector, v Vector) Vector {
	res := la.NewVector(len(u))
	la.VecAdd(res, 1, u, -1, v)
	return res
}

func CreateZeroVector(size int) Vector {
	v := la.NewVector(size)
	v.Fill(0)
	return v
}

func CreateManyZeroVector(sizes []int) []Vector {
	res := make([]Vector, len(sizes))
	for i, s := range sizes {
		res[i] = CreateZeroVector(s)
	}
	return res
}

// CreateNormalRandom vector
func CreateNormalRandom(size int) Vector {
	vec := la.NewVector(size)
	for i := 0; i < size; i++ {
		vec[i] = rand.NormFloat64()
	}
	return vec
}

// CreateManyNormalRandom vectors
func CreateManyNormalRandom(sizes []int) []Vector {
	vecs := make([]Vector, len(sizes))
	for i := 0; i < len(sizes)-1; i++ {
		vecs[i] = CreateNormalRandom(sizes[i])
	}
	return vecs
}
