package vector

import (
	"github.com/cpmech/gosl/la"
	"math"
	"math/rand"
	"time"
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

// Hadamard product
func Hadamard(u Vector, v Vector) Vector {
	res := la.NewVector(len(u))
	for i := 0; i < len(u); i++ {
		res[i] = u[i] * v[i]
	}
	return res
}

// Scale by a value
func Scale(u Vector, scalar float64) Vector {
	res := make(Vector, len(u))
	for i := range u {
		res[i] = u[i] * scalar
	}
	return res
}

func Apply(v Vector, f func(f float64) float64) Vector {
	res := make(Vector, len(v))
	for i, u := range v {
		res[i] = f(u)
	}
	return res
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

func CreateZero(size int) Vector {
	v := la.NewVector(size)
	v.Fill(0)
	return v
}

func CreateManyZero(sizes []int) []Vector {
	res := make([]Vector, len(sizes))
	for i, s := range sizes {
		res[i] = CreateZero(s)
	}
	return res
}

// CreateNormalRandom vector
func CreateNormalRandom(size, normParameter int) Vector {
	rand.Seed(time.Now().UnixNano())
	vec := la.NewVector(size)
	for i := 0; i < size; i++ {
		vec[i] = rand.NormFloat64() / math.Sqrt(float64(normParameter))
	}
	return vec
}

// CreateManyNormalRandom vectors
func CreateManyNormalRandom(sizes, normParameters []int) []Vector {
	vecs := make([]Vector, len(sizes))
	for i := 0; i < len(sizes); i++ {
		vecs[i] = CreateNormalRandom(sizes[i], normParameters[i])
	}
	return vecs
}

// CreateManyFixedSizeNormalRandom vectors
func CreateManyFixedSizeNormalRandom(amount, size, normParameter int) []Vector {
	vecs := make([]Vector, amount)
	for i := 0; i < amount; i++ {
		vecs[i] = CreateNormalRandom(size, normParameter)
	}
	return vecs
}
