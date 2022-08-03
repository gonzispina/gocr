package mnist

import (
	"github.com/gonzispina/gocr/internal/network"
	"github.com/gonzispina/gocr/kit/vector"
	"github.com/petar/GoMNIST"
)

var (
	trainingData *GoMNIST.Set
	testData     *GoMNIST.Set
)

func loadData(dataPath string) error {
	if trainingData != nil && testData != nil {
		return nil
	}

	train, test, err := GoMNIST.Load(dataPath)
	if err != nil {
		return err
	}

	trainingData = train
	testData = test

	return nil
}

func ReadTrainingData(dataPath string) ([]*network.Input, error) {
	err := loadData(dataPath)
	if err != nil {
		return nil, err
	}

	return marshallData(trainingData), nil
}

func ReadTestData(dataPath string) ([]*network.Input, error) {
	return marshallData(testData), nil
}

func marshallData(dataset *GoMNIST.Set) []*network.Input {
	result := make([]*network.Input, 0, dataset.Count())
	for i := 0; i < dataset.Count(); i++ {
		image, label := dataset.Get(i)

		vectorizedImage := make(vector.Vector, 0, 784)
		bounds := image.Bounds()
		for x := 0; x < bounds.Max.X; x++ {
			for y := 0; y < bounds.Max.Y; y++ {
				r, _, _, _ := image.At(x, y).RGBA()
				vectorizedImage = append(vectorizedImage, float64(r)/100000)
			}
		}
		t := &network.Input{
			Input:    vectorizedImage,
			Expected: vectorizeOutput(uint8(label)),
		}
		result = append(result, t)
	}

	return result
}

func vectorizeOutput(n uint8) vector.Vector {
	res := make([]float64, 10)
	res[n] = 1.0
	return res
}
