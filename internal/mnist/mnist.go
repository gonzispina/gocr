package mnist

import (
	"fmt"
	"github.com/gonzispina/gocr/internal/network"
	"github.com/gonzispina/gocr/kit/vector"
	"os"
	"path/filepath"
)

const (
	TrainImagesFile = "train-images.idx3-ubyte"
	TrainLabelsFile = "train-labels.idx1-ubyte"
	TestImagesFile  = "t10k-images.idx3-ubyte"
	TestLabelsFile  = "t10k-labels.idx1-ubyte"

	labelsFileMagic = 0x00000801
	imagesFileMagic = 0x00000803

	msgInvalidFormat = "Invalid format: %s"
	msgSizeUnmatch   = "Data size does not match: %s %s"
)

func fileError(f *os.File) error {
	return fmt.Errorf(msgInvalidFormat, f.Name())
}

// internal: Read 4 bytes and convert to big endian integer
func readInt32(f *os.File) (int, error) {
	buf := make([]byte, 4)
	n, e := f.Read(buf)
	switch {
	case e != nil:
		return 0, e
	case n != 4:
		return 0, fileError(f)
	}
	v := 0
	for _, x := range buf {
		v = v*256 + int(x)
	}
	return v, nil
}

// internal: raw image data
type imageData struct {
	N    int
	W    int
	H    int
	Data []uint8
}

func readImagesFile(path string) (*imageData, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	magic, e := readInt32(f)
	if e != nil || magic != imagesFileMagic {
		return nil, fileError(f)
	}
	n, e := readInt32(f)
	if e != nil {
		return nil, fileError(f)
	}
	w, e := readInt32(f)
	if e != nil {
		return nil, fileError(f)
	}
	h, e := readInt32(f)
	if e != nil {
		return nil, fileError(f)
	}
	sz := n * w * h
	data := &imageData{n, w, h, make([]uint8, sz)}
	len, e := f.Read(data.Data)
	if e != nil || len != sz {
		return nil, fileError(f)
	}
	return data, nil
}

// internal: raw label data
type labelData struct {
	N    int
	Data []uint8
}

func readLabelsFile(path string) (*labelData, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	magic, e := readInt32(f)
	if e != nil || magic != labelsFileMagic {
		return nil, fileError(f)
	}
	n, e := readInt32(f)
	if e != nil {
		return nil, fileError(f)
	}
	data := &labelData{n, make([]uint8, n)}
	len, e := f.Read(data.Data)
	if e != nil || len != n {
		return nil, fileError(f)
	}
	return data, nil
}

// DigitImage Single digit+image datum
type DigitImage struct {
	Digit int
	Image [][]uint8
}

// DataSet
type DataSet struct {
	N    int
	W    int
	H    int
	Data []DigitImage
}

func dataset2NetInput(d *DataSet) []*network.Input {
	res := make([]*network.Input, d.N)
	for i, testCase := range d.Data {
		expected := vector.CreateZero(10)
		expected[testCase.Digit] = 1

		input := vector.CreateZero(d.W * d.H)
		for j := 0; j < d.W; j++ {
			for k := 0; k < d.H; k++ {
				input[j*d.W+k] = float64(testCase.Image[j][k]) / 255
			}
		}

		res[i] = &network.Input{
			Expected: expected,
			Input:    input,
		}
	}
	return res
}

// ReadDataSet ...
func ReadDataSet(imagesPath, labelsPath string) ([]*network.Input, error) {
	images, e := readImagesFile(imagesPath)
	if e != nil {
		return nil, e
	}
	labels, e := readLabelsFile(labelsPath)
	if e != nil {
		return nil, e
	}
	if images.N != labels.N {
		return nil, fmt.Errorf(msgSizeUnmatch, labelsPath, imagesPath)
	}
	dataSet := &DataSet{N: images.N, W: images.W, H: images.H}
	dataSet.Data = make([]DigitImage, dataSet.N)
	rows := splitToRows(images.Data, images.N, images.H)
	for i := 0; i < dataSet.N; i++ {
		data := &dataSet.Data[i]
		data.Digit = int(labels.Data[i])
		data.Image = rows[0:dataSet.H]
		rows = rows[dataSet.H:]
	}
	return dataset2NetInput(dataSet), nil
}

func ReadTrainSet(dir string) ([]*network.Input, error) {
	imagesPath := filepath.Join(dir, TrainImagesFile)
	labelsPath := filepath.Join(dir, TrainLabelsFile)
	return ReadDataSet(imagesPath, labelsPath)
}

func ReadTestSet(dir string) ([]*network.Input, error) {
	imagesPath := filepath.Join(dir, TestImagesFile)
	labelsPath := filepath.Join(dir, TestLabelsFile)
	return ReadDataSet(imagesPath, labelsPath)
}

func splitToRows(data []uint8, N, H int) [][]uint8 {
	nR := N * H
	rows := make([][]uint8, nR)
	for i := 0; i < nR; i++ {
		rows[i] = data[0:H]
		data = data[H:]
	}
	return rows
}

// (debugging utility)
func PrintImage(image [][]uint8) {
	for _, row := range image {
		for _, pix := range row {
			if pix == 0 {
				fmt.Print(" ")
			} else {
				fmt.Printf("%X", pix/16)
			}
		}
		fmt.Println()
	}
}
