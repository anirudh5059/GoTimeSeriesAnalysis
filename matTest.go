package main

import (
	"fmt"
	"encoding/csv"
	"gonum.org/v1/gonum/mat"
    "log"
    "os"
	"strconv"
	"gonum.org/v1/gonum/stat"
	"github.com/sajari/regression"
)

func readCsvFile(filePath string) [][]string {
    f, err := os.Open( filePath )
    if err != nil {
        log.Fatal( "Unable to read input file " + filePath, err )
    }
    defer f.Close()

    csvReader := csv.NewReader( f )
    records, err := csvReader.ReadAll()
    if err != nil {
        log.Fatal( "Unable to parse file as CSV for " + filePath, err )
    }

    return records
}

func acf_loop(x []float64) {
	for i := 1; i < 11; i++ {
		// Shift the series.
		adjusted := x[i:len(x)]
		lag := x[0 : len(x)-i]
		// Calculate the autocorrelation.
		ac := stat.Correlation(adjusted, lag, nil)
		fmt.Printf("Lag %d period: %0.2f\n", i, ac)
	}
}

func autoregressive(x []float64, lag int) ([]float64, float64) {
	// Create a regresssion.Regression value needed to train
	// a model using github.com/sajari/regression.
	var r regression.Regression
	r.SetObserved("x")
	// Define the current lag and all of the intermediate lags.
	for i := 0; i < lag; i++ {
		r.SetVar(i, "x"+strconv.Itoa(i))
	}
	// Shift the series.
	xAdj := x[lag:len(x)]
	// Loop over the series creating the data set
	// for the regression.
	for i, xVal := range xAdj {
		// Loop over the intermediate lags to build up
		// our independent variables.
		laggedVariables := make([]float64, lag)
		for idx := 1; idx <= lag; idx++ {
			// Get the lagged series variables.
			laggedVariables[idx-1] = x[lag+i-idx]
		}
		// Add these points to the regression value.
		r.Train(regression.DataPoint(xVal, laggedVariables))
	}
	// Fit the regression.
	r.Run()
	// coeff hold the coefficients for our lags.
	var coeff []float64
	for i := 1; i <= lag; i++ {
		coeff = append(coeff, r.Coeff(i))
	}
	return coeff, r.Coeff(0)
}

func main() {
	records := readCsvFile( "tseries.csv" )
	var result = []float64{}
	for _, arr := range records {
		for _, item := range arr {
			//fmt.Println( item )
			if s, err := strconv.ParseFloat( item, 64 ); err == nil {
				result = append(result, s )
			}
		}
	}
	m := mat.NewDense( len( records ), 2, result )
	//fa := mat.Formatted( m, mat.Prefix("    "), mat.Squeeze() )
	//fmt.Println( result )
	//fmt.Println( mat.Formatted( m ) )
	series := mat.Col( nil, 1, m)
	coeffs, intercept := autoregressive( series, 2 )
	fmt.Printf("\nlog(x(t)) - log(x(t-1)) = %0.6f + lag1*%0.6f + lag2*%0.6f\n\n", intercept, coeffs[0], coeffs[1])
	//acf_loop( series )
	//fmt.Println( series )
	//fmt.Println( records )
}
