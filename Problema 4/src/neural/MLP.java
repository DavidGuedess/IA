package neural;

import math.Matrix;
import neural.activation.IDifferentiableFunction;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MLP {

    private Matrix[] w;  //weights for each layer
    private Matrix[] b;  //biases for each layer
    private Matrix[] yp; //outputs for each layer
    private IDifferentiableFunction[] act; //activation functions for each layer
    private int numLayers;

    // For storing purposes
    private Matrix[] bestW;
    private Matrix[] bestB;

    /* Create a Multi-Layer Perceptron with the given layer sizes.
     * layerSizes is an array where each element represents the number of neurons in that layer.
     * For example, new MLP(new int[]{3, 5, 2}) creates a MLP with 3 input neurons,
     * 5 hidden neurons, and 2 output neurons.
     *
     * PRE: layerSizes.length >= 2
     * PRE: act.length == layerSizes.length - 1
     */
    public MLP(int[] layerSizes, IDifferentiableFunction[] act, int seed) {
        if (seed < 0)
            seed = (int) System.currentTimeMillis();

        numLayers = layerSizes.length;

        this.act = act;

        yp = new Matrix[numLayers];

        w = new Matrix[numLayers-1];
        b = new Matrix[numLayers-1];

        Random rnd = new Random(seed);

        for (int i=0; i < numLayers-1; i++) {
            double scale = Math.sqrt(2.0 / layerSizes[i]);
            w[i] = Matrix.Rand(layerSizes[i], layerSizes[i + 1], rnd.nextInt());
            w[i] = w[i].mult(scale).add(-scale/2);

            b[i] = new Matrix(1, layerSizes[i + 1]);
        }
    }

    public Matrix predict(Matrix X) {
        yp[0] = X;
        for (int l=0; l < numLayers-1; l++)
            yp[l + 1] = yp[l].dot(w[l]).addRowVector(b[l]).apply(act[l].fnc());
        return yp[numLayers-1];
    }

    public Matrix backPropagation(Matrix y, double lr) {
        Matrix Eout = null;
        Matrix e = null;
        Matrix delta = null;

        for (int l = numLayers-2; l >= 0; l--) {
            if (l == numLayers-2) {
                Eout = e = y.sub(yp[l+1]);
            }
            else {
                e = delta.dot(w[l+1].transpose());
            }
            Matrix dy = yp[l+1].apply(act[l].derivative());
            delta = e.mult(dy);

            w[l] = w[l].add(yp[l].transpose().dot(delta).mult(lr));
            b[l] = b[l].addRowVector(delta.sumColumns().mult(lr));
        }
        return Eout;
    }


    public double[] train(Matrix X, Matrix y, double learningRate, int epochs) {
        int nSamples = X.rows();
        double[] mse = new double[epochs];

        for (int epoch=0; epoch < epochs; epoch++) {
            Matrix ypo = predict(X);

            Matrix e = backPropagation(y, learningRate);

            mse[epoch] = e.dot(e.transpose()).get(0, 0) / nSamples;
        }
        return mse;
    }

    public double[] train(Matrix trainX, Matrix trainY, Matrix valX, Matrix valY, double learningRate, int epochs, int patience) {
        int numSamples = trainX.rows();
        int valSamples = valX.rows();
        double bestValMSE = Double.MAX_VALUE;
        int epochsWithoutImprovement = 0;

        double[] valMSE = new double[epochs];

        for (int epoch = 0; epoch < epochs; epoch++) {
            predict(trainX);
            Matrix e = backPropagation(trainY, learningRate);
            double trainMSE = e.transpose().dot(e).get(0, 0) / numSamples;

            Matrix valPred = predict(valX);
            Matrix valErr = valPred.sub(valY);
            valMSE[epoch] = valErr.transpose().dot(valErr).get(0, 0) / valSamples;

            if (valMSE[epoch] < bestValMSE) {
                bestValMSE = valMSE[epoch];
                epochsWithoutImprovement = 0;
                saveBestWeights();
            } else {
                epochsWithoutImprovement++;
            }

            if (epochsWithoutImprovement >= patience) {
                System.err.printf("Early stopping at epoch %d (no improvement for %d epochs)%n", epoch, patience);
                restoreBestWeights();
                break;
            }
            if (epoch % 100 == 0) {
                System.err.printf("Epoch %d: trainMSE=%.15f, valMSE=%.15f)%n", epoch, trainMSE, valMSE[epoch]);
            }
        }
        return valMSE;
    }

    private void saveBestWeights() {
        if (bestW == null) {
            bestW = new Matrix[numLayers - 1];
            bestB = new Matrix[numLayers - 1];
        }
        for (int i = 0; i < numLayers - 1; i++) {
            bestW[i] = w[i].copy();
            bestB[i] = b[i].copy();
        }
    }

    // Restore best weights from internal copy
    private void restoreBestWeights() {
        if (bestW != null) {
            for (int i = 0; i < numLayers - 1; i++) {
                w[i] = bestW[i].copy();
                b[i] = bestB[i].copy();
            }
        }
    }

    public void saveWeights(String filename) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeInt(numLayers - 1);

            for (int i = 0; i < numLayers - 1; i++) {
                oos.writeInt(w[i].rows());
                oos.writeInt(w[i].cols());

                for (int r = 0; r < w[i].rows(); r++) {
                    for (int c = 0; c < w[i].cols(); c++) {
                        oos.writeDouble(w[i].get(r, c));
                    }
                }

                oos.writeInt(b[i].rows());
                oos.writeInt(b[i].cols());

                for (int r = 0; r < b[i].rows(); r++) {
                    for (int c = 0; c < b[i].cols(); c++) {
                        oos.writeDouble(b[i].get(r, c));
                    }
                }
            }
        }
        System.err.println("Weights saved to " + filename);
    }

    public void loadWeights(String filename) throws IOException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            int savedLayers = ois.readInt();

            if (savedLayers != numLayers - 1) {
                throw new IOException("Model architecture mismatch: expected " +
                        (numLayers - 1) + " layer connections, but file contains " + savedLayers);
            }

            for (int i = 0; i < numLayers - 1; i++) {
                int wRows = ois.readInt();
                int wCols = ois.readInt();

                if (wRows != w[i].rows() || wCols != w[i].cols()) {
                    throw new IOException("Weight matrix size mismatch at layer " + i);
                }

                double[][] wData = new double[wRows][wCols];
                for (int r = 0; r < wRows; r++) {
                    for (int c = 0; c < wCols; c++) {
                        wData[r][c] = ois.readDouble();
                    }
                }
                w[i] = new Matrix(wData);

                int bRows = ois.readInt();
                int bCols = ois.readInt();

                if (bRows != b[i].rows() || bCols != b[i].cols()) {
                    throw new IOException("Bias vector size mismatch at layer " + i);
                }

                double[][] bData = new double[bRows][bCols];
                for (int r = 0; r < bRows; r++) {
                    for (int c = 0; c < bCols; c++) {
                        bData[r][c] = ois.readDouble();
                    }
                }
                b[i] = new Matrix(bData);
            }
        }
    }
}
