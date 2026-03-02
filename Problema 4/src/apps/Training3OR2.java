package apps;

import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.MLP;
import neural.activation.LeakyReLU;
import neural.activation.Sigmoid;
import neural.activation.Step;

import java.io.*;
import java.util.*;

public class Training3OR2 {
    private static final String WEIGHTS_FILE = "src/models/teste.dat";
    private static final String MINMAX_FILE = "src/models/minmax.dat";

    private static final double lr = 0.0001;
    private static final int epochs = 10000;
    private static final int patience = 100;
    private static final int seed = 42;
    private static final int[] topology = {400, 256, 128, 64, 1};

    private static final double tr = 0.8; //train ratio

    public static void main(String[] args) throws IOException {
        Matrix trainX = new Matrix(loadData("training/train_dataset.csv"));
        Matrix trainY = loadLabelsAsMatrix("training/train_labels.csv");
        Matrix valX = new Matrix(loadData("training/val_dataset.csv"));
        Matrix valY = loadLabelsAsMatrix("training/val_labels.csv");

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (int i = 0; i < trainX.rows(); i++) {
            for (int j = 0; j < trainX.cols(); j++) {
                double v = trainX.get(i, j);
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        saveMinMax(min, max);
        normalize(trainX, min, max);
        normalize(valX, min, max);

        MLP mlp = new MLP(topology,
                new IDifferentiableFunction[]{
                        new LeakyReLU(),
                        new LeakyReLU(),
                        new LeakyReLU(),
                        new Sigmoid()
                },
                seed);
        //mlp.loadWeights(WEIGHTS_FILE);

        mlp.train(trainX, trainY, valX, valY, lr, epochs, patience);
        mlp.saveWeights(WEIGHTS_FILE);

        Matrix valPred = mlp.predict(valX);
        int correct = 0;
        for (int i = 0; i < valX.rows(); i++) {
            double pred = valPred.get(i, 0);
            int predClass = pred < 0.5 ? 0 : 1;
            int trueClass = (int) valY.get(i, 0);
            if (predClass == trueClass) correct++;
        }
        double accuracy = 100.0 * correct / valX.rows();
        System.err.printf("Validation Accuracy: %.2f%%%n", accuracy);
    }

    private static List<Matrix> splitData(List<double[]> data, List<Integer> labels) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) indices.add(i);
        Collections.shuffle(indices, new Random(seed));

        int trainSize = (int)(data.size() * tr);

        double[][] trainX = new double[trainSize][400];
        double[][] trainY = new double[trainSize][1];

        double[][] valX = new double[data.size() - trainSize][400];
        double[][] valY = new double[data.size() - trainSize][1];

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices.get(i);
            if (i < trainSize) {
                trainX[i] = data.get(idx);
                trainY[i][0] = labels.get(idx) == 2 ? 0.0 : 1.0;
            } else {
                valX[i - trainSize] = data.get(idx);
                valY[i - trainSize][0] = labels.get(idx)  == 2 ? 0.0 : 1.0;
            }
        }

        List<Matrix> result = new ArrayList<>();
        result.add(new Matrix(trainX));
        result.add(new Matrix(trainY));
        result.add(new Matrix(valX));
        result.add(new Matrix(valY));
        return result;
    }

    private static void normalize(Matrix X, double min, double max) {
        double range = max - min;
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                X.set(i, j, (X.get(i, j) - min) / range);
            }
        }
    }

    private static double[][] loadData(String filename) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] row = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    row[i] = Double.parseDouble(values[i]);
                }
                rows.add(row);
            }
        }
        return rows.toArray(new double[0][]);
    }

    private static Matrix loadLabelsAsMatrix(String filename) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                int label = Integer.parseInt(line.trim());
                rows.add(new double[]{label == 2 ? 0.0 : 1.0});
            }
        }
        return new Matrix(rows.toArray(new double[0][]));
    }

    private static List<Integer> loadLabels(String filename) {
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                int label = Integer.parseInt(line.trim());
                labels.add(label);
            }
        } catch (IOException e) {
            System.err.println("Error loading labels from " + filename + ": " + e.getMessage());
            System.exit(1);
        }
        return labels;
    }

    private static void saveMinMax(double min, double max) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(MINMAX_FILE))) {
            bw.write(min + "\n");
            bw.write(max + "\n");
            System.err.println("Min/max values saved to " + MINMAX_FILE);
        }
    }

    private static double[] loadMinMax(String filename) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            double min = Double.parseDouble(br.readLine());
            double max = Double.parseDouble(br.readLine());
            return new double[]{min, max};
        }
    }
}