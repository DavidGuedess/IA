import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.MLP;
import neural.activation.LeakyReLU;
import neural.activation.Sigmoid;
import neural.activation.Step;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class P4 {
    private static final String WEIGHTS_FILE = "src/models/teste.dat";
    private static final String MINMAX_FILE = "src/models/minmax.dat";

    private static final int seed = 42;
    private static final int[] topology = {400, 256, 128, 64, 1};

    public static void main(String[] args) throws IOException {
        double[] minmax = loadMinMax(MINMAX_FILE);
        double min = minmax[0];
        double max = minmax[1];

        MLP mlp = new MLP(topology,
                new IDifferentiableFunction[]{
                        new LeakyReLU(),
                        new LeakyReLU(),
                        new LeakyReLU(),
                        new Sigmoid()
                },
                seed);
        mlp.loadWeights(WEIGHTS_FILE);

        Scanner scanner = new Scanner(System.in);
        ArrayList<Integer> outputs = new ArrayList<>();
        Step step = new Step();

        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String[] values = line.split(",");

            double[] input = new double[400];
            for (int i = 0; i < 400; i++) {
                double val = Double.parseDouble(values[i].trim());
                input[i] = (val - min) / (max - min);
                input[i] = Math.min(1.0, Math.max(0.0, input[i]));
            }

            Matrix in = new Matrix(new double[][]{input});
            Matrix pred = mlp.predict(in);

            pred = pred.apply(step.fnc());
            int output = (int)pred.get(0,0);
            outputs.add(output);
        }

        for (Integer output : outputs) {
            System.out.println(output);
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