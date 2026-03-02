package apps;

import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.MLP;
import neural.activation.Sigmoid;
import neural.activation.Step;

import java.util.Scanner;

public class MLPNXOR {
    public static void main(String[] args) {
        double lr    = 0.01; //define a learning rate in range [0, 1]
        int   epochs = 400000;//define the number of epochs in the order of thousands
        int[] topology = {2, 2, 1};

        //Dataset
        Matrix trX = new Matrix(
                new double[][]{
                        {0, 0},
                        {0, 1},
                        {1, 0},
                        {1, 1}});

        Matrix trY = new Matrix(
                new double[][]{
                        {1},
                        {0},
                        {0},
                        {1}});


        //Get input and create evaluation Matrix
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        double[][] input = new double[n][2];

        for (int i = 0; i < n; i++) {
            input[i][0] = sc.nextDouble();
            input[i][1] = sc.nextDouble();
        }
        Matrix evX = new Matrix(input);

        //Train the MLP
        MLP mlp = new MLP(topology,
                new IDifferentiableFunction[]{
                        new Sigmoid(),
                        new Sigmoid(),},
                15);

        mlp.train(trX, trY, lr, epochs);

        /*
        //MLP com os pesos e bias obitdos do treino. (lr = 0.01, epochs = 400000, seed = 15)
        Matrix[] trainedW = {
                new Matrix(new double[][] {
                        {6.398, 4.498},
                        {6.400,  4.499}
                }),
                new Matrix(new double[][] {
                        {-9.347},
                        {10.045}
                })
        };

        Matrix[] trainedB = {
                new Matrix(new double[][] {{-2.831, -6.904}}),
                new Matrix(new double[][] {{4.312}})
        };

        MLP trainedMLP = new MLP(trainedW, trainedB, new IDifferentiableFunction[]{
                new Sigmoid(),
                new Sigmoid(),}
        );

         */


        //Predict and output results
        Matrix pred = mlp.predict(evX);

        //convert probabilities to integer classes: 0 or 1
        pred = pred.apply(new Step().fnc());

        //print output
        //insert code here to print the pred Matrix as integers
        //…
        for (int i = 0; i < pred.rows(); i++) {
            for (int j = 0; j < pred.cols(); j++) {
                System.out.print((int)pred.get(i, j));
            }
            System.out.println();
        }

        sc.close();
    }
}
