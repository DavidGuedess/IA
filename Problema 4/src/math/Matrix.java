package math;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

public class Matrix {

    private double[][] data;
    private int rows, cols;

    public Matrix(int rows, int cols) {
        data = new double[rows][cols];
        this.rows = rows;
        this.cols = cols;
    }


    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(data[i], 0, this.data[i], 0, cols);

    }


    static public Matrix Rand(int rows, int cols, int seed) {
        Matrix out = new Matrix(rows, cols);

        if (seed < 0)
            seed = (int) System.currentTimeMillis();

        Random rand = new Random(seed);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out.data[i][j] = rand.nextDouble();

        return out;
    }


    //accessors
    public double get(int row, int col) {
        return data[row][col];
    }
    public int rows() { return rows; }
    public int cols() { return cols; }

    //getters
    public void set(int row, int col, double value) {
        data[row][col] = value;
    }


    //==============================================================
    //  Element operations
    //==============================================================

    //Apply Function<Double, Double> to all elements of the matrix
    //store the result in matrix result
    private Matrix traverse(Function<Double, Double> fnc) {
        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = fnc.apply(data[i][j]);
            }
        }
        return result;
    }

    public Matrix apply(Function<Double, Double> fnc) {
        return traverse(fnc);
    }

    //multiply matrix by scalar
    public Matrix mult(double scalar) {

        // insert code here to multiply all elements by scalar
        return traverse((x) -> x * scalar);
    }

    //add scalar to matrix
    public Matrix add(double scalar) {

        // insert code here to add to all elements the scalar
        return traverse((x) -> x + scalar);
    }

    //sub matrix from scalar:   scalar - M
    public Matrix subFromScalar(double scalar) {

        // insert code here to subtract from the scalar all the elements
        return traverse((x) -> scalar - x);
    }


    //==============================================================
    //  Element-wise operations between two matrices
    //==============================================================

    //Element wise operation
    private Matrix elementWise(Matrix other, BiFunction<Double, Double, Double> fnc) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Incompatible matrix sizes for element wise.");
        }

        Matrix result = new Matrix(rows, cols);

        //add element by element
        //store the result in matrix result
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                result.data[i][j] = fnc.apply(this.data[i][j], other.data[i][j]);
        }
        return result;
    }

    //add two matrices
    public Matrix add(Matrix other) {
        return this.elementWise(other, (a, b) -> a + b);
    }

    //multiply two matrices (element wise)
    public Matrix mult(Matrix other) {

        // insert code here to multiply element-wise
        return this.elementWise(other, (a, b) -> a * b);
    }

    //sub two matrices
    public Matrix sub(Matrix other) {

        // insert code here to subtract (this-other) element-wise
        return this.elementWise(other, (a, b) -> a - b);
    }


    //==============================================================
    //  Other math operations
    //==============================================================

    //sum all elements of the matrix
    public double sum() {
        double total = 0.0;

        // insert code here to return the sum of all elements
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                total += data[i][j];
            }
        }

        return total;
    }


    //Sum by columns
    public Matrix sumColumns() {
        Matrix result = new Matrix(1, this.cols);

        for (int j = 0; j < this.cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < this.rows; i++)
                sum += this.data[i][j];
            result.data[0][j] = sum;
        }
        return result;
    }


    //multiply two matrices (dot product)
    public Matrix dot(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Incompatible matrix sizes for multiplication.");
        }

        Matrix result = new Matrix(this.rows, other.cols);

        //multiply 2 matrices
        //store the result in matrix result
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < this.cols; k++) {
                    result.data[i][j] += this.data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }


    //Add row vector to each row of the matrix
    public Matrix addRowVector(Matrix rowVector) {
        if (rowVector.rows() != 1 || rowVector.cols() != this.cols) {
            throw new IllegalArgumentException("Incompatible sizes for adding row vector.");
        }

        Matrix result = new Matrix(this.rows, this.cols);

        //add row vector to each row
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] + rowVector.data[0][j];
            }
        }
        return result;
    }


    //==============================================================
    //  Column and row operations
    //==============================================================

    //transpose matrix
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);

        //transpose the matrix
        //store the result in matrix result
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }


    //==============================================================
    //  Convert operations
    //==============================================================

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (double[] row : data) {
            for (double val : row) {
                sb.append(String.format("%.3f ", val));
            }
            sb.append("\n");
        }
        return sb.toString();
    }


    //==============================================================
    //  Compare operations
    //==============================================================

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Matrix matrix)) return false;
        return rows == matrix.rows && cols == matrix.cols && Objects.deepEquals(data, matrix.data);
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.deepHashCode(data), rows, cols);
    }

    public Matrix copy() {
        Matrix copy = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                copy.data[i][j] = this.data[i][j];
            }
        }
        return copy;
    }
}
