package neural.activation;

import java.util.function.Function;

public class LeakyReLU implements IDifferentiableFunction {
    @Override
    public Function<Double, Double> fnc() {
        return z -> z > 0.0 ? z : 0.01 * z;
    }

    @Override
    public Function<Double, Double> derivative() {
        return y -> y > 0.0 ? 1.0 : 0.01;
    }
}