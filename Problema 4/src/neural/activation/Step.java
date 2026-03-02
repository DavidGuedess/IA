package neural.activation;

import java.util.function.Function;

public class Step implements IDifferentiableFunction {

    static private double threshold = 0.5;

    @Override
    public Function<Double, Double> fnc() {
        return (x) -> x >= threshold ? 3.0 : 2.0;
    }

    @Override
    public Function<Double, Double> derivative() {
        throw new UnsupportedOperationException("Step function is not differentiable.");
    }

}
