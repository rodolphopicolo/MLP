package mlp.activation;

/**
 *
 * @author rodolpho
 */
public class Sigmoid extends ActivationFunction {
    @Override
    public double calculate(double value) {
        // 1/(1+e^(-x))
        double sigmoid = 1.0 / (1 + Math.exp(value * (-1)));
        return sigmoid;
    }
}
