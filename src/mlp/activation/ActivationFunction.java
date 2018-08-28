package mlp.activation;

/**
 *
 * @author rodolpho
 */
public abstract class ActivationFunction {
    public abstract double calculate(double value);
    public abstract double dCalculate(double value);
}
