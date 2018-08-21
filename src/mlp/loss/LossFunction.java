package mlp.loss;

/**
 *
 * @author rodolpho
 */
public interface LossFunction {
    public abstract double calculate(double targetValue, double calculatedValue);
}
