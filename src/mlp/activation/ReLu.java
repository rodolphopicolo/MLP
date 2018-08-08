package mlp.activation;

/**
 *
 * @author rodolpho
 */
public class ReLu extends ActivationFunction {
    @Override
    public double calculate(double value) {
        return (value < 0 ? 0 : value);
    }
    
}
