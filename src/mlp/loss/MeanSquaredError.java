package mlp.loss;

/**
 *
 * @author rodolpho
 */
public class MeanSquaredError implements LossFunction {
    @Override
    public double calculate(double targetValue, double calculatedValue) {
        double error = Math.pow((targetValue - calculatedValue), 2)/2;
        return error;
    }
    
}
