package mlp.activation;

/**
 *
 * @author rodolpho
 */
public class Sigmoid extends ActivationFunction {
    @Override
    public double calculate(double value) {
        // 1/(1+e^(-x))
        double sigmoid = 1.0 / (1 + Math.exp(-value));
        return sigmoid;
    }
    
    
    @Override
    public double dCalculate(double value){
        /*
            sigmoid = s = (1 + e^-x)^-1
            ds/dx = -(1+e^-x)^-2 * e^-x * -1 = e^-x / (1 + e^-x)^2 = s(1-s)
        */
        
        double sigmoid = calculate(value);
        double dSigmoid = sigmoid * (1 - sigmoid);
        return dSigmoid;
    }
}
