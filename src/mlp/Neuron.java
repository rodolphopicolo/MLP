package mlp;

import mlp.activation.ActivationFunction;
import mlp.exception.InputSizeException;

/**
 *
 * @author rodolpho
 */
public class Neuron {
    private final double[] weights;
    private double output;
    private double bias;
    private final ActivationFunction activationFunction;
    
    private final int layerPosition;
    private final int neuronPosition;
    
    private final double MIN_INITIALIZATION_VALUE = -0.5;
    private final double MAX_INITIALIZATION_VALUE =  0.5;
    
    public Neuron(int axons, ActivationFunction activationFunction, int layerPosition, int neuronPosition){
        this.weights = new double[axons];
        this.activationFunction = activationFunction;
        
        this.layerPosition = layerPosition;
        this.neuronPosition = neuronPosition;
    }
    
    public double getOutput(){
        return this.output;
    }
    
    public void initializeWeightsNBias(){
        this.bias = Helper.random(MIN_INITIALIZATION_VALUE, MAX_INITIALIZATION_VALUE);
        
        for(int i = 0; i < weights.length; i++){
            this.weights[i] = Helper.random(MIN_INITIALIZATION_VALUE, MAX_INITIALIZATION_VALUE);
        }
    }
    
    public double activate(double[] inputValues) throws InputSizeException{
        if(this.weights.length != inputValues.length){
            throw new InputSizeException();
        }
        
        double sum = this.bias;
        int size = inputValues.length;
        for(int i = 0; i < size; i++){
            sum += this.weights[i] * inputValues[i];
        }
        
        this.output = this.activationFunction.calculate(sum);
        
        return this.output;
    }
    
    @Override
    public String toString(){
        String text = "Neuron [" + String.valueOf(this.layerPosition) + ", " + String.valueOf(this.neuronPosition) + "]";
        return text;
    }
}