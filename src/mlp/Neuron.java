package mlp;

import mlp.activation.ActivationFunction;
import mlp.exception.InputSizeException;

/**
 *
 * @author rodolpho
 */
public class Neuron {
    private final int dentrites;
    private final double[] weights;
    private final double[] corrections;
    private double[] inputValues;
    private double output;
    private double bias;
    private final ActivationFunction activationFunction;
    
    private final int layerPosition;
    private final int neuronPosition;
    
    private double error;
    
    private final double MIN_INITIALIZATION_VALUE = -0.5;
    private final double MAX_INITIALIZATION_VALUE =  0.5;
    
    public Neuron(int dentrites, ActivationFunction activationFunction, int layerPosition, int neuronPosition){
        this.dentrites = dentrites;
        this.weights = new double[dentrites];
        this.corrections = new double[dentrites];
        this.activationFunction = activationFunction;
        
        this.layerPosition = layerPosition;
        this.neuronPosition = neuronPosition;
    }
    
    public double getOutput(){
        return this.output;
    }
    
    public double[] getWeights(){
        return this.weights;
    }
    
    public void setCorrection(int index, double value){
        this.corrections[index] = value;
    }
    
    public void setError(double error){
        this.error = error;
    }
    
    public double getError(){
        return this.error;
    }
    
    void applyCorrection(){
        for(int i = 0; i < dentrites; i++){
            this.weights[i] = this.weights[i] + this.corrections[i];
        }
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
        
        this.inputValues = inputValues;
        
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
    
    public double getWeight(int index){
        return this.weights[index];
    }
    
    public double getInputValue(int index){
        return this.inputValues[index];
    }
    
    public int dentrites(){
        return this.dentrites;
    }
    
    
}