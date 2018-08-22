package mlp;

import java.util.List;
import mlp.activation.ActivationFunction;
import mlp.dataset.Sample;
import mlp.exception.InputSizeException;
import mlp.exception.NoSampleException;

/**
 *
 * @author rodolpho
 */
public class Neuron {
    private final int dentrites;
    private final double[] weights;
    private final double[] correctionsFactor;
    private double[] inputValues;
    private double output;
    private double bias;
    private double biasCorrectionFactor;
    private final ActivationFunction activationFunction;
    
    private Sample currentSample = null;
    
    private final List<Neuron[]> layers;
    
    private Double delta = null;
    
    private final int layerPosition;
    private final int neuronPosition;
    
    private double error;
    
    private final double MIN_INITIALIZATION_VALUE = -0.5;
    private final double MAX_INITIALIZATION_VALUE =  0.5;
    
    public Neuron(int dentrites, ActivationFunction activationFunction, int layerPosition, int neuronPosition, List<Neuron[]> layers){
        this.dentrites = dentrites;
        this.weights = new double[dentrites];
        this.correctionsFactor = new double[dentrites];
        this.activationFunction = activationFunction;
        
        this.layerPosition = layerPosition;
        this.neuronPosition = neuronPosition;
        
        this.layers = layers;
    }
    
    public double getOutput(){
        return this.output;
    }
    
    public double[] getWeights(){
        return this.weights;
    }
    
    public void setCorrection(int index, double value){
        this.correctionsFactor[index] = value;
    }
    
    public void setError(double error){
        this.error = error;
    }
    
    public double getError(){
        return this.error;
    }
    
    void applyCorrection(){
        for(int i = 0; i < dentrites; i++){
            this.weights[i] = this.weights[i] + this.correctionsFactor[i];
        }
    }
    
    public void initializeWeightsNBias(){
        this.bias = Helper.random(MIN_INITIALIZATION_VALUE, MAX_INITIALIZATION_VALUE);
        
        for(int i = 0; i < weights.length; i++){
            this.weights[i] = Helper.random(MIN_INITIALIZATION_VALUE, MAX_INITIALIZATION_VALUE);
        }
    }
    
    public double activate(double[] input) throws InputSizeException, NoSampleException{

        if(this.currentSample == null){
            throw new NoSampleException ();
        }
        
        if(this.weights.length != input.length){
            throw new InputSizeException();
        }
        
        this.inputValues = input;
        
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
    
    public void setCurrentSample(Sample sample){
        this.currentSample = sample;
    }
    

    public boolean isOutputLayer(){
        return layerPosition == (layers.size() - 1);
    }
    
    public double calculateDelta(){
        if(isOutputLayer()){
            
            double sampleValue = 0;
            if (this.currentSample.activateNeuron() == this.neuronPosition){
                sampleValue = 1;
            }
            
            this.delta = (sampleValue - this.output) * this.output * (1 - this.output);
        } else {
            
            double nextLayerDeltaSum = 0;
            
            Neuron[] nextLayer = layers.get(this.layerPosition + 1);
            for(int i = 0; i < nextLayer.length; i++){                
                Neuron neuronNextLayer = nextLayer[i];
                nextLayerDeltaSum += neuronNextLayer.calculateDelta() * neuronNextLayer.getWeight(this.neuronPosition);
            }
            
            this.delta = nextLayerDeltaSum * (this.output * (1 - this.output));
        }

        return this.delta;
    }
    
    public void calculateCorrectionsFactor(double learningRate){
        for(int i = 0; i < this.correctionsFactor.length; i++){
            double deltaW = learningRate * this.delta * this.inputValues[i];
            this.correctionsFactor[i] = deltaW;
        }
        this.biasCorrectionFactor = learningRate * this.delta;
    }
    
    public void applyCorrections(){
        for(int i = 0; i < this.weights.length; i++){
            this.weights[i] = this.weights[i] - this.correctionsFactor[i];
        }
        bias = bias - this.biasCorrectionFactor;
    }
    

}