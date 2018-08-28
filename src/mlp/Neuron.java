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
        this.dentrites = dentrites + 1;
        this.weights = new double[this.dentrites];
        this.correctionsFactor = new double[this.dentrites];
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
    
    public void initializeWeights(){
        for(int i = 0; i < weights.length; i++){
            this.weights[i] = Helper.random(MIN_INITIALIZATION_VALUE, MAX_INITIALIZATION_VALUE);
        }
    }
    
    public double activate(double[] input) throws InputSizeException, NoSampleException{

        if(this.currentSample == null){
            throw new NoSampleException ();
        }
        
        if(this.weights.length != input.length + 1){//Bias
            throw new InputSizeException();
        }

        this.inputValues = new double[input.length + 1];  //+1 for bias
        System.arraycopy(input, 0, this.inputValues, 0, input.length);
        this.inputValues[this.inputValues.length - 1] = 1; //Bias
         
        double sum = 0;
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
        double dSigmoid = this.output * (1 - this.output);
        if(isOutputLayer()){

            double sampleValue = this.currentSample.getOutput()[this.neuronPosition];

            double error = (sampleValue - this.output);
            this.delta = error * dSigmoid;
        } else {
            
            double nextLayerDeltaSum = 0;
            
            Neuron[] nextLayer = layers.get(this.layerPosition + 1);
            for(int i = 0; i < nextLayer.length; i++){                
                Neuron neuronNextLayer = nextLayer[i];
                nextLayerDeltaSum += neuronNextLayer.calculateDelta() * neuronNextLayer.getWeight(this.neuronPosition);
            }
            
            this.delta = nextLayerDeltaSum * dSigmoid;
        }

        return this.delta;
    }
    
    public void calculateCorrectionsFactor(double learningRate){
        for(int i = 0; i < this.correctionsFactor.length; i++){
            double deltaW = learningRate * this.delta * this.inputValues[i];
            this.correctionsFactor[i] = deltaW;
        }
    }
    
    public void applyCorrections(){
        for(int i = 0; i < this.weights.length; i++){
            this.weights[i] = this.weights[i] - this.correctionsFactor[i];
        }
    }
}