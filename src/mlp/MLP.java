package mlp;

import java.io.IOException;
import mlp.dataset.Sample;
import java.util.ArrayList;
import java.util.List;
import mlp.activation.Sigmoid;
import mlp.dataset.Dataset;
import mlp.exception.InputSizeException;
import mlp.loss.LossFunction;
import mlp.loss.MeanSquaredError;

/**
 *
 * @author rodolpho
 */
public class MLP {

    private final List<Neuron[]> layers;
    private final int inputParametersQuantity;
    private final int[] topology;
    private double currentError;

    public MLP(int[] topology, int inputParametersQuantity, int datasetFeaturesQuantity) {
        this.layers = new ArrayList();
        this.topology = topology;
        this.inputParametersQuantity = inputParametersQuantity;

        for (int i = 0; i < topology.length; i++) {
            addLayer(topology[i], datasetFeaturesQuantity);
        }
    }

    private void addLayer(int neurons, int datasetFeaturesQuantity) {
        int parameters;
        if (layers.isEmpty()) {
            //input layer
            parameters = datasetFeaturesQuantity;
        } else {
            parameters = layers.get(layers.size() - 1).length;
        }

        int layerPosition = layers.size();
        Neuron[] layer = new Neuron[neurons];
        for (int neuronPosition = 0; neuronPosition < neurons; neuronPosition++) {
            layer[neuronPosition] = new Neuron(parameters, new Sigmoid(), layerPosition, neuronPosition);
        }

        layers.add(layer);
    }

    public void inicializeWeights() {
        for (int i = 0; i < layers.size(); i++) {
            Neuron[] layer = layers.get(i);
            for (int j = 0; j < layer.length; j++) {
                Neuron neuron = layer[j];
                neuron.initializeWeightsNBias();
            }
        }
    }

    public synchronized TrainingState train(Dataset dataset, double learningRate, double maxError, Integer maxEpochs, Long maxTime) throws InputSizeException {
        TrainingState trainingState = new TrainingState(maxError, maxEpochs, maxTime);
        LossFunction lossFunction = new MeanSquaredError();
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                try {
                    int datasetSize = dataset.getSize();
                    int epoch = 0;
                    trainingState.startTraining();
                    while (true) {
                        double datasetError = 0;
                        for (int i = 0; i < datasetSize; i++) {
                            epoch++;
                            Sample sample = dataset.getSample(i);
                            feedForward(sample.getFeatures());
                            
                            calculateError(dataset, sample);
                            datasetError += MLP.this.currentError;
                            
                            if (trainingState.informCompleteEpoch(datasetError, 1)) {
                                return;
                            }
                            backpropagate(learningRate, dataset, sample);
                            displayOutput(sample, epoch, datasetError);
                        }
                    }
                } catch (Exception ex) {
                    trainingState.finishWithError(ex);
                    return;
                }
            }
        };
        
        Thread thread = new Thread(runnable);
        thread.start();

        return trainingState;

    }

    private void feedForward(double[] inputValuesForInputLayer) throws InputSizeException {

        double[] currentInput = inputValuesForInputLayer;

        for (int i = 0; i < this.layers.size(); i++) {
            Neuron[] layer = this.layers.get(i);

            double[] inputForNextLayer = new double[layer.length];

            for (int j = 0; j < layer.length; j++) {
                Neuron neuron = layer[j];
                double output = neuron.activate(currentInput);
                inputForNextLayer[j] = output;
            }

            currentInput = inputForNextLayer;
        }
    }

    private void backpropagate(double learningRate, Dataset dataset, Sample sample) {
        for(int i = layers.size() - 1; i >= 0; i--){
            Neuron[] neurons =  layers.get(i);
            for(int j = 0; j < neurons.length; j++){
                Neuron neuron = neurons[j];
                int dentrites = neuron.dentrites();
                double outputValue = neuron.getOutput();
                
                
                        
                int expectedActiveNeuron = dataset.outputNeuronIndexForLabel(sample.getLabel());
                double expectedValue = (expectedActiveNeuron == j ? 1: 0);
                        
                double[] ws = neuron.getWeights();
                for(int k = 0; k < dentrites; k++){
                    //double w = ws[k];

                    double inputValue = neuron.getInputValue(k);
                    
                    double correction;
                    if(i == layers.size() - 1){
                        //OutputLayer
                        correction = calculateCorrectionFactorForOutputLayer(learningRate, expectedValue, outputValue, inputValue);
                    } else {
                        //HiddenLayer
                        double deltaForOutputLayer = deltaForOutputLayer(expectedValue, outputValue);
                        correction = calculateCorrectionFactorForHiddenLayer(deltaForOutputLayer, ws, inputValue, learningRate, outputValue);
                    }
                    neuron.setCorrection(k, correction);
                }
            }
        }
        
        for(int i = layers.size() - 1; i >= 0; i--){
            Neuron[] neurons =  layers.get(i);
            for(int j = 0; j < neurons.length; j++){
                Neuron neuron = neurons[j];
                neuron.applyCorrection();
            }
        }
    }
    
    private double deltaForOutputLayer(double sampleValue, double calculatedActivationValue){
        double delta = (sampleValue - calculatedActivationValue) * calculatedActivationValue * (1 - calculatedActivationValue);
        return delta;
    }
    
    private double calculateCorrectionFactorForOutputLayer(double learningRate, double expectedValue, double outputLayerCalculatedValue, double previousHiddenLayerCalculatedValue){
        double delta = deltaForOutputLayer(expectedValue, outputLayerCalculatedValue);
        double correction = learningRate * delta * previousHiddenLayerCalculatedValue;
        return correction;
    }
    
    private double deltaForHiddenLayer(double deltaForOutputLayer, double[] wLastEdges, double calculatedValueOfHiddenLayer){
        double delta = 0;
        for(int i = 0; i < wLastEdges.length; i++){
            delta += deltaForOutputLayer * wLastEdges[i];
        }
        delta = delta * calculatedValueOfHiddenLayer * (1 - calculatedValueOfHiddenLayer);
        return delta;
    }
    
    private double calculateCorrectionFactorForHiddenLayer(double deltaForOutputLayer, double[] wLastEdges, double calculatedValueOfHiddenLayer, double learningRate, double calculatedValue){
        double delta = deltaForHiddenLayer(deltaForOutputLayer, wLastEdges, calculatedValueOfHiddenLayer);
        double correction = learningRate * delta * calculatedValue;
        return correction;
    }
    
    private void displayOutput(Sample sample, int epoch, double datasetError) throws IOException{
        System.out.println("=================================================");
        System.out.println("Epoch " + epoch + " Dataset error: " + datasetError);
        sample.write(System.out);
        System.out.println("\tError: " + this.currentError);
        Neuron[] lastLayer = this.layers.get(this.layers.size() - 1);
        for(int i = 0; i < lastLayer.length; i++){
            Neuron neuron = lastLayer[i];
            System.out.println("Neuron: " + i + "; Output: " + neuron.getOutput());
        }
    }
    
    private void calculateError(Dataset dataset, Sample sample){
        LossFunction lossFunction = new MeanSquaredError();
        int targetNeuronIndex = dataset.outputNeuronIndexForLabel(sample.getLabel());
        Neuron[] outputLayer = this.layers.get(this.layers.size() - 1);
        double totalError = 0;
        for(int i = 0; i < outputLayer.length; i++){
            Neuron neuron = outputLayer[i];
            double outputValue = neuron.getOutput();
            double targetValue = (targetNeuronIndex == i ? 1: 0);

            double error = lossFunction.calculate(targetValue, outputValue);

            neuron.setError(error);
            
            totalError += error;
        }
        this.currentError = totalError;
    }
}   