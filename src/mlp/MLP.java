package mlp;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import mlp.dataset.Sample;
import java.util.ArrayList;
import java.util.List;
import mlp.activation.Sigmoid;
import mlp.dataset.Dataset;
import mlp.exception.InputSizeException;
import mlp.exception.NoSampleException;
import mlp.loss.LossFunction;
import mlp.loss.MeanSquaredError;

/**
 *
 * @author rodolpho
 */
public class MLP {

    private final List<Neuron[]> layers;
    private final int inputSize;
    private final int[] topology;
    private double currentError;

    public MLP(int[] hiddenLayersTopology, int inputSize, int outputSize) {
        this.layers = new ArrayList();
        this.topology = new int[hiddenLayersTopology.length + 1];
        
        System.arraycopy(hiddenLayersTopology, 0, this.topology, 0, hiddenLayersTopology.length);
        this.topology[this.topology.length - 1] = outputSize;
        
        this.inputSize = inputSize;

        for (int i = 0; i < topology.length; i++) {
            int layerInputSize;
            if(i == 0){
                layerInputSize = inputSize;
            } else {
                layerInputSize = topology[i - 1];
            }
            addLayer(topology[i], layerInputSize);
        }
    }

    private void addLayer(int neuronSize, int inputSize) {
        int layerPosition = layers.size();
        Neuron[] layer = new Neuron[neuronSize];
        for (int neuronPosition = 0; neuronPosition < neuronSize; neuronPosition++) {
            layer[neuronPosition] = new Neuron(inputSize, new Sigmoid(), layerPosition, neuronPosition, layers);
        }
        layers.add(layer);
    }

    public void inicializeWeights() {
        for (int i = 0; i < layers.size(); i++) {
            Neuron[] layer = layers.get(i);
            for (int j = 0; j < layer.length; j++) {
                Neuron neuron = layer[j];
                neuron.initializeWeights();
            }
        }
    }

    public synchronized TrainingState train(Dataset dataset, double learningRate, double maxError, Integer maxEpochs, Long maxTime) throws InputSizeException {
        TrainingState trainingState = new TrainingState(maxError, maxEpochs, maxTime);
        LossFunction lossFunction = new MeanSquaredError();
        Runnable runnable;
        runnable = new Runnable() {
            @Override
            public void run() {
                try {
                    int datasetSize = dataset.getSize();
                    int epoch = 0;
                    double datasetError = 0;
                    double maxError = 0;
                    double currentError = 0;
                    trainingState.startTraining();
                    while (true) {
                        datasetError = 0;
                        maxError = 0;
                        System.out.format("Epochs: %d ---------------------\n", epoch);
                        for (int i = 0; i < datasetSize; i++) {
                            Sample sample = dataset.getSample(i);
                            feedForward(sample);
                            
                            calculateError(sample);
                            datasetError += MLP.this.currentError;
                            
                            if(MLP.this.currentError > maxError){
                                maxError = MLP.this.currentError;
                            }

                            backpropagate(learningRate, sample);
                            displayOutput(sample, maxError);
                        }
                        epoch++;
                        datasetError = datasetError / datasetSize;
                        
                        //System.out.println("Error: " + BigDecimal.valueOf(maxError).setScale(7, RoundingMode.HALF_EVEN).toPlainString() + " Epochs: " + epoch);
                        if (trainingState.informCompleteEpoch(maxError, 1)) {
                            System.out.format("Epochs: %d ---------------------\n", epoch);
                            for (int i = 0; i < datasetSize; i++) {
                                Sample sample = dataset.getSample(i);
                                feedForward(sample);
                                displayOutput(sample, maxError);
                            }
                            return;
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

    private void feedForward(Sample sample) throws InputSizeException, NoSampleException {

        double[] currentInput = sample.getInput();;

        for (int i = 0; i < this.layers.size(); i++) {
            Neuron[] layer = this.layers.get(i);

            double[] inputForNextLayer = new double[layer.length];

            for (int j = 0; j < layer.length; j++) {
                Neuron neuron = layer[j];
                neuron.setCurrentSample(sample);
                double output = neuron.activate(currentInput);
                inputForNextLayer[j] = output;
            }

            currentInput = inputForNextLayer;
        }
    }

    private void backpropagate(double learningRate, Sample sample) {
        for(int i = layers.size() - 1; i >= 0; i--){
            Neuron[] neurons =  layers.get(i);
            for(int j = 0; j < neurons.length; j++){
                Neuron neuron = neurons[j];
                neuron.calculateDelta();
                neuron.calculateCorrectionsFactor(learningRate);
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
    
    private void displayOutput(Sample sample, double error) throws IOException{
        sample.write(System.out);

        Neuron[] lastLayer = this.layers.get(this.layers.size() - 1);
        System.out.print("\t");
        for(int i = 0; i < lastLayer.length; i++){
            Neuron neuron = lastLayer[i];
            if(i > 0){
                System.out.print(";");
            }
            System.out.format("%.6f", neuron.getOutput());
        }
        System.out.format("\tError: %.4f\n", this.currentError);
    }
    
    private void calculateError(Sample sample){
        LossFunction lossFunction = new MeanSquaredError();
        Neuron[] outputLayer = this.layers.get(this.layers.size() - 1);
        double totalError = 0;
        for(int i = 0; i < outputLayer.length; i++){
            Neuron neuron = outputLayer[i];
            double outputValue = neuron.getOutput();
            double targetValue = sample.getOutput()[i];

            //double error = lossFunction.calculate(targetValue, outputValue);
            
            double error = targetValue - outputValue;

            neuron.setError(error);
            
            totalError += error;
        }
        this.currentError = totalError;
    }
}   