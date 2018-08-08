package mlp;

import mlp.dataset.Sample;
import java.util.ArrayList;
import java.util.List;
import mlp.activation.ReLu;
import mlp.dataset.Dataset;
import mlp.exception.InputSizeException;

/**
 *
 * @author rodolpho
 */
public class MLP {

    private final List<Neuron[]> layers;
    private final int inputParametersQuantity;
    private final int[] topology;

    public MLP(int[] topology, int inputParametersQuantity) {
        this.layers = new ArrayList();
        this.topology = topology;
        this.inputParametersQuantity = inputParametersQuantity;

        for (int i = 0; i < topology.length; i++) {
            addLayer(topology[i]);
        }
    }

    private void addLayer(int neurons) {
        int parameters;
        if (layers.isEmpty()) {
            //input layer
            parameters = inputParametersQuantity;
        } else {
            parameters = layers.get(layers.size() - 1).length;
        }

        int layerPosition = layers.size();
        Neuron[] layer = new Neuron[neurons];
        for (int neuronPosition = 0; neuronPosition < neurons; neuronPosition++) {
            layer[neuronPosition] = new Neuron(parameters, new ReLu(), layerPosition, neuronPosition);
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

    public synchronized TrainingState train(Dataset dataset, double maxError, Integer maxEpochs, Long maxTime) throws InputSizeException {
        TrainingState trainingState = new TrainingState(maxError, maxEpochs, maxTime);

        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                try {
                    int datasetSize = dataset.getSize();

                    trainingState.startTraining();
                    while (true) {
                        for (int i = 0; i < datasetSize; i++) {
                            Sample sample = dataset.getSample(i);
                            feedForward(sample.getFeatures());
                            if (trainingState.informCompleteEpoch(1, 1)) {
                                return;
                            }
                            backpropagate();
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

    private void feedForward(double[] values) throws InputSizeException {

        double[] currentInput = values;

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

    private void backpropagate() {
    }
}
