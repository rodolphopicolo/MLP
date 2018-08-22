package mlp.dataset;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;

/**
 *
 * @author rodolpho
 */
public class Sample {
    
    private static final double NUMERIC_LABEL_INCREMENT = 1;
    
    private final String label;
    private final double[] features;
    private final Dataset dataset;
    
    public Sample(String label, double[] features, Dataset dataset){
        this.label = label;
        this.features = new double[features.length];
        this.dataset = dataset;
        System.arraycopy(features, 0, this.features, 0, features.length);
    }
    
    public int featuresQuantity(){
        return this.features.length;
    }
    
    public double[] getFeatures(){
        return this.features;
    }
    
    public String getLabel(){
        return this.label;
    }
    
    public int activateNeuron(){
        return this.dataset.outputNeuronIndexForLabel(this.label);
    }
    
    public void write(OutputStream outputStream) throws UnsupportedEncodingException, IOException{
        String message = "Label " + this.label;
        message += "\tNeuron " + this.activateNeuron();
        message += "\tFeatures: ";
        outputStream.write(message.getBytes("UTF-8"));
        for(int i = 0; i < features.length; i++){
            if(i > 0){
                outputStream.write(';');
            }
            outputStream.write(String.valueOf(features[i]).getBytes("UTF-8"));
        }
    }
    
}
