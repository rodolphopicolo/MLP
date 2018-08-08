package mlp.dataset;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 *
 * @author rodolpho
 */
public class Sample {
    
    private static final double NUMERIC_LABEL_INCREMENT = 1;
    
    private final String label;
    private final double[] features;
    
    public Sample(String label, double[] features){
        this.label = label;
        this.features = new double[features.length];
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
    
    public void write(OutputStream outputStream) throws UnsupportedEncodingException, IOException{
        outputStream.write(this.label.getBytes("UTF-8"));
        for(int i = 0; i < features.length; i++){
            outputStream.write('\t');
            outputStream.write(String.valueOf(features[i]).getBytes("UTF-8"));
        }
    }
    

}
