package mlp.dataset;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 *
 * @author rodolpho
 */
public class Dataset {

    private final List<Sample> samples;
    private final HashMap<String, Integer> labelsMap;
    
    private Dataset(List<Sample> samples, HashMap<String, Integer> labelsMap){
        this.samples = samples;
        this.labelsMap = labelsMap;
    }
    
    public void write(OutputStream outputStream) throws IOException{
        for(int i = 0; i < samples.size(); i++){
            if(i > 0){
                outputStream.write('\n');
            }
            Sample sample = samples.get(i);
            sample.write(outputStream);
        }
    }
    
    public int getSize(){
        return this.samples.size();
    }
    
    public Sample getSample(int index){
        return this.samples.get(index);
    }
    

    public static Dataset load(String filePath) throws FileNotFoundException, IOException, Exception{
        File file = new File(filePath);
        FileReader fileReader = new FileReader(file);

        HashMap<String, Integer> labelsMap = new HashMap();
        int lastLabelIndex = -1;
        
        int featuresQuantity = -1;
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        String line = bufferedReader.readLine();
        List<Sample> samples = new ArrayList();
        while(line != null){

            String[] splitted = line.split("\t");
            
            if(featuresQuantity == -1){
                featuresQuantity = splitted.length - 1;
            } else if(featuresQuantity != splitted.length - 1){
                throw new Exception("Features quantity differs from sample to sample");
            }
            
            String label = splitted[0];
            double[] features = new double[featuresQuantity];
            for(int i = 1; i < splitted.length; i++){
                double feature = Double.parseDouble(splitted[i]);
                features[i - 1] = feature;
            }
            
            if(labelsMap.containsKey(label) == false){
                lastLabelIndex++;
                labelsMap.put(label, lastLabelIndex);
            }
            
            Sample sample = new Sample(label, features);
            samples.add(sample);
            
            line = bufferedReader.readLine();
        }
        Dataset dataset = new Dataset(samples, labelsMap);
        return dataset;
    }    
}