package mlp.dataset;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import mlp.exception.InvalidSampleFormat;

/**
 *
 * @author rodolpho
 */
public class Dataset {

    private List<Sample> samples;
    
    
    private Dataset(){}
    
    private void setSamples(List<Sample> samples){
        this.samples = samples;
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

        int featuresSize = -1;
        int outputSize = -1;
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        String line = bufferedReader.readLine();
        List<Sample> samples = new ArrayList();
        Dataset dataset = new Dataset();
        while(line != null){

            line = line.trim();

            if(line.startsWith("#") || line.isEmpty()){
                continue;
            }
            
            String[] inputOutput = line.split("\\|");
            if(inputOutput.length != 2){
                throw new InvalidSampleFormat(line);
            }
            
            String[] input = inputOutput[0].split(";");
            if(input.length == 0){
                throw new InvalidSampleFormat(line);
            }
            
            String[] output = inputOutput[1].split(";");
            if(output.length == 0){
                throw new InvalidSampleFormat(line);
            }

            if(featuresSize == -1){
                featuresSize = input.length;
            } else if(featuresSize != input.length){
                throw new Exception("Features size differs from sample to sample");
            }
            
            if(outputSize == -1){
                outputSize = output.length;
            } else if(outputSize != output.length){
                throw new Exception("Output size differs from sample to sample");
            }
            
            double[] dInput = new double[featuresSize];
            for(int i = 0; i < input.length; i++){
                dInput[i] = Double.parseDouble(input[i]);
            }

            double[] dOutput = new double[outputSize];
            for(int i = 0; i < output.length; i++){
                dOutput[i] = Double.parseDouble(output[i]);
            }
            
            Sample sample = new Sample(dInput, dOutput);
            samples.add(sample);
            
            line = bufferedReader.readLine();
        }
        dataset.setSamples(samples);

        return dataset;
    }    
}