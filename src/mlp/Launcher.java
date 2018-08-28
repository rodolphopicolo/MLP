package mlp;

import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Date;
import mlp.dataset.Dataset;

/**
 *
 * @author rodolpho
 */
public class Launcher {
    
    private static final String DATASET = "DATASET";
    private static final String HIDDEN_LAYERS_TOPOLOGY = "HIDDEN_LAYERS_TOPOLOGY";
    private static final String MAX_ERROR = "MAX_ERROR";
    private static final String MAX_EPOCHS = "MAX_EPOCHS";
    private static final String LEARNING_RATE = "LEARNING_RATE";
    
    public static void main(String[] args) throws Exception{
        
        Date date = new Date();
        
        OffsetDateTime d1 = date.toInstant().atZone(ZoneId.systemDefault()).toOffsetDateTime();
         System.out.println(d1.format(DateTimeFormatter.ISO_OFFSET_DATE_TIME));
        
        String datasetFilePath = null;
        int[] hiddenLayersTopology = null;
        Double maxError = null;
        Integer maxEpochs = null;
        Double learningRate = null;
        
        for(int i = 0; i < args.length; i+=2){
            if(args[i].equalsIgnoreCase(DATASET)){
                datasetFilePath = args[i+1];
            } else if(args[i].equalsIgnoreCase(HIDDEN_LAYERS_TOPOLOGY)){
                String[] splitted = args[i+1].split(";");
                hiddenLayersTopology = new int[splitted.length];
                for(int j = 0; j < splitted.length; j++){
                    hiddenLayersTopology[j] = Integer.parseInt(splitted[j]);
                }
            } else if(args[i].equalsIgnoreCase(MAX_ERROR)){
                maxError = Double.parseDouble(args[i+1]);
            } else if(args[i].equalsIgnoreCase(MAX_EPOCHS)){
                maxEpochs = Integer.parseInt(args[i+1]);
            } else if(args[i].equalsIgnoreCase(LEARNING_RATE)){
                learningRate = Double.parseDouble(args[i+1]);                
            }
        }
        if(datasetFilePath == null){
            throw new Exception("No file name with dataset for training and test specified");
        }
        Dataset dataset = Dataset.load(datasetFilePath);
        //dataset.write(System.out);
        //System.out.write('\n');
        
        System.out.print("\nHidden layers topology: ");
        for(int i = 0; i < hiddenLayersTopology.length; i++){
            if(i > 0){
                System.out.print("; ");
            }
            System.out.print(String.valueOf(hiddenLayersTopology[i]));
        }
        System.out.println("");

        MLP mlp = new MLP(hiddenLayersTopology, dataset.getSample(0).inputSize(), dataset.getSample(0).outputSize());
        mlp.inicializeWeights();
        mlp.train(dataset, learningRate, maxError, maxEpochs, null);
    }
}