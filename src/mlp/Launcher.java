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
    private static final String TOPOLOGY = "TOPOLOGY";
    
    public static void main(String[] args) throws Exception{
        
        Date date = new Date();
        
        OffsetDateTime d1 = date.toInstant().atZone(ZoneId.systemDefault()).toOffsetDateTime();
         System.out.println(d1.format(DateTimeFormatter.ISO_OFFSET_DATE_TIME));
        
        
        
        String datasetFilePath = null;
        int[] topology = null;
        
        for(int i = 0; i < args.length; i+=2){
            if(args[i].equalsIgnoreCase(DATASET)){
                datasetFilePath = args[i+1];
            } else if(args[i].equalsIgnoreCase(TOPOLOGY)){
                String[] splitted = args[i+1].split(";");
                topology = new int[splitted.length];
                for(int j = 0; j < splitted.length; j++){
                    topology[j] = Integer.parseInt(splitted[j]);
                }
            }
        }
        if(datasetFilePath == null){
            throw new Exception("No file name with dataset for training and test specified");
        }
        Dataset dataset = Dataset.load(datasetFilePath);
        dataset.write(System.out);
        System.out.write('\n');
        
        System.out.print("\nTopology: ");
        for(int i = 0; i < topology.length; i++){
            if(i > 0){
                System.out.print("; ");
            }
            System.out.print(String.valueOf(topology[i]));
        }
        System.out.println("");

        MLP mlp = new MLP(topology, dataset.getSize(), dataset.getSample(0).featuresQuantity());
        mlp.inicializeWeights();
        mlp.train(dataset, 0.3, 0.01, 100, null);
    }
}
