package mlp;

import java.util.Random;

/**
 *
 * @author rodolpho
 */
public class Helper {
    
    private static final Random RANDOM = new Random();
    
    public static double random(double minValue, double maxValue){
        double w = RANDOM.nextDouble();
        w = w * (maxValue - minValue) + minValue;
        return w;
    }    
}
