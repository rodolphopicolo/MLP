package mlp;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author rodolpho
 */
public class TrainingState {
    public enum State {
        NOT_STARTED, TRAINING, FINISHED;
    }
    
    public enum StopReason {
        MAX_EPOCHS, MAX_TIME, DESIRED_ERROR_ACHIEVED, ERROR
    }
    
    private State state;
    private Long startTime;
    private Long endTime;
    
    private int processedEpochs = 0;
    
    private final double maxError;
    private final int maxEpochs;
    private final long maxTime;
    
    private double error;
    private double accuracy;
    private List<double[]> results;
    
    private StopReason stopReason;
    private Exception errorCause;
    
    public TrainingState(double maxError, Integer maxEpochs, Long maxTime){
        this.state = State.NOT_STARTED;
        this.startTime = null;
        this.results = new ArrayList();
        
        this.maxError = maxError;

        if(maxEpochs != null){
            this.maxEpochs = maxEpochs;
        } else {
            this.maxEpochs = Integer.MAX_VALUE;
        }
        
        if(maxTime != null){
            this.maxTime = maxTime;
        } else {
            this.maxTime = Long.MAX_VALUE;
        }
    }
    
    void startTraining(){
        this.state = State.TRAINING;
        this.startTime = System.currentTimeMillis();
    }
    
    private void finishTraining(StopReason stopReason){
        this.state = State.FINISHED;
        this.endTime = System.currentTimeMillis();
        this.stopReason = stopReason;
    }
    
    boolean informCompleteEpoch(double error, double accuracy){
        this.processedEpochs++;

        this.error = error;
        this.accuracy = accuracy;
        this.results.add(new double[]{error, accuracy});
        

        if(this.error <= this.maxError){
            finishTraining(StopReason.DESIRED_ERROR_ACHIEVED);
            return true;
        }
        
        if(this.processedEpochs > this.maxEpochs){
            finishTraining(StopReason.MAX_EPOCHS);
            return true;
        }
        
        if((System.currentTimeMillis() - this.startTime) > maxTime){
            finishTraining(StopReason.MAX_TIME);
            return true;
        }

        return false;
    }
    
    void finishWithError(Exception cause){
        this.state = State.FINISHED;
        this.stopReason = StopReason.ERROR;
        this.errorCause = cause;
    }
}
