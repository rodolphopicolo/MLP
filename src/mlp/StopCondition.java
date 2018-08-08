package mlp;

/**
 *
 * @author rodolpho
 */
public class StopCondition {
    public enum StopReason {
        MAX_EPOCHS, MAX_TIME, DESIRED_ERROR_ACHIEVED, ERROR
    }
    
    private StopReason stopReason;
    private final double maxErrorAllowed;
    private int maxEpochsAllowed;
    private long maxTimeAllowed;
    
    public StopCondition(double maxErrorAllowed){
        this.maxErrorAllowed = maxErrorAllowed;
        this.maxEpochsAllowed = Integer.MAX_VALUE;
        this.maxTimeAllowed = Long.MAX_VALUE;
        this.stopReason = null;
    }

    public void setMaxEpochsAllowed(int maxEpochsAllowed) {
        this.maxEpochsAllowed = maxEpochsAllowed;
    }

    public void setMaxTimeAllowed(long maxTimeAllowed) {
        this.maxTimeAllowed = maxTimeAllowed;
    }
    
    public boolean stop(double currentError, int currentEpoch, long startTime){
        if(currentError <= maxErrorAllowed){
            this.stopReason = StopReason.DESIRED_ERROR_ACHIEVED;
            return true;
        }
        
        if(currentEpoch >= this.maxEpochsAllowed){
            this.stopReason = StopReason.MAX_EPOCHS;
            return true;
        }
        
        if((System.currentTimeMillis() - startTime) >= this.maxTimeAllowed){
            this.stopReason = StopReason.MAX_TIME;
            return true;
        }

        return false;
    }

}
