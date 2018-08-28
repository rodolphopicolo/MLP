package mlp.exception;

/**
 *
 * @author rodolpho
 */
public class InvalidSampleFormat extends Exception {
    public InvalidSampleFormat(String sampleFormat){
        super(sampleFormat);
    }
}
