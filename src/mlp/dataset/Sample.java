package mlp.dataset;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;

/**
 *
 * @author rodolpho
 */
public class Sample {

    private final double[] input;
    private final double[] output;
    private final String textRepresentation;

    public Sample(double[] input, double[] output) {
        this.input = new double[input.length];
        this.output = new double[output.length];

        System.arraycopy(input, 0, this.input, 0, input.length);
        System.arraycopy(output, 0, this.output, 0, output.length);

        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < input.length; i++) {
            if (i > 0) {
                stringBuilder.append(";");
            }
            stringBuilder.append(String.valueOf(input[i]));
        }
        stringBuilder.append("|");
        for (int i = 0; i < output.length; i++) {
            if (i > 0) {
                stringBuilder.append(";");
            }
            stringBuilder.append(String.valueOf(output[i]));
        }
        this.textRepresentation = stringBuilder.toString();
    }

    public int inputSize() {
        return this.input.length;
    }

    public double[] getInput() {
        return this.input;
    }

    public int outputSize() {
        return this.output.length;
    }

    public double[] getOutput() {
        return this.output;
    }

    @Override
    public String toString() {
        return this.textRepresentation;
    }

    public void write(OutputStream outputStream) throws UnsupportedEncodingException, IOException {
        outputStream.write(textRepresentation.getBytes("UTF-8"));
    }

}
