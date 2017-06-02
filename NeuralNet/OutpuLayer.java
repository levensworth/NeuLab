package Model.NeuralNetwork;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by SB on 29/05/2017.
 */
public class OutpuLayer extends Layer {

    public OutpuLayer(int input, int output, Activation.ACTIVATION function) {
        super(input, output, function);
    }

    @Override
    public INDArray backPropagate(INDArray delta) {

        setDelta(delta);
        INDArray derivative = getInputArray().transpose().mmul(getDelta());
        updateWeight(derivative);
        updateBias();
        return delta.mmul(getWeight().transpose());
    }
}
