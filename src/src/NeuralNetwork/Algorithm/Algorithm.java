package NeuralNetwork.Algorithm;

/*
 * Created by SB on 08/06/2017.
 *  this class will contain the anum of algorithms we have
 *
 * */

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Algorithm {
    public static enum ALGORITHMS {ADAM,SGD};

    public abstract INDArray calculateDerivative(INDArray derivative);

}
