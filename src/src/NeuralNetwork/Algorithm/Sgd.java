package NeuralNetwork.Algorithm;

import NeuralNetwork.Layer;
import NeuralNetwork.Network;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by SB on 08/06/2017.
 */
public class Sgd extends Algorithm {

    private Network net;
    private Layer l;

    public Sgd(Network net, Layer l) {
        this.net = net;
        this.l = l;
    }

    public INDArray calculateDerivative(INDArray derivative){
        INDArray result = derivative.mul(net.getLearninRate());
        result.mul(-1);
        result.add(l.getWeight().mul(net.getWeightDecay()));// this would be weigh decay
        return result;
    }
}
