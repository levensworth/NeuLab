package NeuralNetwork.Algorithm;

import NeuralNetwork.Network;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static java.lang.Math.pow;

/**
 * Created by SB on 08/06/2017.
 * the class represents the adam back propagation algorithm
 * which will be called from each layer
 * as the error spreads
 */
public class Adam extends Algorithm {

    private INDArray firstMomentum;
    private INDArray secondMomentum;
    private int rows;// number of rows
    private  int cols;//number of columns
    private  Network net;

    public Adam(int rows, int cols, Network net) {
        this.rows = rows;
        this.cols = cols;
        this.net = net;

        firstMomentum = Nd4j.zeros(rows, cols);
        secondMomentum = Nd4j.zeros(rows, cols);

    }

    /*
    * the method needs
    * @Input: the derivative calculated by the backpropagation method in each layer
    * @output: a matrix Correction for the weights in the matrix
    * */
    public INDArray calculateDerivative(INDArray derivative){

        //calculates the adam momentum
        INDArray M = firstMomentum.mul(net.getBeta_1()).add(derivative.mul((1-net.getBeta_1())) );
        INDArray R = secondMomentum.mul(net.getBeta_2()).add(  Transforms.pow(derivative,2).mul( (1-net.getBeta_2()) )  );

        //update the momentum rates
        firstMomentum = M;
        secondMomentum = R;

        //now lets tune them with correction bias
        INDArray Mhat = M.div((1-pow(net.getBeta_1(), net.getCurrentEpoch())));
        INDArray Rhat = R.div((1-pow(net.getBeta_2(), net.getCurrentEpoch())));
        //
        INDArray aux1 = Transforms.sqrt(Rhat).add(net.getEpsilon());

        INDArray aux2 = Mhat.mul(net.getLearninRate());
        //element wise operation for division
        INDArray result = Nd4j.zeros(rows, cols);
        for (int i = 0; i < result.rows(); i++) {
            for (int j = 0; j < result.columns() ; j++) {
                result.putScalar(new int[]{i,j},aux2.getDouble(i,j) / aux1.getDouble(i,j));
            }
        }

        return  result;
    }


}
