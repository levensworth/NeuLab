package NeuralNetwork;
import NeuralNetwork.Algorithm.Adam;
import NeuralNetwork.Algorithm.Algorithm;
import NeuralNetwork.Algorithm.Sgd;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.ops.transforms.Transforms;

import static java.lang.Math.pow;


/**
 * Created by SB on 28/05/2017.
 */
public class Layer {



    /*
    * a layer is the name of a weight matrix between two levels of the the net
    * */


    private int input;
    private int output;
    private INDArray weight;
    private INDArray bias;
    private INDArray delta;
    private INDArray activated;
    private INDArray nonActivated;
    private INDArray inputArray;
    private Activation.ACTIVATION function;
    private Network net;
    private  Algorithm algorithm;

    public Layer(int input, int output, Activation.ACTIVATION function){
        if(input < 0 || output < 0 || function == null)
            throw  new IllegalArgumentException("input and output must be positive integers and ACTIVATION not null");

        this.function = function;
        this.input = input;
        this.output = output;
        inputArray = null;
        initWeights();
        initBias();
        initDelta();
        initActivated();
        initNonActivated();

    }

    private void initWeights(){
        Random ran = Nd4j.getRandom();
        int[] shape = {input,output};
        weight = Nd4j.rand(shape , -1.0 , 1.0 ,  ran);

    }

    private  void initBias(){
        Random ran = Nd4j.getRandom();

        int[] shape = {1, output};
        bias = Nd4j.rand(shape,-1,1,ran);

    }

    private  void initDelta(){
        int[] shape = {input,output};
        delta = Nd4j.zeros(input, output);
    }


    public INDArray propagate(INDArray input){
        inputArray = input;

        INDArray zMatrix = input.mmul(getWeight());
        zMatrix.addRowVector(bias);
        nonActivated = zMatrix;
        switch (function){
            case SIGMOID:
                activated = Transforms.tanh(zMatrix);
                break;
            case LEAKY_RELU:
                activated = Transforms.leakyRelu(zMatrix);
                break;
            case TANH:
                activated = Transforms.tanh(zMatrix);
                break;
        }


        return activated;
    }

    private void setAlgorithm(Algorithm.ALGORITHMS algorithm){
        switch (algorithm){
            case ADAM:
                this.algorithm = new Adam(input, output, net);
                break;

            case SGD:
                this.algorithm = new Sgd(net, this);
                break;
        }
    }


    public INDArray backPropagate(INDArray delta){

        //computes the function derivative of the weigh matrix
        switch (function){
            case SIGMOID:
                this.delta = Nd4j.getExecutioner().execAndReturn(new HardSigmoidDerivative(nonActivated));
                break;
            case LEAKY_RELU:
                this.delta = Nd4j.getExecutioner().execAndReturn(new LeakyReLUDerivative(nonActivated));
                break;
            case TANH:
                this.delta = Nd4j.getExecutioner().execAndReturn(new HardTanhDerivative(nonActivated));
                break;
        }


        this.delta = this.delta.mul(delta);

        INDArray derivative = inputArray.transpose().mmul(this.delta);
        updateWeight(derivative);
        updateBias();
        return delta.mmul(getWeight().transpose());
    }

    protected void updateBias() {
        bias = bias.add(delta.sum(0).mul(-1));
    }

    protected void updateWeight(INDArray derivative) {

        weight = weight.add(algorithm.calculateDerivative(derivative));
    }

    public INDArray getWeight() {
        return weight;
    }

    public INDArray getDelta() {
        return delta;
    }

    public void initActivated(){
        activated = Nd4j.zeros(input, output);
    }

    public void initNonActivated(){
        nonActivated = Nd4j.zeros(input, output);
    }

    public int getInput() {
        return input;
    }




    public void setNetwork(Network net){
        if(net == null)
            throw new IllegalArgumentException("neural netwrok not found");

        this.net = net;
        setAlgorithm(net.getAlgorithm());// sets the algorithm to use
    }

    public void setDelta(INDArray delta) {
        this.delta = delta;
    }

    protected INDArray getInputArray() {
        return inputArray;
    }


}
