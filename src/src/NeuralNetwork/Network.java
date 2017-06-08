package NeuralNetwork;
/*
* created by Levensworth
* info:
*   this class is intended to create a feedforward neural network
*   the hyperparameter for the net are configured bia a builder object
*
* */


import NeuralNetwork.Algorithm.Adam;
import NeuralNetwork.Algorithm.Algorithm;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class Network{

    private double learninRate;
    private double momentum;
    private double weightDecay;
    private Activation.ACTIVATION activation;
    private Cost.COST cost;
    private List<Layer> layerList;
    private INDArray lastActivation;
    private int epoch;
    private Cost costFunction;
    private double beta_1;
    private double beta_2;
    private double epsilon;
    private long currentEpoch;
    private Algorithm.ALGORITHMS algorithm;


    public Network(double learninRate, double momentum, double weightDecay, Activation.ACTIVATION activation,
                   Cost.COST cost, List<Layer> layerList, int epoch, double beta_1, double beta_2, double epsilon,
                   Algorithm.ALGORITHMS algorithm) {
        if(learninRate < 0 || momentum < 0 || weightDecay < 0 || activation == null || layerList == null || epoch < 0)
            throw new IllegalArgumentException("the parameters where wrong");
        this.learninRate = learninRate;
        this.momentum = momentum;
        this.weightDecay = weightDecay;
        this.activation = activation;
        this.cost = cost;
        this.layerList = layerList;
        lastActivation = null;
        this.epoch = epoch;
        costFunction = new Cost(cost);
        this.beta_1 = beta_1;
        this.beta_2 = beta_2;
        this.epsilon = epsilon;
        this.currentEpoch = 0;
        this.algorithm = algorithm;
    }

    public long getCurrentEpoch(){
        return currentEpoch;
    }
    public double getLearninRate() {
        return learninRate;
    }

    public double getMomentum() {
        return momentum;
    }

    public double getWeightDecay() {
        return weightDecay;
    }

    public Activation.ACTIVATION getActivation() {
        return activation;
    }

    public Cost.COST getCost() {
        return cost;
    }

    public INDArray predict(INDArray input){
        if(input.columns() != layerList.get(0).getInput())
            throw  new IllegalArgumentException(" the input does not match the number of input neurons");

        INDArray activation = layerList.get(0).propagate(input);

        for (int i = 1; i <layerList.size() ; i++) {
            activation =  layerList.get(i).propagate(activation);
        }

        lastActivation = activation;

        return activation;
    }

    public double calculateCost (INDArray lastActivation, INDArray output){
        return costFunction.getCost(lastActivation, output);
    }


    public Algorithm.ALGORITHMS getAlgorithm(){
        return algorithm;
    }
    public double getBeta_1() {
        return beta_1;
    }

    public double getBeta_2() {
        return beta_2;
    }

    public double getEpsilon() {
        return epsilon;
    }



    public void train(INDArray input ,INDArray output){

        if(input == null || input.columns() != layerList.get(0).getInput())
            throw  new RuntimeException("the network hasn't been used yet");
        for (int current = 0; current < epoch; current++) {
            currentEpoch++;
            INDArray error  = output.add(predict(input).mul(-1));
            for (int i = layerList.size()-1; i >= 0 ; i--) {
                error = layerList.get(i).backPropagate(error);

            }


        }


        System.out.println("propagation complete");

        System.out.println(" the error for this network is " + calculateCost(lastActivation,output));
    }

}

