package NeuralNetwork;

import sun.nio.ch.Net;

import java.util.ArrayList;
import java.util.List;

import static NeuralNetwork.Cost.COST.NSE;
/*
* this class will reacieve and adapt the hyperparams
* for a neural network , leting the user configure just enough
* for it to be simple
*
* params:
*   learning rate;
*   momentum
*   weight decay
*   activation function (limited functionality for the moment)
*   loss function   (limited functionality )
* */

public class NetworkBuilder{

    private double learningRate;
    private double momentum;
    private double weightDecay;
    private Activation.ACTIVATION activation;
    private Cost.COST cost;
    private List<Layer> layersList;
    private int epoch;
    private double beta_1;
    private double beta_2;
    private double epsilon;
    public NetworkBuilder(){
        learningRate = 0.01;
        momentum = 0.9;
        weightDecay = 0.0;
        activation = Activation.ACTIVATION.SIGMOID;
        cost = NSE;
        layersList = new ArrayList<Layer>();
        epoch = 0;
        beta_1 = 0.9;
        beta_2 = 0.99;
        epsilon = 0.0000001;

    }

    public NetworkBuilder setBeta_1(double beta_1) {
        this.beta_1 = beta_1;
        return this;
    }

    public NetworkBuilder setBeta_2(double beta_2) {
        this.beta_2 = beta_2;
        return this;
    }

    public NetworkBuilder setEpsilon(double epsilon) {
        this.epsilon = epsilon;
        return this;
    }

    public NetworkBuilder setLearningRate(double x){
        if( x  < 0 ){
            throw  new IllegalArgumentException();
        }

        this.learningRate = x;
        return this;
    }

    public NetworkBuilder setMomentum(double x){
        if( x  < 0 ){
            throw  new IllegalArgumentException();
        }

        this.momentum= x;
        return this;
    }

    public NetworkBuilder setWeightDecay(double x){
        if( x < 0)
            throw  new IllegalArgumentException();

        this.weightDecay = x;
        return this;
    }

    public NetworkBuilder setActivation(Activation.ACTIVATION fun){
        this.activation = fun;
        return this;
    }

    public NetworkBuilder setCost(Cost.COST cost){
        this.cost = cost;
        return this;
    }

    public NetworkBuilder addLayer(int index , Layer l){
        if(index < 0 )
            throw  new IllegalArgumentException();
        if(index > layersList.size())
            layersList.add(l);
        else
            layersList.add(index, l);
        return this;
    }

    public NetworkBuilder setEpoch(int epochs){
        epoch = epochs;
        return this;
    }

    public Network build(){
        if(layersList.isEmpty())
            throw  new RuntimeException("there must be at leat two layers specified");
        Network net = new Network(learningRate, momentum, weightDecay, activation,cost, layersList, epoch,beta_1,beta_2,epsilon);
        for(Layer l : layersList){
            l.setNetwork(net);
        }
        return  net;

    }


}