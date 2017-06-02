package Model.NeuralNetwork;



import static java.lang.Math.exp;
//import java.lang.Math.tanh;

public class Activation{

    public static enum ACTIVATION {
        SIGMOID, TANH, LEAKY_RELU
    }

    private double  leakeage;

    private ACTIVATION function;

    public Activation(ACTIVATION act, double leakeage){

        if(leakeage > 0)
            this.leakeage = leakeage;
        else
            throw new IllegalArgumentException("the leakeage should be a non zero positive value");

        function = act;



    }

    private double sigmoid(double x){
        return 1/(1+ exp(-x));
    }

    private double sigmoidPrime(double x){
        return sigmoid(x)*(1-sigmoid(x));
    }

    private double tanh(double x){
        return Math.tanh(x);
    }

    private double tanhPrime(double x){
        return 1 - (tanh(x)*tanh(x));
    }

    private double relu(double x){
        return (x > 0)? x : leakeage * x;
    }
    private double reluPrime(double x){
        return (x > 0)? 1 : leakeage;
    }
    private ACTIVATION getFunction(){
        return function;
    }


    public double actiavte(double x){
        switch (getFunction()){
            case SIGMOID :
                return sigmoid(x);

            case  TANH :
                return tanh(x);

            case LEAKY_RELU:
                return relu(x);

            default:
                throw new RuntimeException("the activation function wasn't properly set");
        }
    }


}

