package Model.NeuralNetwork;

import Model.Movie;
import Model.MovieFeedBack;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

/*
* info : this class will be the interface between end user and the neural net
*
* methods:
*   getRecomendations()
*   @params:
*   int amount
*   @output
*   collection<Model.Movie>
*
*   getRecomendations()
*   @params:
*   void
*   @output
*   collection<Model.Movie>
*       default size = 3
*
*
*   sendFeedBack()
*   @params:
*       Collection<MovieFeedBack>
*
*    @output:
*       void
*
*    info:
   *    this method updates the weights
* */
public class API {

    private Network net;

    private boolean trained;

    private Collection<Movie> moviedb;

    private ArrayList<String> genres;

    public API(Collection<Movie> DB) {
        // creates a default neural net which cannot be modify.
        if (DB == null)
            throw new IllegalArgumentException("the API needs a movie data base from where to read");

        moviedb = DB;
        int input = 30;// the input vector created from the DB
        net = new NetworkBuilder()
                .setCost(Cost.COST.MSE)
                .addLayer(0, new Layer(input, 200, Activation.ACTIVATION.LEAKY_RELU))
                .addLayer(1, new Layer(200, 300, Activation.ACTIVATION.LEAKY_RELU))
                .addLayer(2, new Layer(300, 40, Activation.ACTIVATION.LEAKY_RELU))
                .addLayer(3, new OutpuLayer(40, 1, Activation.ACTIVATION.TANH))
                .setEpoch(1000)
                .setLearningRate(0.1)
                .setWeightDecay(0.95)
                .setMomentum(0.2)
                .build();


        createGenres();
        //train the network
        trained = false;


    }


    public Collection<Movie> getRecommendation() {
        //if it's the first time you will get the moview with
        //imdb score higher than 7

        ArrayList<Movie> recomendation = new ArrayList<Movie>();
        int maxRecomendaiton = 3;
        double minRating = 7;
        int index = 0;
        if (trained == false) {
            for (Movie mov : moviedb) {
                if (index == maxRecomendaiton) {
                    return recomendation;
                }
                if (mov.getIMDbScore() >= minRating) {
                    recomendation.add(mov);
                    index++;
                }

            }
        }

        for (Movie mov : moviedb) {
            if (prediction(mov) >= minRating) {
                recomendation.add(mov);
            }
        }

        return recomendation;

    }


    private double prediction(Movie movie) {
        // creates a vector out of the movie's genres and feeds it to the net
        // it's a dummy boolean vector
        INDArray mov = Nd4j.zeros(1, genres.size());

        for (String gen : movie.getGenre()) {
            mov.putScalar(new int[]{0, genres.indexOf(gen.toLowerCase())}, 1);
        }

        return net.predict(mov).getDouble(0, 0);
    }

    private void createGenres() {
        ArrayList<String> gen = new ArrayList<>();
        //creates a vector of all genres in the db
        for (Movie mov : moviedb) {
            for (String genre : mov.getGenre()) {
                if (!gen.contains(genre.toLowerCase()))
                    gen.add(genre.toLowerCase());
            }
        }

        Collections.sort(gen);

        genres = gen;
    }


    public void sendFeedBack(Collection<MovieFeedBack> feedbBack) {

        //this class will have to get the movies rated by the user
        INDArray movies_rated = Nd4j.zeros(feedbBack.size(), genres.size());

        INDArray user_ratings = Nd4j.zeros(feedbBack.size(), 1);
        int index = 0;
        for (MovieFeedBack rates : feedbBack) {
            for (String genre : rates.getMovie().getGenre()) {
                movies_rated.putScalar(new int[]{index, genres.indexOf(genre.toLowerCase())}, 1);
            }

            user_ratings.putScalar(new int[]{index, 0}, rates.getRating());
            index++;
        }

        net.train(movies_rated, user_ratings);
        trained = true;

    }


}
