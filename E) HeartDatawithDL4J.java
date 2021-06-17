package org.deeplearning4j.examples.dataexamples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HeartData {
    private static Logger log = LoggerFactory.getLogger(HeartData.class);

    public static void main(String[] args) throws  Exception {

        //First: get the dataset using the record reader. CSVRecordReader handles loading or parsing
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter); //recordReader is the Variable. Instance of the Class. CSVRecordReader is the Class that takes 2 arguments that are inside the parenthesis
        recordReader.initialize(new FileSplit(new ClassPathResource("HeartData.txt").getFile())); //Method is the initialize. Filesplit is the Method to split the data to Training & Test Data

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 13;     //Since the labels for the Heart starts at the last column, out of the Total 14 Columns in the CSV File.
        int numClasses = 2;     //Classes have integer values 0, 1. Total 2 classes (starting from 0 to 1) in the heart data set.
        int batchSize = 180;    //The number of training instances used in 1 iteration. Heart data set: 180 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next(); //it would point to the next record.
        allData.shuffle(); // Shuffle the values in a single column, make predictions using the resulting data-set. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.80);  //Used 80% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance, which is a Standard Normalization):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set


        final int numInputs = 13; //columns
        int outputNum = 2; // 2 levels
        long seed = 6;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.1))
            .l2(1e-4)
            .list() // 3 layers used here although we are just predicting 2 classes here, Heart Disease Present or Not
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(2) //Input Layer
                .build())
            .layer(1, new DenseLayer.Builder().nIn(2).nOut(2)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(2).nOut(outputNum).build())
            .backprop(true).pretrain(false) //pre-train is used to specify a certain value of weights so that weights get converged quickly
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100)); // ScoreIterationListener will simply print the current error score for your network.Iterations will run for < 3999 times & starting from 0, each Iteration will jump to another 100 times.

        for(int i=0; i<4000; i++ ) {
            model.fit(trainingData);
        }

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(2); //Creates an Evaluation object with 2 Classes
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
        System.out.println(eval.accuracy());
        System.out.println(eval.precision());
        System.out.println(eval.recall());

        //eval.getConfusionMatrix(); Added this one to check
        //eval.getConfusionMatrix().toHTML();

    }


















}
