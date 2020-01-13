package org.srlab.costerjs;

public class Config {
    //Path to the project (Compulsary)
    private static final String ROOT_FOLDER = "/home/khaledkucse/Project/backup/costerjs/";

    //Train and Test file path
    public static final String TRAIN_DATASET_FILE_PATH = ROOT_FOLDER + "data/train.csv/";// filename of data corpus to learn
    public static final String TEST_DATASET_FILE_PATH = ROOT_FOLDER + "data/test.csv/"; //Path of the vocabulary of input language (context/previous code tokens)


    //Leucene model file path
    public static final String LUCENE_INVERTED_INDEX_FILE_PATH = ROOT_FOLDER + "model/lucene_inverted.index";// filename of the trained luecene index file

}
