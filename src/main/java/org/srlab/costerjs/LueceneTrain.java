package org.srlab.costerjs;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class LueceneTrain {

    private static void print(Object s){System.out.println(s.toString());}

    private static void train(){

        List<CodeToken> trainingData = new ArrayList<>();

        try{
            BufferedReader bufferedReader = new BufferedReader(new FileReader(Config.TRAIN_DATASET_FILE_PATH));
            String line = bufferedReader.readLine();
            while ((line=bufferedReader.readLine())!= null)
            {
                String[] values = line.split(",");
                if (values.length == 6)
                {
                    CodeToken codeToken = new CodeToken(values[0],values[1],values[2],values[3],values[4],values[5]);
                    trainingData.add(codeToken);
                }
            }

        } catch (Exception ex){
            System.err.println("Error Occured while collecting training data in the Luecene.jar file!!!!");
            ex.printStackTrace();
        }
        finally {
            LueceneTrain.createLueceneIndexFile(trainingData);
        }


    }

    private static void createLueceneIndexFile(List<CodeToken> trainingData){

        List<Document> docs = new ArrayList<>();

        for(CodeToken codeToken:trainingData){

            Document document = new Document();
            FieldType contentType = new FieldType();
            contentType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
            contentType.setStored(true);
            contentType.setTokenized(true);

            FieldType idType = new FieldType();
            idType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
            idType.setStored(true);
            idType.setTokenized(true);

            document.add(new Field("context",codeToken.getContext(),contentType));
            document.add(new Field("type",""+codeToken.getActualLabel(),idType));
            docs.add(document);
        }
        FSDirectory dir;
        try {
            dir = FSDirectory.open(Paths.get(Config.LUCENE_INVERTED_INDEX_FILE_PATH));
            IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
            IndexWriter writer = new IndexWriter(dir, config);
            writer.deleteAll();
            writer.addDocuments(docs);
            writer.commit();
            writer.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
    private static ArrayList<String> recommend(String query){
        //first load the createLueceneIndexFile structure
        Directory dir;
        ArrayList<String> recommendations = new ArrayList<>();
        try {
            dir = FSDirectory.open(Paths.get(Config.LUCENE_INVERTED_INDEX_FILE_PATH));
            IndexReader reader = DirectoryReader.open(dir);
            IndexSearcher searcher = new IndexSearcher(reader);
            Similarity searcherSimilarity=searcher.getSimilarity();
            QueryParser queryParser = new QueryParser("context", new StandardAnalyzer());

            Query parsedQuery = queryParser.parse(QueryParser.escape(query));
            TopDocs hits = searcher.search(parsedQuery, 100,Sort.RELEVANCE);
            for(ScoreDoc sd:hits.scoreDocs){
                Document d = searcher.doc(sd.doc);
                print(d);
                print(sd);
                recommendations.add(d.getField("context").stringValue());
            }
        }catch(Exception e){
            e.printStackTrace();
        }
        return recommendations;
    }
    public static void main(String[] args) {
        // TODO Auto-generated method stub

        System.out.println("Choose one of the following option:\n1.train\n2.test");

        Scanner scanner = new Scanner(System.in);

        args = new String[2];
        args[0]=scanner.next();


        if(args[0].equals("train"))
            LueceneTrain.train();

        if(args[0].equals("test")){
            System.out.println("Please provide Query String:");
            args[1]=scanner.next();
            LueceneTrain.recommend(args[1]);

        }
    }

}