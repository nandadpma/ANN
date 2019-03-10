package elm;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
//@author nanda
class Elm {
    static double [][]dataCSV;
    static double []alldata;
    static String activator;
    double max, min;
    int jumlah_neuron;
    int hidden_layer;
    double range_weight = 1;
    double [][] training_data, testing_data;
    double []training_target, testing_target;
    int trainingS, trainingT, testingS, testingT;
    double [][]weight;
    double []Bheta;
    double []predict;
    double []error;
    double MAPE;
    
    
    public double[][] gaussJordan(double [][]a){
        double [][] identitas = new double[a.length][a[0].length];
        for(int i = 0; i < identitas.length; i++){
            for(int j = 0; j < identitas[i].length; j++){
                if(i==j){
                    identitas[i][j] = 1;
                }
            }
        }
        for(int i = 0; i < a.length; i++){
            double diagonal = a[i][i];
            for(int j = 0; j < a[i].length; j++){
                a[i][j] = a[i][j]/diagonal;
                identitas[i][j] = identitas[i][j]/diagonal; 
            }
            for(int k = 0; k < a.length; k++){
                if(k!=i){
                    double x = a[k][i];
                    for(int l = 0; l < a[k].length; l++){
                        a[k][l] = a[k][l]-(x*a[i][l]);
                        identitas[k][l] = identitas[k][l]-(x*identitas[i][l]);
                    }
                }
            }
        }
        return identitas;
    }
    public Elm(int trainingS, int trainingT, int testingS, int testingT, int jumlah_neuron, int hidden_layer){
        this.trainingS = trainingS;
        this.trainingT = trainingT;
        this.testingS = testingS;
        this.testingT = testingT;
        this.jumlah_neuron = jumlah_neuron;
        this.hidden_layer = hidden_layer;
    }
    public void pairingNeuron2Target(){
        double [][]all_neuron = new double[alldata.length-this.jumlah_neuron-1][this.jumlah_neuron];
        double []target = new double[alldata.length-this.jumlah_neuron-1];
        for(int i = 0; i < alldata.length; i++){
        }
        for(int i = 0; i < alldata.length-this.jumlah_neuron-1; i++){
            double []neuron = new double[this.jumlah_neuron];
            for(int j = i; j < i+this.jumlah_neuron; j++){
                neuron[j-i] = alldata[j];
            }
            all_neuron[i] = neuron;
            target[i] = alldata[i+this.jumlah_neuron];
        }
        this.training_data = new double[this.trainingT-this.trainingS][this.jumlah_neuron];
        this.training_target = new double[this.trainingT-this.trainingS];
        this.testing_data = new double[this.testingT-this.testingS][this.jumlah_neuron];
        this.testing_target = new double[this.testingT-this.testingS];
        for(int i = this.trainingS; i < this.trainingT; i++){
            this.training_data[i-this.trainingS] = all_neuron[i-this.trainingS];
            this.training_target[i-this.trainingS] = target[i-this.trainingS];
        }
        for(int i = this.testingS; i < this.testingT; i++){
            this.testing_data[i-this.testingS] = all_neuron[i-this.testingS];
            this.testing_target[i-this.testingS] = target[i-this.testingS];
        }
    }
    
    public void runELM(){
        inisialize();
        training();
        testing();
    }
    
    public void inisialize(){
        normalisasi();
        pairingNeuron2Target();
    }
    
    public double [][]generateWeight(){
        double [][]weight = new double[this.hidden_layer][this.jumlah_neuron];
        //System.out.println("WEIGHT");
        for(int i = 0; i < this.hidden_layer; i++){
            double w = Math.random();
            double plusminus = Math.random();
            if(plusminus>0.5){w = this.range_weight * w;}
            else{w = -1 * this.range_weight * w;}
            for(int j = 0; j < this.jumlah_neuron; j++){
                weight[i][j] = w;
                //System.out.print(" "+w);
            }//System.out.println("");
        }
        return weight;
    }
    
    public double [][]transpose(double [][]matrix){
        double [][]result = new double[matrix[0].length][matrix.length];
        for(int i = 0; i < result.length; i++){
            for(int j = 0; j < result[i].length; j++){
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }
    
    public double [][]matmul(double [][]a, double [][]b){
        double [][] result = new double[a.length][b[0].length];
        for(int i = 0; i < a.length; i++){
            for(int j = 0; j < b[0].length; j++){
                double mul = 0;
                for(int k = 0; k < a[i].length; k++){
                    mul = mul + (a[i][k] * b[k][j]);
                }
                result[i][j] = mul;
            }
        }
        return result;
    }
    
    public double []matmul(double [][]a, double []b){
        double [] result = new double[a.length];
        for(int i = 0; i < a.length; i++){
            double mul = 0;
            for(int j = 0; j < a[i].length; j++){
                mul = mul + (a[i][j]*b[j]);
            }
            result[i] = mul;
        }
        return result;
    }
    
    public double [][] aktivasi(double [][] Hinit, String activator){
        double [][]H = new double[Hinit.length][Hinit[0].length];
        if(activator=="Sigmoid"){
            for(int i = 0; i < Hinit.length; i++){
                for(int j = 0; j < Hinit[i].length; j++){
                    H[i][j] = 1/(1+(Math.exp(-1*Hinit[i][j])));
                }
            }
        }
        else if(activator=="Scaled Sigmoid"){
            for(int i = 0; i < Hinit.length; i++){
                for(int j = 0; j < Hinit[i].length; j++){
                    H[i][j] = (2/(1+(Math.exp(-2*Hinit[i][j]))))-1;
                }
            }
        }
        else if(activator=="Hard Limit"){
            for(int i = 0; i < Hinit.length; i++){
                for(int j = 0; j < Hinit[i].length; j++){
                    if(Hinit[i][j]>=0){
                        H[i][j] = 1;
                    }
                    else{
                        H[i][j] = 0;
                    }
                }
            }
        }
        else if(activator=="RBF"){
            for(int i = 0; i < Hinit.length; i++){
                for(int j = 0; j < Hinit[i].length; j++){
                    H[i][j] = Math.exp(-1*Math.pow(Hinit[i][j], 2));
                }
            }
        }
        return H;
    }
    
    public void hitungMAPE(double []target, double []prediksi){
        double sum = 0;
        this.error = new double[target.length];
        //System.out.println("Error");
        for(int i = 0; i < target.length; i++){
            //error[i] = ((target[i]-prediksi[i])/target[i]);
            error[i] = Math.abs((target[i]-prediksi[i])/target[i]);
            //System.out.println(prediksi[i]+"\t"+target[i]+"\t\t"+Math.abs((target[i]-prediksi[i])/target[i])+"\t"+error[i]);
            sum = sum + error[i];
        }
        this.MAPE = (sum/target.length)*100;
        //System.out.println("MAPE : "+this.MAPE);
    }
    
    public void training(){
        this.weight = generateWeight();
        double [][]Hinit = matmul(this.training_data, transpose(this.weight));
        double [][]H = aktivasi(Hinit, activator);
        double [][]Hdagger = matmul(gaussJordan(matmul(transpose(H), H)), transpose(H));
        this.Bheta = matmul(Hdagger, this.training_target);
    }
    
    public void testing(){
        double [][]Hinit = matmul(this.testing_data, transpose(this.weight));
        double [][]H = aktivasi(Hinit, activator);
        this.predict = matmul(H,this.Bheta);
        double []denormalized_predict = denormalisasi(predict);
        double []denormalized_target = denormalisasi(this.testing_target);
        hitungMAPE(denormalized_target, denormalized_predict);
    }
    
    public void normalisasi(){
        this.max = this.min = this.dataCSV[0][0];
        for(int i = 1; i < this.dataCSV[0].length; i++){
            if(this.dataCSV[0][i]>this.max){
                this.max = this.dataCSV[0][i];
            }
        }
        for(int i = 1; i < this.dataCSV[0].length; i++){
            if(this.dataCSV[0][i]<this.min){
                this.min = this.dataCSV[0][i];
            }
        }
        for(int i = 0; i < alldata.length; i++){
            alldata[i] = (this.dataCSV[0][i]-this.min)/(this.max-this.min);
        }
    }
    
    public double []denormalisasi(double []normalized){
        double []denormalized = new double[normalized.length];
        for(int i = 0; i < normalized.length; i++){
            denormalized[i] = (normalized[i]*(this.max-this.min))+this.min;
        }
        return denormalized;
    }
    
    public static void readCSV(String fname){
        int lineOfCSV = 0;
        int columnOfCSV = 0;
        try {
            Scanner stream = new Scanner(new File(fname));
            while(stream.hasNext()){
                String []tokens = stream.next().split(",");
                columnOfCSV = tokens.length;
                lineOfCSV++;
            }stream = new Scanner(new File(fname));
            dataCSV = new double[lineOfCSV][columnOfCSV];
            alldata = new double[columnOfCSV];
            String [][]data = new String[lineOfCSV][columnOfCSV];
            int line = 0;
            while(stream.hasNext()){
                data[line] = stream.next().split(",");
                line++;
            }
            stream.close();
            for(int i = 0; i < dataCSV.length; i++){
                for(int j = 0; j < dataCSV[i].length; j++){
                    dataCSV[i][j] = Double.parseDouble(data[i][j]);
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Elm.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
public class ExtremeLearningMachine{
    
    public static double findMinMAPE(double []allMAPE){
        double min = allMAPE[0];
        for(int i = 1; i < allMAPE.length; i++){
            if(allMAPE[i]<min){
                min = allMAPE[i];
            }
        }
        return min;
    }
    
    public static void main(String[] args) {
        Elm.readCSV("D:\\prediksi harga cabai\\hargacabai2.csv");
        Elm.activator = "Sigmoid";
        Elm [][][]elm = new Elm[6][10][100];
        double [][]minMAPE = new double[elm.length][elm[0].length];
        for(int i = 1; i < elm.length; i++){
            for(int j = 2; j < elm[i].length; j++){
                System.out.println("=== Jumlah Neuron : "+i+" | Hidden Neuron : "+j+" ===");
                double []MAPE = new double[elm[i][j].length];
                for(int k = 0; k < elm[i][j].length; k++){
                    elm[i][j][k] = new Elm(0,81,81,101,i,j);
                    elm[i][j][k].runELM();
                    MAPE[k] = elm[i][j][k].MAPE;
                }
                minMAPE[i][j] = findMinMAPE(MAPE);
                System.out.println("MAPE : "+minMAPE[i][j]);
            }
        }
    }
}
