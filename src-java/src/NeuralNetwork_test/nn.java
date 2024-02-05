package NeuralNetwork_test;

import java.util.Arrays;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class nn {

    private double[][] weights1; // Pesos da primeira camada oculta
    private double[][] weights2; // Pesos da segunda camada oculta
    private double[][] weights3; // Pesos da terceira camada oculta
    private double[][] weightsOutput; // Pesos da camada de saída

    private double[] biases1; // Biases da primeira camada oculta
    private double[] biases2; // Biases da segunda camada oculta
    private double[] biases3; // Biases da terceira camada oculta
    private double[] biasesOutput; // Biases da camada de saída

    public nn(int inputSize, int hidden1Size, int hidden2Size, int hidden3Size, int outputSize) {
        // Inicializar pesos e biases com valores aleatórios
        weights1 = initializeWeights(inputSize, hidden1Size);
        weights2 = initializeWeights(hidden1Size, hidden2Size);
        weights3 = initializeWeights(hidden2Size, hidden3Size);
        weightsOutput = initializeWeights(hidden3Size, outputSize);

        biases1 = initializeBiases(hidden1Size);
        biases2 = initializeBiases(hidden2Size);
        biases3 = initializeBiases(hidden3Size);
        biasesOutput = initializeBiases(outputSize);
    }

    private double[][] initializeWeights(int inSize, int outSize) {
        // Implementar a lógica para inicializar pesos aleatórios com valores entre -1 e 1 e bem pequenos
        double[][] weights = new double[inSize][outSize];

        for (int i = 0; i < inSize; i++) {
            for (int j = 0; j < outSize; j++) {
                weights[i][j] = 0.003; // (Math.random() * 2 - 1) * 0.01;
            }
        }

        return weights;
    }

    private double[] initializeBiases(int size) {
        // Implementar a lógica para inicializar biases aleatórios com valores entre -1 e 1
        double[] bias = new double[size];

        for (int i = 0; i < size; i++) {
            bias[i] = 0.05; // (Math.random() * 2 - 1);
        }

        return bias;
    }

    private double[] relu(double[] x) {
        // Implementar a função de ativação ReLU
        return Arrays.stream(x).map(val -> Math.max(0, val)).toArray();
    }

    private double[] softmax(double[] x) {
        // Implementar a função de ativação Softmax
        double[] expValues = Arrays.stream(x).map(Math::exp).toArray();
        double sumExpValues = Arrays.stream(expValues).sum();
        return Arrays.stream(expValues).map(val -> val / sumExpValues).toArray();
    }
    
    private double[] matrixVectorMultiplication(double[] vector, double[][] matrix) {
        // Implementar a multiplicação de vetor por matriz
        double[] result = new double[matrix[0].length];
        for (int i = 0; i < matrix[0].length; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += vector[j] * matrix[j][i];
            }
        }
        return result;
    }

    private double[] vectorVectorSum(double[] vector1, double[] vector2) {
        // Implementar a soma de vetores
        double[] result = new double[vector1.length];
        for (int i = 0; i < vector1.length; i++) {
            result[i] = vector1[i] + vector2[i];
        }
        return result;
    }

    public double[] predict(double[] input) {
        // Propagação para frente (forward propagation)
        double[] product1 = matrixVectorMultiplication(input, weights1);
        double[] hidden1 = relu(vectorVectorSum(product1, biases1));
        //double[] hidden1 = (vectorVectorSum(product1, biases1));

        double[] product2 = matrixVectorMultiplication(hidden1, weights2);
        double[] hidden2 = relu(vectorVectorSum(product2, biases2));
        //double[] hidden2 = (vectorVectorSum(product2, biases2));

        double[] product3 = matrixVectorMultiplication(hidden2, weights3);
        //double[] hidden3 = (vectorVectorSum(product3, biases3));
        double[] hidden3 = relu(vectorVectorSum(product3, biases3));

        double[] productOutput = matrixVectorMultiplication(hidden3, weightsOutput);
        //double[] output = softmax(vectorVectorSum(productOutput, biasesOutput));
        double[] output = (vectorVectorSum(productOutput, biasesOutput));

        return output;
    }


    public static void main(String[] args) {
        // Exemplo de uso da rede neural
        int inputSize = 323;
        int hidden1Size = 646;
        int hidden2Size = 230;
        int hidden3Size = 115;
        int outputSize = 5;

        nn neuralNetwork = new nn(inputSize, hidden1Size, hidden2Size, hidden3Size, outputSize);

        long times2run = 50000;
        // Substitua o array de entrada pelos seus próprios dados
        double[] input = new double[inputSize];
        Arrays.fill(input, 1);

        // Marca o tempo de início
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < times2run; i++) {
            double[] output = neuralNetwork.predict(input);

            // Exibir a saída prevista
            //System.out.println("Output: " + Arrays.toString(output));
        }

        // Marca o tempo de fim
        long endTime = System.currentTimeMillis();

        // Exibir o tempo total de execução em milissegundos
        System.out.println("Total execution time: " + (endTime-startTime) + "ms");

        // Exibir o tempo médio de execução em microssegundos
        System.out.println("Average execution time: " + (endTime-startTime)/times2run + "ms");

        double[] output = neuralNetwork.predict(input);

        //Exibir a saída prevista
        System.out.println("Output: " + Arrays.toString(output));

    }
}