package Manager;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import Config.ParametersSimulation;
import Network.Structure.OpticalLink;
import RSA.Routing.Route;

public class FolderManager {

    private String folderName;
    private String folderPath;
    private boolean status;

    private OpticalLink[][] networkOpticalLinks;
    private int reqID;

    private int nSim;
    private int indexDF;
    private double networkLoad;

    /**
     * Classe para gerenciar o acesso a pasta de relatórios
     *
     * @param tagName
     * @throws Exception
     */
    public FolderManager(String tagName) throws Exception {

        this.indexDF = 0;

        // Data e hora do momento do início da simulação
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        Date date = new Date();
        String dateTime = sdf.format(date);

        this.folderName = dateTime + "_" + ParametersSimulation.getTopologyType() + "_" + ParametersSimulation.getRoutingAlgorithmType() + "_" + ParametersSimulation.getSpectralAllocationAlgorithmType() + "_" + ParametersSimulation.getRSAOrderType() + "_" + tagName;

        this.folderPath = ParametersSimulation.getPathToSaveResults() + this.folderName;

        this.status = new java.io.File(this.folderPath).mkdirs();

        if (this.status){
            System.out.println("Pasta criada com o nome: " + this.folderName);
        }
        else{
            throw new Exception("ERRO: Pasta não foi criada");
        }

        this.writeParameters();
    }

    public void setNetworkOpticalLinks(OpticalLink[][] networkOpticalLinks) {
        this.networkOpticalLinks = networkOpticalLinks;
    }

    public int getReqID() {
        return reqID;
    }

    public void setReqID(int reqID) {
        this.reqID = reqID;
    }

    /**
     * Método para salvar o conteúdo do arquivo.
     *
     * @param fileName
     * @param content
     */
    private void writeFile(String fileName, String content) {

        File file = new File(this.folderPath + "/" + fileName);
        try {
            file.createNewFile();
            FileWriter fileWriter = new FileWriter(file, true);
            fileWriter.write(content);
            fileWriter.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * Método para salvar o estado final da simulação, junto com o tempo de execução completo.
     *
     * @param totalTime Tempo de execução
     */
    public void writeDone(double totalTime) {

        String content = "Simulação finalizada com sucesso!\n" +
        "Tempo total de execução: " + totalTime + " milissegundos\n";

        this.writeFile("done.txt", content);

        // Renomeia a pasta para facilitar a visualização

        final File oldNameFile = new File(this.folderPath);
        final File newNameFile = new File(this.folderPath + "_DONE");
        oldNameFile.renameTo(newNameFile);

        this.folderPath = this.folderPath + "_DONE";

        System.out.println(this.folderPath);
    }

    /**
     *  Salva os paramétros dessa simulação
     */
    private void writeParameters() {
        this.writeFile("Parameters.txt", ParametersSimulation.save());
    }

    /**
     * Salva a topologia dessa simulação
     *
     * @param content
     */
    public void writeTopology(String content) {
        this.writeFile("Topology.txt", content);
    }

    /**
     * Salva as rotas dessa simulação
     *
     * @param content
     */
    public void writeRoutes(String content) {
        this.writeFile("Routes.txt", content);
    }

    /**
     * Salva os resultados dessa simulação
     *
     * @param content
     */
    public void writeResults(String content) {
        this.writeFile("Results.txt", content);
    }

    public boolean isStatus() {
        return status;
    }

    public String getFolderName() {
        return folderName;
    }

    public String getFolderPath() {
        return folderPath;
    }

    public void saveCapacityLossMSCLblock(List<Double> lostCapacityTotal, int sourceNodeID, int destinationNodeID, List<Integer> startSlot, final int reqSize) throws Exception {

        final int blockSlotsSize = 20;

        final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();

        int blockSlotsInit;
        int blockSlotsEnd;

        String slotsInformationAux;
        String data = "";

        for (int x = 0; x < lostCapacityTotal.size(); x++) {

            // Armazena as informações de uma estrutura de dados para os links
            slotsInformationAux = "";

            // Selecina somente os slots próximos a requisição que entrarão para o cálculo da capacidade.

            //blockSlotsInit = startSlot.get(x) + (int)Math.ceil((double)reqSize / 2) - (int)Math.ceil((double)blockSlotsSize / 2);
            blockSlotsInit = 125 + (int)Math.ceil((double)reqSize / 2) - (int)Math.ceil((double)blockSlotsSize / 2);

            blockSlotsEnd = blockSlotsInit + blockSlotsSize;

            if (blockSlotsInit < 0) {
                blockSlotsInit = 0;
                blockSlotsEnd = blockSlotsInit + blockSlotsSize;
            }

            if (blockSlotsEnd > numOfSlots) {
                blockSlotsInit = numOfSlots - blockSlotsSize;
                blockSlotsEnd = numOfSlots;
            }

            int test = 0;

            for (int o = 0; o < this.networkOpticalLinks.length; o++) {
                for (int d = 0; d < this.networkOpticalLinks.length; d++) {
                    OpticalLink optLink = this.networkOpticalLinks[o][d];
                    if (optLink != null) {
                        for (int s = blockSlotsInit; s < blockSlotsEnd; s++) {
                            if (optLink.isAvailableSlotAt(s)){
                                slotsInformationAux += "0;";
                                test++;
                            } else {
                                slotsInformationAux += "1;";
                                test++;
                            }
                        }
                    }
                }
            }

            if (test != 40) {
                throw new Exception("Erro dos vintes");
            }

            final String slotsInformation = slotsInformationAux.substring(0, slotsInformationAux.length() - 1);

            //Linha de dados
            data += this.reqID + ";" + lostCapacityTotal.get(x) + ";" + sourceNodeID +";" + destinationNodeID + ";" + startSlot.get(x) + ";" + reqSize + ";" + slotsInformation +"\n";
        }

        this.writeFile("data_MSCL" + this.reqID % 3000 + ".csv", data);
    }


    public void saveCapacityLossMSCL(List<Double> lostCapacityTotal, int sourceNodeID, int destinationNodeID, List<Integer> startSlot, final int reqSize) throws Exception {

        final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();

        String slotsInformationAux;
        // Armazena as informações de uma estrutura de dados para os links
        slotsInformationAux = "";

        for (int o = 0; o < this.networkOpticalLinks.length; o++) {
            for (int d = 0; d < this.networkOpticalLinks.length; d++) {
                OpticalLink optLink = this.networkOpticalLinks[o][d];
                if (optLink != null) {
                    for (int s = 0; s < numOfSlots; s++) {
                        if (optLink.isAvailableSlotAt(s)){
                            slotsInformationAux += "0;";
                        } else {
                            slotsInformationAux += "1;";
                        }
                    }
                }
            }
        }
        final String slotsInformation = slotsInformationAux.substring(0, slotsInformationAux.length() - 1);

        String data = "";

        for (int x = 0; x < lostCapacityTotal.size(); x++) {

            //Linha de dados
            data += this.reqID + ";" + lostCapacityTotal.get(x) + ";" + sourceNodeID +";" + destinationNodeID + ";" + startSlot.get(x) + ";" + reqSize + ";" + slotsInformation +"\n";
        }

        this.writeFile("data_MSCL" + this.reqID % 3000 + ".csv", data);
    }

    public void saveDataMSCLBin(final int reqID, final int bestSlotToCapacity, final int bestLostCapacity, final int sourceNodeID, final int destinationNodeID, final int reqSize) throws Exception {

        final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();

        try {
            // Criar um objeto FileOutputStream para o arquivo "dados.bin"
            FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "dataMSCL.bin", true);
            // Criar um objeto DataOutputStream para escrever no FileOutputStream
            DataOutputStream dos = new DataOutputStream(fos);

            // Armazena os meta dados
            dos.writeInt(reqID);
            //dos.write(reqID);
            dos.write(bestSlotToCapacity);
            dos.write(bestLostCapacity);
            dos.write(sourceNodeID);
            dos.write(destinationNodeID);
            dos.write(reqSize);

            // System.out.print(reqID);
            // System.out.print(',');
            // //dos.write(reqID);
            // System.out.print(bestSlotToCapacity);
            // System.out.print(',');
            // System.out.print(bestLostCapacity);
            // System.out.print(',');
            // System.out.print(sourceNodeID);
            // System.out.print(',');
            // System.out.print(destinationNodeID);
            // System.out.print(',');
            // System.out.print(reqSize);
            // System.out.print(',');

            OpticalLink optLink = this.networkOpticalLinks[0][1];
            for (int j = 0; j < numOfSlots; j++) { // Para cada coluna da linha
                if (optLink.isAvailableSlotAt(j)){
                    dos.write(0);
                } else {
                    dos.write(1);
                }
            }

            //Salva a ocupação dos links
            // for (int o = 0; o < this.networkOpticalLinks.length; o++) {
            //     for (int d = 0; d < this.networkOpticalLinks.length; d++) {
            //         OpticalLink optLink = this.networkOpticalLinks[o][d];
            //         if (optLink != null) {
            //             for (int s = 0; s < numOfSlots; s++) {
            //                 if (optLink.isAvailableSlotAt(s)){
            //                     dos.write(0);
            //                     System.out.print(0);
            //                     System.out.print(',');
            //                 } else {
            //                     dos.write(1);
            //                     System.out.print(1);
            //                     System.out.print(',');
            //                 }
            //             }
            //         }
            //     }
            // }

            //System.out.println("END");

            // Fechar o DataOutputStream e o FileOutputStream
            dos.close();
            fos.close();
        } catch (IOException e) {
            System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
        }
    }

    public void saveDataBinLists(int reqID2, int reqNumbOfSlots, List<Integer> saveSlotIndexes,
        List<Integer> saveCapacityLoss, List<Float> saveCapacityLossMean, List<Integer> saveCurrentSlotReqs, List<Integer> saveBestLostCapacity,
        List<Integer> saveBestSlotToCapacity, int source, int destination) {

        final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();

        try {
            // Criar um objeto FileOutputStream para o arquivo "dados.bin"
            FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "dataMSCL.bin", true);
            // Criar um objeto DataOutputStream para escrever no FileOutputStream
            DataOutputStream dos = new DataOutputStream(fos);

            for (int i = 0; i < saveSlotIndexes.size(); i++){

                // Armazena os meta dados
                dos.writeInt(reqID2);
                dos.write(reqNumbOfSlots);
                
                dos.write(source);
                dos.write(destination);
                
                dos.write(saveSlotIndexes.get(i));
                dos.write(saveCapacityLoss.get(i));
                dos.writeFloat(saveCapacityLossMean.get(i));
                dos.write(saveCurrentSlotReqs.get(i));

                if (saveCurrentSlotReqs.get(i) == 2){
                    dos.write(saveBestLostCapacity.get(0));
                    dos.write(saveBestSlotToCapacity.get(0));
                }

                if (saveCurrentSlotReqs.get(i) == 3){
                    dos.write(saveBestLostCapacity.get(1));
                    dos.write(saveBestSlotToCapacity.get(1));
                }

                if (saveCurrentSlotReqs.get(i) == 6){
                    dos.write(saveBestLostCapacity.get(2));
                    dos.write(saveBestSlotToCapacity.get(2));
                }

                OpticalLink optLink = this.networkOpticalLinks[0][1];
                for (int j = 0; j < numOfSlots; j++) { // Para cada coluna da linha
                    if (optLink.isAvailableSlotAt(j)){
                        dos.write(0);
                    } else {
                        dos.write(1);
                    }
                }
                optLink = this.networkOpticalLinks[0][5];
                for (int j = 0; j < numOfSlots; j++) { // Para cada coluna da linha
                    if (optLink.isAvailableSlotAt(j)){
                        dos.write(0);
                    } else {
                        dos.write(1);
                    }
                }
                optLink = this.networkOpticalLinks[1][2];
                for (int j = 0; j < numOfSlots; j++) { // Para cada coluna da linha
                    if (optLink.isAvailableSlotAt(j)){
                        dos.write(0);
                    } else {
                        dos.write(1);
                    }
                }
                optLink = this.networkOpticalLinks[1][4];
                for (int j = 0; j < numOfSlots; j++) { // Para cada coluna da linha
                    if (optLink.isAvailableSlotAt(j)){
                        dos.write(0);
                    } else {
                        dos.write(1);
                    }
                }
                optLink = this.networkOpticalLinks[2][3];
                for (int j = 0; j < numOfSlots; j++) { // Para cada coluna da linha
                    if (optLink.isAvailableSlotAt(j)){
                        dos.write(0);
                    } else {
                        dos.write(1);
                    }
                }
                optLink = this.networkOpticalLinks[2][5];
                for (int j = 0; j < numOfSlots; j++) { // Para cada coluna da linha
                    if (optLink.isAvailableSlotAt(j)){
                        dos.write(0);
                    } else {
                        dos.write(1);
                    }
                }
                optLink = this.networkOpticalLinks[3][4];
                for (int j = 0; j < numOfSlots; j++) { // Para cada coluna da linha
                    if (optLink.isAvailableSlotAt(j)){
                        dos.write(0);
                    } else {
                        dos.write(1);
                    }
                }
                optLink = this.networkOpticalLinks[5][4];
                for (int j = 0; j < numOfSlots; j++) { // Para cada coluna da linha
                    if (optLink.isAvailableSlotAt(j)){
                        dos.write(0);
                    } else {
                        dos.write(1);
                    }
                }
            }

            // Fechar o DataOutputStream e o FileOutputStream
            dos.close();
            fos.close();
        } catch (IOException e) {
            System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
        }
    }

    public void saveTopologyState(int sourceNodeID, int destinationNodeID) throws Exception {

        final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();

        String slotsInformationAux;
        // Armazena as informações de uma estrutura de dados para os links
        slotsInformationAux = "";

        for (int o = 0; o < this.networkOpticalLinks.length; o++) {
            for (int d = 0; d < this.networkOpticalLinks.length; d++) {
                OpticalLink optLink = this.networkOpticalLinks[o][d];
                if (optLink != null) {
                    for (int s = 0; s < numOfSlots; s++) {
                        if (optLink.isAvailableSlotAt(s)){
                            slotsInformationAux += "0;";
                        } else {
                            slotsInformationAux += "1;";
                        }
                    }
                }
            }
        }
        final String slotsInformation = slotsInformationAux.substring(0, slotsInformationAux.length() - 1);

        String data = this.reqID + ";" + sourceNodeID +";" + destinationNodeID + ";" + slotsInformation + "\n";

        this.writeFile("data_STATE.csv", data);
    }

    public void saveDataBinListsFFMSCL(int reqID2, int sourceNodeID, int destinationNodeID,
            List<Integer> saveSlotIndexes, List<Integer> saveCapacityLoss, List<Integer> saveCurrentSlotReqs,
            List<Integer> saveAlgTag, List<Float> saveCapacityPercent, List<Integer> saveCapacityDiff) {

            final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();
    
            try {

                // Armazena os meta dados
                FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "metaData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos = new DataOutputStream(fos);            

                for (int i = 0; i < saveSlotIndexes.size(); i++){
    
                    // Armazena os meta dados
                    dos.writeInt(this.indexDF);
                    dos.writeInt(reqID2);
                    dos.write(sourceNodeID);
                    dos.write(destinationNodeID);
                    
                    dos.writeInt(saveSlotIndexes.get(i));
                    dos.writeInt(saveCapacityLoss.get(i));
                    dos.write(saveCurrentSlotReqs.get(i));
                    dos.write(saveAlgTag.get(i));
                    dos.writeFloat(saveCapacityPercent.get(i));
                    dos.writeInt(saveCapacityDiff.get(i));
                }

                // Fechar o DataOutputStream e o FileOutputStream
                dos.close();
                fos.close();

                // Armazena o estado da rede a cada requisição
                FileOutputStream fos2 = new FileOutputStream(this.folderPath + "/" + "binStateData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos2 = new DataOutputStream(fos2); 

                for (int o = 0; o < this.networkOpticalLinks.length; o++) {
                    for (int d = 0; d < this.networkOpticalLinks.length; d++) {
                        OpticalLink optLink = this.networkOpticalLinks[o][d];
                        if (optLink != null) {
                            // Variáveis para armazenar os bits compactados
                            byte[] compactedData = new byte[40]; // 320 bits = 40 bytes
                            int compactedDataIndex = 0;
                            byte currentByte = 0;
                            int remainingBits = 8; // Bits restantes no byte atual

                            for (int s = 0; s < numOfSlots; s++) {
                                
                                // Definir o valor do bit (0 ou 1)
                                int bit;
                                if (optLink.isAvailableSlotAt(s)){
                                    bit = 0;
                                } else {
                                    bit = 1;
                                }
                                
                                // Definir o bit no byte atual
                                currentByte |= (bit << (remainingBits - 1));

                                // Decrementar o contador de bits restantes
                                remainingBits--;
                                
                                // Verificar se o byte está completo
                                if (remainingBits == 0) {
                                    // Adicionar o byte compactado ao array
                                    compactedData[compactedDataIndex] = currentByte;

                                    // Reiniciar as variáveis para o próximo byte
                                    compactedDataIndex++;
                                    currentByte = 0;
                                    remainingBits = 8;
                                }
                            }

                            // Verificar se ainda há bits restantes no último byte
                            if (remainingBits < 8) {
                                compactedData[compactedDataIndex] = currentByte;
                            }

                            // Escrever os bytes compactados no arquivo
                            fos2.write(compactedData);
                        }
                    }
                }
                
                indexDF++;

                // Fechar o DataOutputStream e o FileOutputStream
                dos2.close();
                fos2.close();
            } catch (IOException e) {
                System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
            }

    }

    public void saveSimulationInfo(int nSim, double networkLoad) {

        this.indexDF = 0;
        this.nSim = nSim;
        this.networkLoad = networkLoad;
    }

    public void saveDataBinListsCL(int reqID2, List<Integer> saveSlotIndexes, List<Integer> saveCapacityLoss,
            List<Integer> saveCurrentSlotReqs, List<Integer> saveBestSlotIndexes, List<Integer> saveBestCapacityLoss) {

            final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();
    
            try {

                // Armazena os meta dados
                FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "metaData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos = new DataOutputStream(fos);            

                for (int i = 0; i < saveSlotIndexes.size(); i++){
    
                    // Armazena os meta dados
                    dos.writeInt(this.indexDF);
                    dos.writeInt(reqID2);

                    dos.writeInt(saveSlotIndexes.get(i));
                    dos.writeInt(saveCapacityLoss.get(i));
                    dos.write(saveCurrentSlotReqs.get(i));
                    dos.writeInt(saveBestSlotIndexes.get(i));
                    dos.writeInt(saveBestCapacityLoss.get(i));
                }

                // Fechar o DataOutputStream e o FileOutputStream
                dos.close();
                fos.close();

                // Armazena o estado da rede a cada requisição
                FileOutputStream fos2 = new FileOutputStream(this.folderPath + "/" + "binStateData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos2 = new DataOutputStream(fos2); 

                for (int o = 0; o < this.networkOpticalLinks.length; o++) {
                    for (int d = 0; d < this.networkOpticalLinks.length; d++) {
                        OpticalLink optLink = this.networkOpticalLinks[o][d];
                        if (optLink != null) {
                            // Variáveis para armazenar os bits compactados
                            byte[] compactedData = new byte[40]; // 320 bits = 40 bytes
                            int compactedDataIndex = 0;
                            byte currentByte = 0;
                            int remainingBits = 8; // Bits restantes no byte atual

                            for (int s = 0; s < numOfSlots; s++) {
                                
                                // Definir o valor do bit (0 ou 1)
                                int bit;
                                if (optLink.isAvailableSlotAt(s)){
                                    bit = 0;
                                } else {
                                    bit = 1;
                                }
                                
                                // Definir o bit no byte atual
                                currentByte |= (bit << (remainingBits - 1));

                                // Decrementar o contador de bits restantes
                                remainingBits--;
                                
                                // Verificar se o byte está completo
                                if (remainingBits == 0) {
                                    // Adicionar o byte compactado ao array
                                    compactedData[compactedDataIndex] = currentByte;

                                    // Reiniciar as variáveis para o próximo byte
                                    compactedDataIndex++;
                                    currentByte = 0;
                                    remainingBits = 8;
                                }
                            }

                            // Verificar se ainda há bits restantes no último byte
                            if (remainingBits < 8) {
                                compactedData[compactedDataIndex] = currentByte;
                            }

                            // Escrever os bytes compactados no arquivo
                            fos2.write(compactedData);
                        }
                    }
                }
                
                indexDF++;

                // Fechar o DataOutputStream e o FileOutputStream
                dos2.close();
                fos2.close();
            } catch (IOException e) {
                System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
            }

    }
    

    public void saveDataBinListsCLiRoutes2(int reqID2, int bestIndexSlot, int bestCapacityLoss, int tag,
            int cpBefore_i1, int cpBefore_i2, int cpBefore_i3, int bestcpBefore, int cpAfter_FF_i1, int cpAfter_FF_i2,
            int cpAfter_FF_i3, int bestcpAfter_FF, int reqNumbOfSlots, Route currentRoute) throws Exception {

            final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();
    
            try {

                // Armazena os meta dados
                FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "metaData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos = new DataOutputStream(fos);            

                // Armazena os meta dados
                dos.writeInt(this.indexDF);
                dos.writeInt(reqID);
                dos.writeInt(bestIndexSlot);
                dos.writeInt(bestCapacityLoss);
                dos.write(tag);

                dos.writeInt(cpBefore_i1);
                dos.writeInt(cpBefore_i2);
                dos.writeInt(cpBefore_i3);

                dos.writeInt(bestcpBefore);

                dos.writeInt(cpAfter_FF_i1);
                dos.writeInt(cpAfter_FF_i2);
                dos.writeInt(cpAfter_FF_i3);

                dos.writeInt(bestcpAfter_FF);

                dos.write(reqNumbOfSlots);

                dos.write(currentRoute.getOriginNone());


                // Fechar o DataOutputStream e o FileOutputStream
                dos.close();
                fos.close();

                // Armazena o estado da rede a cada requisição
                FileOutputStream fos2 = new FileOutputStream(this.folderPath + "/" + "binStateData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos2 = new DataOutputStream(fos2); 

                // Seleciona os 8 links da rede
                //List<OpticalLink> optLinksIRoute = new ArrayList<OpticalLink>();

                //OpticalLink optLink = this.networkOpticalLinks[0][1];






                for (int o = 0; o < this.networkOpticalLinks.length; o++) {
                    for (int d = 0; d < this.networkOpticalLinks.length; d++) {
                        OpticalLink optLink = this.networkOpticalLinks[o][d];
                        if ((optLink != null) && (optLink.getOpticalLinkID() % 2 == 1)) {

                            int number_of_bytes = numOfSlots / 8;

                            if (number_of_bytes != ((double)numOfSlots / 8)){
                                throw new Exception("Erro ao compactar os dados");
                            }

                            // Variáveis para armazenar os bits compactados
                            byte[] compactedData = new byte[number_of_bytes]; // 320 bits = 40 bytes
                            int compactedDataIndex = 0;
                            byte currentByte = 0;
                            int remainingBits = 8; // Bits restantes no byte atual

                            for (int s = 0; s < numOfSlots; s++) {
                                
                                // Definir o valor do bit (0 ou 1)
                                int bit;
                                if (optLink.isAvailableSlotAt(s)){
                                    bit = 0;
                                } else {
                                    bit = 1;
                                }
                                
                                // Definir o bit no byte atual
                                currentByte |= (bit << (remainingBits - 1));

                                // Decrementar o contador de bits restantes
                                remainingBits--;
                                
                                // Verificar se o byte está completo
                                if (remainingBits == 0) {
                                    // Adicionar o byte compactado ao array
                                    compactedData[compactedDataIndex] = currentByte;

                                    // Reiniciar as variáveis para o próximo byte
                                    compactedDataIndex++;
                                    currentByte = 0;
                                    remainingBits = 8;
                                }
                            }

                            // Verificar se ainda há bits restantes no último byte
                            if (remainingBits < 8) {
                                compactedData[compactedDataIndex] = currentByte;
                            }

                            // Escrever os bytes compactados no arquivo
                            fos2.write(compactedData);
                        }
                    }
                }
                
                indexDF++;

                // Fechar o DataOutputStream e o FileOutputStream
                dos2.close();
                fos2.close();
            } catch (IOException e) {
                System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
            }

    }
    public void saveDataBinListsCLiRoutes(int reqID, int slotIndex, int capacityLoss, int currentSlotReq, int routeID, int tag) throws Exception {

            final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();
    
            try {

                // Armazena os meta dados
                FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "metaData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos = new DataOutputStream(fos);            

                // Armazena os meta dados
                dos.writeInt(this.indexDF);
                dos.writeInt(reqID);
                dos.writeInt(slotIndex);
                dos.writeInt(capacityLoss);
                dos.write(currentSlotReq);
                dos.write(routeID);
                dos.write(tag);

                // Fechar o DataOutputStream e o FileOutputStream
                dos.close();
                fos.close();

                // Armazena o estado da rede a cada requisição
                FileOutputStream fos2 = new FileOutputStream(this.folderPath + "/" + "binStateData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos2 = new DataOutputStream(fos2); 

                // Seleciona os 8 links da rede
                //List<OpticalLink> optLinksIRoute = new ArrayList<OpticalLink>();

                //OpticalLink optLink = this.networkOpticalLinks[0][1];






                for (int o = 0; o < this.networkOpticalLinks.length; o++) {
                    for (int d = 0; d < this.networkOpticalLinks.length; d++) {
                        OpticalLink optLink = this.networkOpticalLinks[o][d];
                        if ((optLink != null) && (optLink.getOpticalLinkID() % 2 == 1)) {

                            int number_of_bytes = numOfSlots / 8;

                            if (number_of_bytes != ((double)numOfSlots / 8)){
                                throw new Exception("Erro ao compactar os dados");
                            }

                            // Variáveis para armazenar os bits compactados
                            byte[] compactedData = new byte[number_of_bytes]; // 320 bits = 40 bytes
                            int compactedDataIndex = 0;
                            byte currentByte = 0;
                            int remainingBits = 8; // Bits restantes no byte atual

                            for (int s = 0; s < numOfSlots; s++) {
                                
                                // Definir o valor do bit (0 ou 1)
                                int bit;
                                if (optLink.isAvailableSlotAt(s)){
                                    bit = 0;
                                } else {
                                    bit = 1;
                                }
                                
                                // Definir o bit no byte atual
                                currentByte |= (bit << (remainingBits - 1));

                                // Decrementar o contador de bits restantes
                                remainingBits--;
                                
                                // Verificar se o byte está completo
                                if (remainingBits == 0) {
                                    // Adicionar o byte compactado ao array
                                    compactedData[compactedDataIndex] = currentByte;

                                    // Reiniciar as variáveis para o próximo byte
                                    compactedDataIndex++;
                                    currentByte = 0;
                                    remainingBits = 8;
                                }
                            }

                            // Verificar se ainda há bits restantes no último byte
                            if (remainingBits < 8) {
                                compactedData[compactedDataIndex] = currentByte;
                            }

                            // Escrever os bytes compactados no arquivo
                            fos2.write(compactedData);
                        }
                    }
                }
                
                indexDF++;

                // Fechar o DataOutputStream e o FileOutputStream
                dos2.close();
                fos2.close();
            } catch (IOException e) {
                System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
            }

    }

    public void saveDataBinListsCLiRoutes_OnlyMain(int reqID, int slotIndex, int capacityLoss, int currentSlotReq, Route cRoute, int tag) throws Exception {

            final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();
    
            try {

                // Armazena os meta dados
                FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "metaData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos = new DataOutputStream(fos);            

                // Armazena os meta dados
                dos.writeInt(this.indexDF);
                dos.writeInt(reqID);
                dos.writeInt(slotIndex);
                dos.writeInt(capacityLoss);
                dos.write(currentSlotReq);
                dos.write(cRoute.getOriginNone());
                dos.write(tag);

                // Fechar o DataOutputStream e o FileOutputStream
                dos.close();
                fos.close();

                // Armazena o estado da rede a cada requisição
                FileOutputStream fos2 = new FileOutputStream(this.folderPath + "/" + "binStateData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos2 = new DataOutputStream(fos2); 

                // Vetor de disponibilidade de slots da rota
                short[] routeSlots = cRoute.getSlotOcupationLink();

                boolean[] routeSlotsBool = new boolean[numOfSlots];
                for (int i = 0; i < numOfSlots; i++) {
                    if (routeSlots[i] == 0){
                        routeSlotsBool[i] = false;
                    } else {
                        routeSlotsBool[i] = true;
                    }
                }
        
                int number_of_bytes = numOfSlots / 8;

                if (number_of_bytes != ((double)numOfSlots / 8)){
                    throw new Exception("Erro ao compactar os dados");
                }

                // Variáveis para armazenar os bits compactados
                byte[] compactedData = new byte[number_of_bytes]; // 320 bits = 40 bytes
                int compactedDataIndex = 0;
                byte currentByte = 0;
                int remainingBits = 8; // Bits restantes no byte atual

                for (int s = 0; s < numOfSlots; s++) {
                                
                    // Definir o valor do bit (0 ou 1)
                    int bit;
                    if (routeSlots[s] == 0){
                        bit = 0;
                    } else {
                        bit = 1;
                    }
                                
                    // Definir o bit no byte atual
                    currentByte |= (bit << (remainingBits - 1));

                    // Decrementar o contador de bits restantes
                    remainingBits--;
                                
                    // Verificar se o byte está completo
                    if (remainingBits == 0) {
                        // Adicionar o byte compactado ao array
                        compactedData[compactedDataIndex] = currentByte;

                        // Reiniciar as variáveis para o próximo byte
                        compactedDataIndex++;
                        currentByte = 0;
                        remainingBits = 8;
                    }
                }

                // Verificar se ainda há bits restantes no último byte
                if (remainingBits < 8) {
                    compactedData[compactedDataIndex] = currentByte;
                }

                // Escrever os bytes compactados no arquivo
                fos2.write(compactedData);

                indexDF++;

                // Fechar o DataOutputStream e o FileOutputStream
                dos2.close();
                fos2.close();
            } catch (IOException e) {
                System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
            }


    }

    public void saveBinAllDatas_AvailableVector(int reqID2, int bestIndexSlot, int bestCapacityLoss, int tag,
            int cpBefore_i1, int cpBefore_i2, int cpBefore_i3, int bestcpBefore, int cpAfter_FF_i1, int cpAfter_FF_i2,
            int cpAfter_FF_i3, int bestcpAfter_FF, int reqNumbOfSlots, Route currentRoute) throws Exception {


            final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();
    
            try {

                // Armazena os meta dados
                FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "metaData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos = new DataOutputStream(fos);            

                // Armazena os meta dados
                dos.writeInt(this.indexDF);
                dos.writeInt(reqID);
                dos.writeInt(bestIndexSlot);
                dos.writeInt(bestCapacityLoss);
                dos.write(tag);

                dos.writeInt(cpBefore_i1);
                dos.writeInt(cpBefore_i2);
                dos.writeInt(cpBefore_i3);

                dos.writeInt(bestcpBefore);

                dos.writeInt(cpAfter_FF_i1);
                dos.writeInt(cpAfter_FF_i2);
                dos.writeInt(cpAfter_FF_i3);

                dos.writeInt(bestcpAfter_FF);

                dos.write(reqNumbOfSlots);

                dos.write(currentRoute.getOriginNone());

                // Fechar o DataOutputStream e o FileOutputStream
                dos.close();
                fos.close();

                // Armazena o estado da rede a cada requisição
                FileOutputStream fos2 = new FileOutputStream(this.folderPath + "/" + "binStateData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos2 = new DataOutputStream(fos2); 

                // Vetor de disponibilidade de slots da rota
                short[] routeSlots = currentRoute.getSlotOcupationLink();

                boolean[] routeSlotsBool = new boolean[numOfSlots * (currentRoute.getAllConflictRoutes().size() + 1)];

                for (int i = 0; i < numOfSlots; i++) {
                    if (routeSlots[i] == 0){
                        routeSlotsBool[i] = false;
                    } else {
                        routeSlotsBool[i] = true;
                    }
                }

                for (int i = 0; i < currentRoute.getAllConflictRoutes().size(); i++) {
                    short[] routeSlotsAux = currentRoute.getAllConflictRoutes().get(i).getSlotOcupationLink();
                    for (int j = 0; j < numOfSlots; j++) {
                        if (routeSlotsAux[j] == 0){
                            routeSlotsBool[j + (numOfSlots * (i + 1))] = false;
                        } else {
                            routeSlotsBool[j + (numOfSlots * (i + 1))] = true;
                        }
                    }
                }
        
                int number_of_bytes = (numOfSlots * 3) / 8;

                if (number_of_bytes != ((double)(numOfSlots * 3) / 8)){
                    throw new Exception("Erro ao compactar os dados");
                }

                // Variáveis para armazenar os bits compactados
                byte[] compactedData = new byte[number_of_bytes]; // 320 bits = 40 bytes
                int compactedDataIndex = 0;
                byte currentByte = 0;
                int remainingBits = 8; // Bits restantes no byte atual

                for (int s = 0; s < numOfSlots; s++) {
                                
                    // Definir o valor do bit (0 ou 1)
                    int bit;
                    if (routeSlots[s] == 0){
                        bit = 0;
                    } else {
                        bit = 1;
                    }
                                
                    // Definir o bit no byte atual
                    currentByte |= (bit << (remainingBits - 1));

                    // Decrementar o contador de bits restantes
                    remainingBits--;
                                
                    // Verificar se o byte está completo
                    if (remainingBits == 0) {
                        // Adicionar o byte compactado ao array
                        compactedData[compactedDataIndex] = currentByte;

                        // Reiniciar as variáveis para o próximo byte
                        compactedDataIndex++;
                        currentByte = 0;
                        remainingBits = 8;
                    }
                }

                // Verificar se ainda há bits restantes no último byte
                if (remainingBits < 8) {
                    compactedData[compactedDataIndex] = currentByte;
                }

                // Escrever os bytes compactados no arquivo
                fos2.write(compactedData);

                indexDF++;

                // Fechar o DataOutputStream e o FileOutputStream
                dos2.close();
                fos2.close();
            } catch (IOException e) {
                System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
            }



    }

    public void saveDataBinListsCLiRoutes3(int reqID2, int bestIndexSlot, int bestCapacityLoss, int tag,
            int cpBefore_i1, int cpBefore_i2, int cpBefore_i3, int bestcpBefore, int cpAfter_FF_i1, int cpAfter_FF_i2,
            int cpAfter_FF_i3, int bestcpAfter_FF, int reqNumbOfSlots, Route currentRoute) throws Exception {
                /*Coleta os dados e as informação da rota principal e das rotas interferentes */

            final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();
    
            try {

                // Armazena os meta dados
                FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "metaData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos = new DataOutputStream(fos);            

                // Armazena os meta dados
                dos.writeInt(this.indexDF);
                dos.writeInt(reqID);
                dos.writeInt(bestIndexSlot);
                dos.writeInt(bestCapacityLoss);
                dos.write(tag);

                dos.writeInt(cpBefore_i1);
                dos.writeInt(cpBefore_i2);
                dos.writeInt(cpBefore_i3);

                dos.writeInt(bestcpBefore);

                dos.writeInt(cpAfter_FF_i1);
                dos.writeInt(cpAfter_FF_i2);
                dos.writeInt(cpAfter_FF_i3);

                dos.writeInt(bestcpAfter_FF);

                dos.write(reqNumbOfSlots);

                dos.write(currentRoute.getOriginNone());


                // Fechar o DataOutputStream e o FileOutputStream
                dos.close();
                fos.close();


                // Armazena o estado da rede a cada requisição
                FileOutputStream fos2 = new FileOutputStream(this.folderPath + "/" + "binStateData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

                DataOutputStream dos2 = new DataOutputStream(fos2); 

                // Vetor de disponibilidade de slots da rota
                short[] routeSlots = currentRoute.getSlotOcupationLink();
                short[] routeSlots_i1 = currentRoute.getAllConflictRoutes().get(0).getSlotOcupationLink();
                short[] routeSlots_i2 = currentRoute.getAllConflictRoutes().get(1).getSlotOcupationLink();

                boolean[] routeSlotsBool = new boolean[numOfSlots * 3];
                for (int i = 0; i < numOfSlots * 3; i++) {
                    if (i < numOfSlots){
                        if (routeSlots[i] == 0){
                            routeSlotsBool[i] = false;
                        } else {
                            routeSlotsBool[i] = true;
                        }
                    } else if (i < numOfSlots * 2){
                        if (routeSlots_i1[i - numOfSlots] == 0){
                            routeSlotsBool[i] = false;
                        } else {
                            routeSlotsBool[i] = true;
                        }
                    } else {
                        if (routeSlots_i2[i - (numOfSlots * 2)] == 0){
                            routeSlotsBool[i] = false;
                        } else {
                            routeSlotsBool[i] = true;
                        }
                    }
                }
        
                int number_of_bytes = (numOfSlots * 3) / 8;

                if (number_of_bytes != ((double)(numOfSlots * 3) / 8)){
                    throw new Exception("Erro ao compactar os dados");
                }

                // Variáveis para armazenar os bits compactados
                byte[] compactedData = new byte[number_of_bytes]; // 320 bits = 40 bytes
                int compactedDataIndex = 0;
                byte currentByte = 0;
                int remainingBits = 8; // Bits restantes no byte atual

                for (int s = 0; s < (numOfSlots * 3); s++) {
                                
                    // Definir o valor do bit (0 ou 1)
                    int bit;
                    if (routeSlotsBool[s] == false){
                        bit = 0;
                    } else {
                        bit = 1;
                    }
                                
                    // Definir o bit no byte atual
                    currentByte |= (bit << (remainingBits - 1));

                    // Decrementar o contador de bits restantes
                    remainingBits--;
                                
                    // Verificar se o byte está completo
                    if (remainingBits == 0) {
                        // Adicionar o byte compactado ao array
                        compactedData[compactedDataIndex] = currentByte;

                        // Reiniciar as variáveis para o próximo byte
                        compactedDataIndex++;
                        currentByte = 0;
                        remainingBits = 8;
                    }
                }

                // Verificar se ainda há bits restantes no último byte
                if (remainingBits < 8) {
                    compactedData[compactedDataIndex] = currentByte;
                }

                // Escrever os bytes compactados no arquivo
                fos2.write(compactedData);

                indexDF++;

                // Fechar o DataOutputStream e o FileOutputStream
                dos2.close();
                fos2.close();
                
            } catch (IOException e) {
                System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
            }








    }

    public void saveDataBinListsMetricsLists(int reqID, int reqNumbOfSlots, Route route, int[] mSCL_return) throws Exception {

        // Obtem o tamanho do espectro
        final int numOfSlots = ParametersSimulation.getNumberOfSlotsPerLink();
    
        try {
            // * Cria o arquivo de META DADOS

            // Armazena os meta dados
            FileOutputStream fos = new FileOutputStream(this.folderPath + "/" + "metaData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

            DataOutputStream dos = new DataOutputStream(fos);            

            // Armazena os meta dados
            dos.writeInt(this.indexDF);
            dos.writeInt(mSCL_return[0]); // bestIndexSlot
            dos.writeInt(mSCL_return[1]); // bestCapacityLoss
            dos.writeInt(mSCL_return[2]); // tag
            dos.writeInt(reqID); // reqID
            dos.writeInt(reqNumbOfSlots); // reqNumbOfSlots
            dos.writeInt(route.getOriginNone()); // routeID

            dos.writeInt(mSCL_return[3]); // cpBefore_r1
            dos.writeInt(mSCL_return[4]); // cpBefore_r2
            dos.writeInt(mSCL_return[5]); // cpBefore_r3

            dos.writeInt(mSCL_return[6]); // cp_FF_After_r1
            dos.writeInt(mSCL_return[7]); // cp_FF_After_r2
            dos.writeInt(mSCL_return[8]); // cp_FF_After_r3

            dos.writeInt(mSCL_return[9]); // cp_LF_After_r1
            dos.writeInt(mSCL_return[10]); // cp_LF_After_r2
            dos.writeInt(mSCL_return[11]); // cp_LF_After_r3

            dos.writeInt(mSCL_return[12]); // ocupationBefore_r1
            dos.writeInt(mSCL_return[13]); // ocupationBefore_r2
            dos.writeInt(mSCL_return[14]); // ocupationBefore_r3

            dos.writeInt(mSCL_return[15]); // slotFF
            dos.writeInt(mSCL_return[16]); // slotLF

            // Fechar o DataOutputStream e o FileOutputStream
            dos.close();
            fos.close();

            // * Cria o arquivo de ESTADO dos vetores de disponibilidade de slots

            // Armazena o estado da rede a cada requisição
            FileOutputStream fos2 = new FileOutputStream(this.folderPath + "/" + "binStateData_" + this.nSim + "_" + this.networkLoad + ".bin", true);

            DataOutputStream dos2 = new DataOutputStream(fos2); 

            // Vetor de disponibilidade de slots da rota
            short[] routeSlots = route.getSlotOcupationLink();
            short[] routeSlots_i1 = route.getAllConflictRoutes().get(0).getSlotOcupationLink();
            short[] routeSlots_i2 = route.getAllConflictRoutes().get(1).getSlotOcupationLink();

            boolean[] routeSlotsBool = new boolean[numOfSlots * 3];
            for (int i = 0; i < numOfSlots * 3; i++) {
                if (i < numOfSlots){
                    if (routeSlots[i] == 0){
                        routeSlotsBool[i] = false;
                    } else {
                        routeSlotsBool[i] = true;
                    }
                } else if (i < numOfSlots * 2){
                    if (routeSlots_i1[i - numOfSlots] == 0){
                        routeSlotsBool[i] = false;
                    } else {
                        routeSlotsBool[i] = true;
                    }
                } else {
                    if (routeSlots_i2[i - (numOfSlots * 2)] == 0){
                        routeSlotsBool[i] = false;
                    } else {
                        routeSlotsBool[i] = true;
                    }
                }
            }
    
            int number_of_bytes = (numOfSlots * 3) / 8;

            if (number_of_bytes != ((double)(numOfSlots * 3) / 8)){
                throw new Exception("Erro ao compactar os dados");
            }

            // Variáveis para armazenar os bits compactados
            byte[] compactedData = new byte[number_of_bytes]; // 320 bits = 40 bytes
            int compactedDataIndex = 0;
            byte currentByte = 0;
            int remainingBits = 8; // Bits restantes no byte atual

            for (int s = 0; s < (numOfSlots * 3); s++) {
                            
                // Definir o valor do bit (0 ou 1)
                int bit;
                if (routeSlotsBool[s] == false){
                    bit = 0;
                } else {
                    bit = 1;
                }
                            
                // Definir o bit no byte atual
                currentByte |= (bit << (remainingBits - 1));

                // Decrementar o contador de bits restantes
                remainingBits--;
                            
                // Verificar se o byte está completo
                if (remainingBits == 0) {
                    // Adicionar o byte compactado ao array
                    compactedData[compactedDataIndex] = currentByte;

                    // Reiniciar as variáveis para o próximo byte
                    compactedDataIndex++;
                    currentByte = 0;
                    remainingBits = 8;
                }
            }

            // Verificar se ainda há bits restantes no último byte
            if (remainingBits < 8) {
                compactedData[compactedDataIndex] = currentByte;
            }

            // Escrever os bytes compactados no arquivo
            fos2.write(compactedData);

            indexDF++;

            // Fechar o DataOutputStream e o FileOutputStream
            dos2.close();
            fos2.close();
            
        } catch (IOException e) {
            System.out.println("Erro ao criar ou gravar o arquivo: " + e.getMessage());
        }
    }
}
