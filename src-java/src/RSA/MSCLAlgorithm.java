package RSA;

import java.util.ArrayList;
import java.util.List;

import CallRequests.CallRequest;
import Config.ParametersSimulation;
import Manager.FolderManager;
import RSA.Routing.Route;

public class MSCLAlgorithm {
    
    /**
     * Rota selecionada pelo algoritmo de roteamento
     */
    private Route route;
    /**
     * Conjunto de slots selecionados pelo algoritmo de alocação do espectro
     */
    private List<Integer> fSlots;
    /**
     * Armazena o valor do ciclo MSCL
     */
    private long cyclesMSCL;
    /**
     * Armazena o slots encontrado para cada rota
     */
    List<List<Integer>> slotsMSCL;

    private FolderManager folderManager;

    public MSCLAlgorithm(FolderManager folderManager){
        this.route = null;
        this.fSlots = new ArrayList<Integer>();
        this.slotsMSCL = new ArrayList<List<Integer>>();
        this.cyclesMSCL = 0;

        this.folderManager = folderManager;
    }

    
    public boolean findMSCLSequencial(List<Route> routeSolution, CallRequest callRequest, List<Route> listAllRoutes) throws Exception {

        double valuesLostCapacity = Double.MAX_VALUE;
        int bestIndexMSCL = 0;

        for (Route currentRoute : routeSolution){

            if (currentRoute != null){
                valuesLostCapacity = findSlotsAndCapacityLoss(currentRoute, callRequest, listAllRoutes);
            }

            if (valuesLostCapacity < (Double.MAX_VALUE * 0.7)){
                this.route = routeSolution.get(bestIndexMSCL);
                this.fSlots = this.slotsMSCL.get(bestIndexMSCL);

                // Calcula o tamanho da requisição
                int reqNumbOfSlots = this.route.getReqSize(callRequest.getSelectedBitRate());

                callRequest.setReqNumbOfSlots(reqNumbOfSlots);

                return true;
            }
            
            bestIndexMSCL++;
        }
        
        this.route = null;
        this.fSlots = new ArrayList<Integer>();
        return false;
    }

    public boolean findMSCLCombinado(List<Route> routeSolution, CallRequest callRequest) throws Exception {

        List<Double> valuesLostCapacity = new ArrayList<Double>();
        int bestIndexMSCL = 0;
        int selectKRouteID = -1;

        for (Route currentRoute : routeSolution){

            if (currentRoute != null){
                //valuesLostCapacity.add(getRouteMSCLCost(currentRoute, callRequest));
                valuesLostCapacity.add(findSlotsAndCapacityLoss(currentRoute, callRequest, routeSolution));
            } else {
                valuesLostCapacity.add(Double.MAX_VALUE * 0.9);
            }
        }

        double minValue = Double.MAX_VALUE * 0.7;

        for (int index = 0; index < routeSolution.size(); index++){

            if (minValue > valuesLostCapacity.get(index)){ // Se for igual, escolher a menor rota

                minValue = valuesLostCapacity.get(index);
                bestIndexMSCL = index;
                selectKRouteID = routeSolution.get(bestIndexMSCL).getkFindIndex();

            } else {

                if ((minValue == valuesLostCapacity.get(index)) && selectKRouteID > routeSolution.get(index).getkFindIndex()) { // Se for igual, escolher a menor rota
                    
                    minValue = valuesLostCapacity.get(index);
                    bestIndexMSCL = index;
                    selectKRouteID = routeSolution.get(bestIndexMSCL).getkFindIndex();
                }
            }
        }

        if (minValue < Double.MAX_VALUE * 0.5){
            this.route = routeSolution.get(bestIndexMSCL);
            this.fSlots = this.slotsMSCL.get(bestIndexMSCL);

            // Calcula o tamanho da requisição
            int reqNumbOfSlots = this.route.getReqSize(callRequest.getSelectedBitRate());
            callRequest.setReqNumbOfSlots(reqNumbOfSlots);
            return true;
        }

        return false;
    }

    /*
     * Essa função calcula a perda de capacidade para a rota principal e para as rotas interferentes em busca do melhor slot para alocar a requisição.
     * 
     * @param currentRoute Rota principal
     * @param callRequest Requisição de chamada
     * 
     * @return Retorna a perda de capacidade da melhor alocação para a requisição.
     */
    private double findSlotsAndCapacityLoss(Route currentRoute, CallRequest callRequest, List<Route> routeSolution) throws Exception {
        
        // Armazena o número máximo de slots por enlace
        final int numberMaxSlotsPerLink = ParametersSimulation.getNumberOfSlotsPerLink();
        
        // Calcula o tamanho da requisição de acordo com a taxa de transmissão
        final int reqNumbOfSlots = currentRoute.getReqSize(callRequest.getSelectedBitRate());
        
        // Armazena o nó de origem e destino da rota principal
        
        // Armazena o ID da requisição para fins de armazenamento dos dados
        final int reqID = this.folderManager.getReqID();
        //System.out.println("\n\n*** Req = " + reqID);
        
        // Armazena todos os possíveis tamanhos de requisição para cada valor de bitrate para a maior modulação possível nessa rota
        final int[] possibleSlotsByRoute = currentRoute.getAllReqSizes();

        if (reqID == 10001){
            System.out.println("Req = " + reqID);
        }

        List<MSCLApeture> allAperturesInMainRoute;
        boolean isPossibleToAlocateReq;
        int[] MSCL_return;
        int bestIndexSlot;
        int bestCapacityLoss;
        int tag;
        // Percorre todas as rotas em uma mesma requisição para aumentar a variedade dos dados
        // for (Route route: routeSolution){
        //     // Percorre todas as demandas da rota para aumentar a variedade dos dados 
        //     DFOR:for (int demand: possibleSlotsByRoute){
        //         // Ignora a rota principal da requisição quando junto a demanda requisitada para deixar ela por último
        //         if ((route == currentRoute) && (demand == reqNumbOfSlots)){
        //             continue DFOR;
        //         }

        //         // Encontra as lacunas para a rota ao longo de todo o espectro
        //         allAperturesInMainRoute = MSCLApeture.genApetures(route, 0, numberMaxSlotsPerLink - 1);

        //         // Verifica se é possível alocar essa requisição dentro da rota
        //         isPossibleToAlocateReq = false;
        //         for (int index = 0; index < allAperturesInMainRoute.size(); index++){
        //             if (allAperturesInMainRoute.get(index).getSize() >= demand){
        //                 isPossibleToAlocateReq = true;
        //             }
        //         }

        //         //Se não há recursos disponíveis para alocar a requisição então continua
        //         if (!isPossibleToAlocateReq){
        //             continue; 
        //         }

        //         // Executa o MSCL para a rota e demanda e encontra os dados para armazenar
        //         MSCL_return = this.runMSCL(route, allAperturesInMainRoute, possibleSlotsByRoute, demand, numberMaxSlotsPerLink);

        //         // Armazena os dados para a rota e demanda
        //         if ((reqID > 10000) && (reqID % 4 == 0)){ // Se já estiver na zona de estabilidade

        //             // Função para armazenar os dados e o vetor de disponibilidade de slots da rota principal e das rotas interferentes

        //             //this.folderManager.saveDataBinListsMetricsLists(reqID, demand, route, MSCL_return); // Armazena o espectro da rota principal e das rotas interferentes
        //         }
        //     }
        // }
        
        // *** Executa o MSCL para a rota principal e encontra os dados para armazenar
    
        // Encontra a lista de lacunas para a rota principal ao longo de todo o espectro
        allAperturesInMainRoute = MSCLApeture.genApetures(currentRoute, 0, numberMaxSlotsPerLink - 1);

        // Verifica se é possível alocar essa requisição dentro da rota principal
        isPossibleToAlocateReq = false;

        for (int index = 0; index < allAperturesInMainRoute.size(); index++){
            if (allAperturesInMainRoute.get(index).getSize() >= reqNumbOfSlots){
                isPossibleToAlocateReq = true;
            }
        }
        
        //Se não há recursos disponíveis para alocar a requisição então retorna um valor alto de perda de capacidade
        if (!isPossibleToAlocateReq){
            // Armazena uma estrutura vazia para a requisição
            this.slotsMSCL.add(new ArrayList<Integer>());
            
            return Double.MAX_VALUE * 0.8; 
        }

        // Executa o MSCL para a rota e demanda e encontra os dados para armazenar
        MSCL_return = this.runMSCL(currentRoute, allAperturesInMainRoute, possibleSlotsByRoute, reqNumbOfSlots, numberMaxSlotsPerLink);

        bestIndexSlot = MSCL_return[0];
        bestCapacityLoss = MSCL_return[1];

        // Armazena os dados para a rota e demanda
        if ((reqID > 10000) && (reqID % 4 == 0)) { // Se já estiver na zona de estabilidade

            //this.folderManager.saveDataBinListsMetricsLists(reqID, reqNumbOfSlots, currentRoute, MSCL_return);
        }

        // Cria uma lista com os slots necessários para alocar a requisição com a melhor perda de capacidade
        List<Integer> slotsReq = new ArrayList<Integer>();
        for (int s = bestIndexSlot; s <= bestIndexSlot + reqNumbOfSlots - 1; s++){
            slotsReq.add(s);
        }

        // Adiciona a lista de slots necessários para alocar a requisição com a melhor perda de capacidade
        this.slotsMSCL.add(slotsReq);

        return bestCapacityLoss;
    }

    private int[] runMSCL(Route route, List<MSCLApeture> allAperturesInMainRoute, int[] possibleSlotsByRoute, int reqNumbOfSlots, int numberMaxSlotsPerLink) throws Exception {
        /*
         * Excuta o MSCL para a rota principal e suas rotas interferentes e coleta os dados para armazenar. Os dados coletados são:
         * 1. O melhor slot para alocar a requisição segundo o MSCL (bestIndexSlot) : int
         * 2. A melhor perda de capacidade para alocar a requisição segundo o MSCL (bestCapacityLoss) : int
         * 3. A tag que indica qual foi o melhor slot encontrado. 0 - First Fit. 1 - MSCL. 2 - Last Fit (tag) : int
         * 4. O ID da requisição (reqID) : int
         * 5. O tamanho da requisição a ser alocada (reqNumbOfSlots) : int
         * 6. O ID da rota principal (routeID) : int
         * 7. O valor da capacidade antes da alocação, na rota principal (cpBefore) : int
         * 8. O valor da capacidade antes da alocação, na rota interferente 1 (cpBefore_i1) : int
         * 9. O valor da capacidade antes da alocação, na rota interferente 2 (cpBefore_i2) : int
         * 10. O valor da capacidade depois da alocação em FF, na rota principal (cp_FF_After_r1) : int
         * 11. O valor da capacidade depois da alocação em FF, na rota interferente 1 (cp_FF_After_r2) : int
         * 12. O valor da capacidade depois da alocação em FF, na rota interferente 2 (cp_FF_After_r3) : int
         * 13. O valor da capacidade depois da alocação em LF, na rota principal (cp_LF_After_r1) : int
         * 14. O valor da capacidade depois da alocação em LF, na rota interferente 1 (cp_LF_After_r2) : int
         * 15. O valor da capacidade depois da alocação em LF, na rota interferente 2 (cp_LF_After_r3) : int
         * 16. Ocupação da rota principal antes da alocação (ocupationBefore_r1) : int
         * 17. Ocupação da rota interferente 1 antes da alocação (ocupationBefore_r2) : int
         * 18. Ocupação da rota interferente 2 antes da alocação (ocupationBefore_r3) : int
         * 19. Slot inicial da alocação em FF (slotFF) : int
         * 20. Slot inicial da alocação em LF (slotLF) : int
        */


        // Armazena a lista com todas as rotas interferentes
        final List<Route> allConflictRoutes = route.getAllConflictRoutes();

        // Inicializa a variável que armazena o melhor slot para alocar a requisição de acordo com o MSCL
        int bestIndexSlot = -1;

        // Inicializa a variável que armazena a melhor perda de capacidade
        int bestCapacityLoss = Integer.MAX_VALUE;

        // Inicializar a variável que armazena a tag do algoritmo utilizando, 0 - First Fit, 1 - MSCL, 2 - Last Fit
        int tag = -1;

        // Lista que armazena os valores de capacidade ANTES da alocação para cada rota
        int[] cpBefore = new int[allConflictRoutes.size() + 1];

        // Lista que armazena os valores de capacidade DEPOIS da alocação para cada rota utilizando o First Fit
        int[] cp_FF_After = new int[allConflictRoutes.size() + 1];

        // Lista que armazena os valores de capacidade DEPOIS da alocação para cada rota utilizando o Last Fit
        int[] cp_LF_After = new int[allConflictRoutes.size() + 1];

        // Lista que armazena os valores de ocupação ANTES da alocação para cada rota
        int[] ocupationBefore = new int[allConflictRoutes.size() + 1];

        // Variável para armazena a posição do slot inicial da alocação em FF e LF
        int slotFF = -1;
        int slotLF = -1;

        // Armazena a perda de capacidade para o slot do First Fit e do Last Fit
        int bestFFCapacityLoss = Integer.MAX_VALUE;
        int bestLFCapacityLoss = Integer.MAX_VALUE;

        // *** Calcula a perda de capacidade ANTES da alocação - Rota principal ***
        int capacityLossBefore = 0;

        int capacityLossBeforeRoute = 0;
        // Percorre todas as lacunas da rota principal
        for (int index = 0; index < allAperturesInMainRoute.size(); index++){

            // Armazena a posição inicial e o tamanho da lacuna
            final int sizeInApertureInMainRoute = allAperturesInMainRoute.get(index).getSize();

            // * Rota principal

            for (int possibleReqSize: possibleSlotsByRoute){

                if (possibleReqSize > sizeInApertureInMainRoute){
                    break;
                }
                capacityLossBeforeRoute += (sizeInApertureInMainRoute - possibleReqSize + 1);
            }
        }
        capacityLossBefore += capacityLossBeforeRoute;
        cpBefore[0] = capacityLossBeforeRoute;
        ocupationBefore[0] = route.getOcupation();
        
        // ** Calcula a perda de capacidade ANTES da alocação - Rotas alternativas ***
        // Percorre todas as rotas interferentes
        for (int indexConflictRoute = 0; indexConflictRoute < allConflictRoutes.size(); indexConflictRoute++) {

            // Encontra as lacunas na rota interferente entre os slots 'minSlot' e 'maxSlot'
            final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(allConflictRoutes.get(indexConflictRoute), 0, numberMaxSlotsPerLink - 1);

            capacityLossBeforeRoute = 0;

            // Percorre as lacunas encontradas na rota interferente
            for (MSCLApeture apetureInConflictRoute : allApeturesAfectInConflictRoute) {

                final int sizeInApetureInConflictRoute = apetureInConflictRoute.getSize();

                for (int possibleReqSize: possibleSlotsByRoute){

                    if (possibleReqSize > sizeInApetureInConflictRoute){
                        break;
                    }
                    capacityLossBeforeRoute += (sizeInApetureInConflictRoute - possibleReqSize + 1);
                }
            }

            capacityLossBefore += capacityLossBeforeRoute;

            cpBefore[1 + indexConflictRoute] = capacityLossBeforeRoute;

            ocupationBefore[1 + indexConflictRoute] = allConflictRoutes.get(indexConflictRoute).getOcupation();
        }

        // *** Calcula a perda de capacidade DEPOIS da alocação ***
        for (int indexSlot = 0; indexSlot < numberMaxSlotsPerLink; indexSlot++) {

            int capacityLossAfter = 0;
            int totalCapacityLoss = 0;

            // Considerando uma requisição de tamanho 'reqNumbOfSlots', estabelece o slot inicial e o final
            int startSlot = indexSlot;
            int finalSlot = indexSlot + reqNumbOfSlots - 1;

            // Verifica as condições de parada para a alocação iniciando em 'slot_index'. São elas:
            // 1. As posições 'start_slot' e 'final_slot' não podem está ocupadas por outra demanda
            // 2. A posição 'final_slot' não pode ultrapassar o limite do espectro
            // 3. Todos os slots da demanda devem estar livres na rota principal
            boolean isPossibleToAlocateReq = true;
            if (finalSlot >= numberMaxSlotsPerLink){
                isPossibleToAlocateReq = false;
            } else {
                for (int slot = startSlot; slot <= finalSlot; slot++){
                    if ((route.getSlotValue(slot) != 0) || (slot >= numberMaxSlotsPerLink)){
                        isPossibleToAlocateReq = false;
                        break;
                    }
                }
            }

            if (!isPossibleToAlocateReq){
                continue;
            }

            // Chegando aqui, significa que a demanda pode ser alocada na rota principal

            // ? Atribui o valor de 'start_slot' para a variável 'best_FF_slot_index' caso ela ainda não tenha sido atribuída. Ou seja, é a primeira vez que um slot é encontrado pelo MSCL

            // Armazena o slot do First Fit
            if (slotFF == -1){
                slotFF = indexSlot;
            }

            // Cria uma lista dos slots fake necessários para alocar a requisição
            List<Integer> slotsReqFake = new ArrayList<Integer>();
            for (int s = startSlot; s <= finalSlot; s++){
                slotsReqFake.add(s);
            }

            // Realiza a alocação fake na rota principal
            route.incrementSlotsOcupy(slotsReqFake);
            
            // ** Calcula a perda de capacidade DEPOIS da alocação - Rota principal

            int capacityLossAfterRoute = 0;

            // Encontra as lacunas formadas após a alocação fake
            final List<MSCLApeture> aperturesInMainRouteAfter = MSCLApeture.genApetures(route, 0, numberMaxSlotsPerLink - 1);

            // Percorre as lacunas encontradas na rota principal
            for (MSCLApeture apetureInApetureMainRouteAfter : aperturesInMainRouteAfter) {

                final int sizeInapetureInApetureMainRouteAfter = apetureInApetureMainRouteAfter.getSize();

                for (int possibleReqSize: possibleSlotsByRoute){
                    if (possibleReqSize > sizeInapetureInApetureMainRouteAfter){
                        break;
                    }
                    capacityLossAfterRoute += (sizeInapetureInApetureMainRouteAfter - possibleReqSize + 1);
                }
            }

            capacityLossAfter += capacityLossAfterRoute;
            
            if (slotFF == indexSlot){
                cp_FF_After[0] = capacityLossAfterRoute;
            }
            cp_LF_After[0] = capacityLossAfterRoute;

            // ** Calcula a perda de capacidade DEPOIS da alocação - Rotas interferentes

            // Percorre todas as rotas interferentes
            for (int indexConflictRoute = 0; indexConflictRoute < allConflictRoutes.size(); indexConflictRoute++) {

                // Encontra as lacunas na rota interferente entre os slots 'minSlot' e 'maxSlot'
                final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(allConflictRoutes.get(indexConflictRoute), 0, numberMaxSlotsPerLink - 1);

                capacityLossAfterRoute = 0;

                // Percorre as lacunas encontradas na rota interferente
                for (MSCLApeture apetureInConflictRoute : allApeturesAfectInConflictRoute) {

                    final int sizeInApetureInConflictRoute = apetureInConflictRoute.getSize();

                    for (int possibleReqSize: possibleSlotsByRoute){

                        if (possibleReqSize > sizeInApetureInConflictRoute){
                            break;
                        }
                        capacityLossAfterRoute += (sizeInApetureInConflictRoute - possibleReqSize + 1);
                    }
                }

                capacityLossAfter += capacityLossAfterRoute;

                if (slotFF == indexSlot){
                    cp_FF_After[1 + indexConflictRoute] = capacityLossAfterRoute;
                }
                cp_LF_After[1 + indexConflictRoute] = capacityLossAfterRoute;
            }

            // Desfaz a alocação fake na rota principal
            route.decreasesSlotsOcupy(slotsReqFake);

            // Calcula a perda de capacidade total para o slot 'indexSlot'
            totalCapacityLoss = capacityLossBefore - capacityLossAfter;

            if (totalCapacityLoss < 0){
                throw new Exception("Erro ao calcular a perda de capacidade total para o slot 'indexSlot'. IndexSlot = " + indexSlot + ". totalCapacityLoss = " + totalCapacityLoss + ". ReqID = " + this.folderManager.getReqID());
            }

            if (indexSlot == slotFF){
                bestFFCapacityLoss = totalCapacityLoss;
            }
            bestLFCapacityLoss = totalCapacityLoss;

            slotLF = indexSlot;

            // Se a perda de capacidade para o slot 'indexSlot' é menor que a melhor perda de capacidade e atualiza a melhor perda de capacidade. Os casos de igualdade são tratados pegando o último slot que teve a melhor perda de capacidade, exceto quando o slot do First Fit tem a melhor perda de capacidade.
            if ((totalCapacityLoss < bestCapacityLoss) || ((totalCapacityLoss == bestCapacityLoss) && (totalCapacityLoss < bestFFCapacityLoss))){

                bestCapacityLoss = totalCapacityLoss;
                bestIndexSlot = indexSlot;
            }

        }

        tag = 1;
        if (bestIndexSlot == slotFF){
            tag = 0;
        } else if (bestIndexSlot == slotLF){
            tag = 2;
        }

        /*
         * Excuta o MSCL para a rota principal e suas rotas interferentes e coleta os dados para armazenar. Os dados coletados são:
         * 1. O melhor slot para alocar a requisição segundo o MSCL (bestIndexSlot) : int
         * 2. A melhor perda de capacidade para alocar a requisição segundo o MSCL (bestCapacityLoss) : int
         * 3. A tag que indica qual foi o melhor slot encontrado. 0 - First Fit. 1 - MSCL. 2 - Last Fit (tag) : int
         * 4. O ID da requisição (reqID) : int
         * 5. O tamanho da requisição a ser alocada (reqNumbOfSlots) : int
         * 6. O ID da rota principal (routeID) : int
         * 7. O valor da capacidade antes da alocação, na rota principal (cpBefore) : int
         * 8. O valor da capacidade antes da alocação, na rota interferente 1 (cpBefore_i1) : int
         * 9. O valor da capacidade antes da alocação, na rota interferente 2 (cpBefore_i2) : int
         * 10. O valor da capacidade depois da alocação em FF, na rota principal (cp_FF_After_r1) : int
         * 11. O valor da capacidade depois da alocação em FF, na rota interferente 1 (cp_FF_After_r2) : int
         * 12. O valor da capacidade depois da alocação em FF, na rota interferente 2 (cp_FF_After_r3) : int
         * 13. O valor da capacidade depois da alocação em LF, na rota principal (cp_LF_After_r1) : int
         * 14. O valor da capacidade depois da alocação em LF, na rota interferente 1 (cp_LF_After_r2) : int
         * 15. O valor da capacidade depois da alocação em LF, na rota interferente 2 (cp_LF_After_r3) : int
         * 16. Ocupação da rota principal antes da alocação (ocupationBefore_r1) : int
         * 17. Ocupação da rota interferente 1 antes da alocação (ocupationBefore_r2) : int
         * 18. Ocupação da rota interferente 2 antes da alocação (ocupationBefore_r3) : int
         * 19. Slot inicial da alocação em FF (slotFF) : int
         * 20. Slot inicial da alocação em LF (slotLF) : int
        */

        return new int[]{   bestIndexSlot, 
                            bestCapacityLoss, 
                            tag,
                            cpBefore[0],
                            cpBefore[1],
                            cpBefore[2],
                            cp_FF_After[0],
                            cp_FF_After[1],
                            cp_FF_After[2],
                            cp_LF_After[0],
                            cp_LF_After[1],
                            cp_LF_After[2],
                            ocupationBefore[0],
                            ocupationBefore[1],
                            ocupationBefore[2],
                            slotFF,
                            slotLF
                        };
    }
    private int[] runMSCL2(Route route, List<MSCLApeture> allAperturesInMainRoute, int[] possibleSlotsByRoute, int reqNumbOfSlots, int numberMaxSlotsPerLink) throws Exception {

        // Armazena a lista com todas as rotas interferentes
        final List<Route> allConflictRoutes = route.getAllConflictRoutes();
        //TODO: Implementar os demais modos de rotas interferentes
        //TODO: Implementar os demais modos de ordenação das rotas interferentes

        // Inicializa a variável que armazena a melhor perda de capacidade
        double bestCapacityLoss = Double.MAX_VALUE * 0.8;
        
        // Inicializa a variável que armazena o melhor slot para alocar a requisição de acordo com o MSCL
        int bestIndexSlot = -1;

        // Inicializa a variável que armazena o melhor slot para alocar a requisição de acordo com o First Fit
        int bestFFIndexSlot = -1;

        // Armazena a perda de capacidade para o slot do First Fit
        double bestFFCapacityLoss = Double.MAX_VALUE * 0.8;

        // Percorre todas as lacunas da rota principal evitando percorrer os slots que não são possíveis alocar a requisição
        for (MSCLApeture apertureInMainRoute: allAperturesInMainRoute) {

            // Armazena a posição inicial e o tamanho da lacuna
            final int initPosInApertureInMainRoute = apertureInMainRoute.getInitPosition();
            final int sizeInApertureInMainRoute = apertureInMainRoute.getSize();

            // Inicializa a variável que armazena a perda de capacidade total para o slot 'indexSlot'
            double totalCapacityLoss = 0.0;
            double capacityLossBefore = 0.0;
            double capacityLossAfter = 0.0;

            // ** Capacidade antes da alocação na rota principal

            // * Rota principal

            for (int possibleReqSize: possibleSlotsByRoute){

                if (possibleReqSize > sizeInApertureInMainRoute){
                    break;
                }
                capacityLossBefore += (sizeInApertureInMainRoute - possibleReqSize + 1);
            }

            // * Conjunto de rotas interferentes

            // Percorre todas as rotas interferentes
            for (Route conflictRoute: allConflictRoutes) {

                // Incrementa uma unidade do ciclos MSCL
                this.cyclesMSCL++;

                // Busca o primeiro slot livre a esquerda antes da próxima lacuna. FIXME:
                final int minSlot = this.findSlotInLeft(conflictRoute, initPosInApertureInMainRoute);

                // Busca o primeiro slot livre a direita antes da próxima lacuna. FIXME:
                final int maxSlot = this.findSlotInRight(conflictRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);

                // Encontra as lacunas na rota interferente entre os slots 'minSlot' e 'maxSlot'
                final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, minSlot, maxSlot);

                // Percorre as lacunas encontradas na rota interferente
                for (MSCLApeture apetureInConflictRoute : allApeturesAfectInConflictRoute) {

                    final int sizeInApetureInConflictRoute = apetureInConflictRoute.getSize();

                    for (int possibleReqSize: possibleSlotsByRoute){

                        if (possibleReqSize > sizeInApetureInConflictRoute){
                            break;
                        }
                        capacityLossBefore += (sizeInApetureInConflictRoute - possibleReqSize + 1);
                    }
                }
            }

            // Percorre todos os slots dentro da lacuna
            for (int indexSlot = initPosInApertureInMainRoute; indexSlot < initPosInApertureInMainRoute + sizeInApertureInMainRoute; indexSlot++) {

                // Considerando uma requisição de tamanho 'reqNumbOfSlots', estabelece o slot inicial e o final
                int startSlot = indexSlot;
                int finalSlot = indexSlot + reqNumbOfSlots - 1;

                // Se o slot final estiver fora do espectro ou da lacuna, então para o laço
                if ((finalSlot >= numberMaxSlotsPerLink) || (finalSlot >= initPosInApertureInMainRoute + sizeInApertureInMainRoute)){
                    break;
                }

                // *** Chegando aqui é possível fazer a alocação e começa o cálculo de capacidade para o slot 'indexSlot'

                // Armazena o slot do First Fit
                if (bestFFIndexSlot == -1){
                    bestFFIndexSlot = indexSlot;
                }

                // Cria uma lista dos slots fake necessários para alocar a requisição
                List<Integer> slotsReqFake = new ArrayList<Integer>();
                for (int s = startSlot; s <= finalSlot; s++){
                    slotsReqFake.add(s);
                }

                // Inicializa a variável que armazena a perda de capacidade total para o slot 'indexSlot'
                totalCapacityLoss = 0.0;
                capacityLossAfter = 0.0;

                // Realiza a alocação fake na rota principal
                route.incrementSlotsOcupy(slotsReqFake);
                
                // ** Capacidade depois da alocação na rota principal
                
                // Encontra os buracos formados após a alocação fake. 
                // FIXME: Verificar se os parâmetros informados a busca dos burracos estão corretos. Seria todos os buracos formados ou somente os laterais?
                final List<MSCLApeture> aperturesInMainRouteAfter = MSCLApeture.genApetures(route, initPosInApertureInMainRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);

                // * Rota principal

                for (MSCLApeture apetureInApetureMainRouteAfter : aperturesInMainRouteAfter) {

                    final int sizeInapetureInApetureMainRouteAfter = apetureInApetureMainRouteAfter.getSize();

                    for (int possibleReqSize: possibleSlotsByRoute){
                        if (possibleReqSize > sizeInapetureInApetureMainRouteAfter){
                            break;
                        }
                        capacityLossAfter += (sizeInapetureInApetureMainRouteAfter - possibleReqSize + 1);
                    }
                }

                // * Conjunto de rotas interferentes

                // Percorre todas as rotas interferentes
                for (Route conflictRoute: allConflictRoutes) {

                    // Incrementa uma unidade do ciclos MSCL
                    this.cyclesMSCL++;

                    // Busca o primeiro slot livre a esquerda antes da próxima lacuna. FIXME:
                    final int minSlot = this.findSlotInLeft(conflictRoute, initPosInApertureInMainRoute);

                    // Busca o primeiro slot livre a direita antes da próxima lacuna. FIXME:
                    final int maxSlot = this.findSlotInRight(conflictRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);

                    // Encontra as lacunas na rota interferente entre os slots 'minSlot' e 'maxSlot'
                    final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, minSlot, maxSlot);

                    // Percorre as lacunas encontradas na rota interferente
                    for (MSCLApeture apetureInConflictRoute : allApeturesAfectInConflictRoute) {

                        final int sizeInApetureInConflictRoute = apetureInConflictRoute.getSize();

                        for (int possibleReqSize: possibleSlotsByRoute){

                            if (possibleReqSize > sizeInApetureInConflictRoute){
                                break;
                            }
                            capacityLossAfter += (sizeInApetureInConflictRoute - possibleReqSize + 1);
                        }
                    }
                }

                // Realiza a desalocação fake na rota principal
                route.decreasesSlotsOcupy(slotsReqFake);

                // Calcula a perda de capacidade na rota principal e nas rotas interferentes para o slot 'indexSlot'
                totalCapacityLoss = capacityLossBefore - capacityLossAfter;

                // Verifica se a perda de capacidade para o slot 'indexSlot' é menor que zero e lança uma exceção
                if (totalCapacityLoss < 0){
                    throw new Exception("Erro: totalCapacityLoss < 0");
                }

                // Se a perda de capacidade é do slot do First Fit, então armazena o valor
                if (indexSlot == bestFFIndexSlot){
                    bestFFCapacityLoss = totalCapacityLoss;
                }

                // Se a perda de capacidade para o slot 'indexSlot' é menor que a melhor perda de capacidade e atualiza a melhor perda de capacidade. Os casos de igualdade são tratados pegando o último slot que teve a melhor perda de capacidade, exceto quando o slot do First Fit tem a melhor perda de capacidade.
                if ((totalCapacityLoss < bestCapacityLoss) || ((totalCapacityLoss == bestCapacityLoss) && (totalCapacityLoss < bestFFCapacityLoss))){
                //if ((totalCapacityLoss < bestCapacityLoss) ){
                    bestCapacityLoss = totalCapacityLoss;
                    bestIndexSlot = indexSlot;
                }
            }
        }  

        int tag = 1;
        if (bestIndexSlot == bestFFIndexSlot){
            tag = 0;
        }

        return new int[]{bestIndexSlot, (int)bestCapacityLoss, (int)bestFFCapacityLoss, tag};
    }

    private double findSlotsAndCapacityLoss_BOA(Route currentRoute, CallRequest callRequest) throws Exception {

        
        // Armazena o número máximo de slots por enlace
        final int numberMaxSlotsPerLink = ParametersSimulation.getNumberOfSlotsPerLink();
        
        // Calcula o tamanho da requisição de acordo com a taxa de transmissão
        final int reqNumbOfSlots = currentRoute.getReqSize(callRequest.getSelectedBitRate());
        
        // Armazena o nó de origem e destino da rota principal
        
        // Armazena o ID da requisição para fins de armazenamento dos dados
        final int reqID = this.folderManager.getReqID();
        //System.out.println("\n\n*** Req = " + reqID);
        
        // Armazena todos os possíveis tamanhos de requisição para cada valor de bitrate para a maior modulação possível nessa rota
        final int[] possibleSlotsByRoute = currentRoute.getAllReqSizes();

        // Encontra a lista de lacunas para a rota principal ao longo de todo o espectro
        List<MSCLApeture> allAperturesInMainRoute = MSCLApeture.genApetures(currentRoute, 0, numberMaxSlotsPerLink - 1);
        
        // Verifica se é possível alocar essa requisição dentro da rota principal
        boolean isPossibleToAlocateReq = false;
        for (int index = 0; index < allAperturesInMainRoute.size(); index++){
            if (allAperturesInMainRoute.get(index).getSize() >= reqNumbOfSlots){
                isPossibleToAlocateReq = true;
            }
            else {
                allAperturesInMainRoute.remove(allAperturesInMainRoute.get(index));
                index--;
            }
        }
        
        //Se não há recursos disponíveis para alocar a requisição então retorna um valor alto de perda de capacidade
        if (!isPossibleToAlocateReq){
            // Armazena uma estrutura vazia para a requisição
            this.slotsMSCL.add(new ArrayList<Integer>());
            
            return Double.MAX_VALUE * 0.8; 
        }

        // Armazena a lista com todas as rotas interferentes
        final List<Route> allConflictRoutes = currentRoute.getAllConflictRoutes();
        //TODO: Implementar os demais modos de rotas interferentes
        //TODO: Implementar os demais modos de ordenação das rotas interferentes

        // Inicializa a variável que armazena a melhor perda de capacidade
        double bestCapacityLoss = Double.MAX_VALUE * 0.8;
        
        // Inicializa a variável que armazena o melhor slot para alocar a requisição de acordo com o MSCL
        int bestIndexSlot = -1;

        // Inicializa a variável que armazena o melhor slot para alocar a requisição de acordo com o First Fit
        int bestFFIndexSlot = -1;

        // Armazena a perda de capacidade para o slot do First Fit
        double bestFFCapacityLoss = Double.MAX_VALUE * 0.8;

        // Percorre todas as lacunas da rota principal evitando percorrer os slots que não são possíveis alocar a requisição
        for (MSCLApeture apertureInMainRoute: allAperturesInMainRoute) {

            // Armazena a posição inicial e o tamanho da lacuna
            final int initPosInApertureInMainRoute = apertureInMainRoute.getInitPosition();
            final int sizeInApertureInMainRoute = apertureInMainRoute.getSize();

            // Inicializa a variável que armazena a perda de capacidade total para o slot 'indexSlot'
            double totalCapacityLoss = 0.0;
            double capacityLossBefore = 0.0;
            double capacityLossAfter = 0.0;

            // ** Capacidade antes da alocação na rota principal

            // * Rota principal

            for (int possibleReqSize: possibleSlotsByRoute){

                if (possibleReqSize > sizeInApertureInMainRoute){
                    break;
                }
                capacityLossBefore += (sizeInApertureInMainRoute - possibleReqSize + 1);
            }

            // * Conjunto de rotas interferentes

            // Percorre todas as rotas interferentes
            for (Route conflictRoute: allConflictRoutes) {

                // Incrementa uma unidade do ciclos MSCL
                this.cyclesMSCL++;

                // Busca o primeiro slot livre a esquerda antes da próxima lacuna. FIXME:
                //final int minSlot = this.findSlotInLeft(conflictRoute, initPosInApertureInMainRoute);

                // Busca o primeiro slot livre a direita antes da próxima lacuna. FIXME:
                //final int maxSlot = this.findSlotInRight(conflictRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);

                // Encontra as lacunas na rota interferente entre os slots 'minSlot' e 'maxSlot'
                //final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, minSlot, maxSlot);
                //final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, initPosInApertureInMainRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);
                final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, 0, numberMaxSlotsPerLink - 1);

                // Percorre as lacunas encontradas na rota interferente
                for (MSCLApeture apetureInConflictRoute : allApeturesAfectInConflictRoute) {

                    final int sizeInApetureInConflictRoute = apetureInConflictRoute.getSize();

                    for (int possibleReqSize: possibleSlotsByRoute){

                        if (possibleReqSize > sizeInApetureInConflictRoute){
                            break;
                        }
                        capacityLossBefore += (sizeInApetureInConflictRoute - possibleReqSize + 1);
                    }
                }
            }

            // Percorre todos os slots dentro da lacuna
            for (int indexSlot = initPosInApertureInMainRoute; indexSlot < initPosInApertureInMainRoute + sizeInApertureInMainRoute; indexSlot++) {

                // Considerando uma requisição de tamanho 'reqNumbOfSlots', estabelece o slot inicial e o final
                int startSlot = indexSlot;
                int finalSlot = indexSlot + reqNumbOfSlots - 1;

                // Se o slot final estiver fora do espectro ou da lacuna, então para o laço
                if ((finalSlot >= numberMaxSlotsPerLink) || (finalSlot >= initPosInApertureInMainRoute + sizeInApertureInMainRoute)){
                    break;
                }

                // *** Chegando aqui é possível fazer a alocação e começa o cálculo de capacidade para o slot 'indexSlot'

                // Armazena o slot do First Fit
                if (bestFFIndexSlot == -1){
                    bestFFIndexSlot = indexSlot;
                }

                // Cria uma lista dos slots fake necessários para alocar a requisição
                List<Integer> slotsReqFake = new ArrayList<Integer>();
                for (int s = startSlot; s <= finalSlot; s++){
                    slotsReqFake.add(s);
                }

                // Inicializa a variável que armazena a perda de capacidade total para o slot 'indexSlot'
                totalCapacityLoss = 0.0;
                capacityLossAfter = 0.0;

                // Realiza a alocação fake na rota principal
                currentRoute.incrementSlotsOcupy(slotsReqFake);
                
                // ** Capacidade depois da alocação na rota principal
                
                // Encontra os buracos formados após a alocação fake. 
                // FIXME: Verificar se os parâmetros informados a busca dos burracos estão corretos. Seria todos os buracos formados ou somente os laterais?
                //final List<MSCLApeture> aperturesInMainRouteAfter = MSCLApeture.genApetures(currentRoute, initPosInApertureInMainRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);
                //final List<MSCLApeture> aperturesInMainRouteAfter = MSCLApeture.genApetures(currentRoute, initPosInApertureInMainRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);
                final List<MSCLApeture> aperturesInMainRouteAfter = MSCLApeture.genApetures(currentRoute, 0, numberMaxSlotsPerLink - 1);

                // * Rota principal

                for (MSCLApeture apetureInApetureMainRouteAfter : aperturesInMainRouteAfter) {

                    final int sizeInapetureInApetureMainRouteAfter = apetureInApetureMainRouteAfter.getSize();

                    for (int possibleReqSize: possibleSlotsByRoute){
                        if (possibleReqSize > sizeInapetureInApetureMainRouteAfter){
                            break;
                        }
                        capacityLossAfter += (sizeInapetureInApetureMainRouteAfter - possibleReqSize + 1);
                    }
                }

                // * Conjunto de rotas interferentes

                // Percorre todas as rotas interferentes
                for (Route conflictRoute: allConflictRoutes) {

                    // Incrementa uma unidade do ciclos MSCL
                    this.cyclesMSCL++;

                    // Busca o primeiro slot livre a esquerda antes da próxima lacuna. FIXME:
                    //final int minSlot = this.findSlotInLeft(conflictRoute, initPosInApertureInMainRoute);

                    // Busca o primeiro slot livre a direita antes da próxima lacuna. FIXME:
                    //final int maxSlot = this.findSlotInRight(conflictRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);

                    // Encontra as lacunas na rota interferente entre os slots 'minSlot' e 'maxSlot'
                    //final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, minSlot, maxSlot);
                    //final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, initPosInApertureInMainRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);
                    final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, 0, numberMaxSlotsPerLink - 1);
  
                    // Percorre as lacunas encontradas na rota interferente
                    for (MSCLApeture apetureInConflictRoute : allApeturesAfectInConflictRoute) {

                        final int sizeInApetureInConflictRoute = apetureInConflictRoute.getSize();

                        for (int possibleReqSize: possibleSlotsByRoute){

                            if (possibleReqSize > sizeInApetureInConflictRoute){
                                break;
                            }
                            capacityLossAfter += (sizeInApetureInConflictRoute - possibleReqSize + 1);
                        }
                    }
                }

                // Realiza a desalocação fake na rota principal
                currentRoute.decreasesSlotsOcupy(slotsReqFake);

                // Calcula a perda de capacidade na rota principal e nas rotas interferentes para o slot 'indexSlot'
                totalCapacityLoss += capacityLossBefore - capacityLossAfter;

                // Verifica se a perda de capacidade para o slot 'indexSlot' é menor que zero e lança uma exceção
                if (totalCapacityLoss < 0){
                    throw new Exception("Erro: totalCapacityLoss < 0");
                }

                // Se a perda de capacidade é do slot do First Fit, então armazena o valor
                if (indexSlot == bestFFIndexSlot){
                    bestFFCapacityLoss = totalCapacityLoss;
                }

                if (((totalCapacityLoss == bestCapacityLoss) && (totalCapacityLoss < bestFFCapacityLoss))){
                    double ded = 0.0;
                }

                // Se a perda de capacidade para o slot 'indexSlot' é menor que a melhor perda de capacidade e atualiza a melhor perda de capacidade. Os casos de igualdade são tratados pegando o último slot que teve a melhor perda de capacidade, exceto quando o slot do First Fit tem a melhor perda de capacidade.
                if ((totalCapacityLoss < bestCapacityLoss) || ((totalCapacityLoss == bestCapacityLoss) && (totalCapacityLoss < bestFFCapacityLoss))){
                //if ((totalCapacityLoss < bestCapacityLoss) ){
                    bestCapacityLoss = totalCapacityLoss;
                    bestIndexSlot = indexSlot;
                }

                // Verifica se a perda de capacidade para o slot 'indexSlot' é menor que a melhor perda de capacidade e atualiza a melhor perda de capacidade
                // if ((totalCapacityLoss < bestCapacityLoss)){
                //     bestCapacityLoss = totalCapacityLoss;
                //     bestIndexSlot = indexSlot;
                // }

                //System.out.println("Req = " + reqID + " - Slot = " + indexSlot + " - Capacity Loss = " + totalCapacityLoss);

            }
        }

        // Cria uma lista com os slots necessários para alocar a requisição com a melhor perda de capacidade
        List<Integer> slotsReq = new ArrayList<Integer>();
        for (int s = bestIndexSlot; s <= bestIndexSlot + reqNumbOfSlots - 1; s++){
            slotsReq.add(s);
        }

        // Adiciona a lista de slots necessários para alocar a requisição com a melhor perda de capacidade
        this.slotsMSCL.add(slotsReq);


        int tag = 1;
        if (bestIndexSlot == bestFFIndexSlot){
            tag = 0;
        }
        if (reqID % 4 == 0) {

            //this.folderManager.saveDataBinListsCLiRoutes(reqID, bestIndexSlot, (int)(bestCapacityLoss), reqNumbOfSlots, currentRoute.getOriginNone(), tag);
            //this.folderManager.saveDataBinListsCLiRoutes_OnlyMain(reqID, bestIndexSlot, (int)(bestCapacityLoss), reqNumbOfSlots, currentRoute, tag);
        }

        if ((tag == 1) && (bestCapacityLoss < bestFFCapacityLoss)){
            double ded = 0.0;

        }


        //System.out.print("ReqID: " + reqID + " - Route Path: " + currentRoute.getPath() + " - Slots: " + slotsReq + " - Capacity Loss: " + bestCapacityLoss + " - Tag: " + tag + "\n");

        return bestCapacityLoss;
    }

    private double findSlotsAndCapacityLoss2(Route currentRoute, CallRequest callRequest) throws Exception {

        // Armazena o número máximo de slots por enlace
        final int numberMaxSlotsPerLink = ParametersSimulation.getNumberOfSlotsPerLink();
        
        // Calcula o tamanho da requisição de acordo com a taxa de transmissão
        final int reqNumbOfSlots = currentRoute.getReqSize(callRequest.getSelectedBitRate());

        // Armazena o nó de origem e destino da rota principal
        final int sourceNodeID = currentRoute.getOriginNone();
        final int destinationNodeID = currentRoute.getDestinationNone();
        
        // Armazena o ID da requisição para fins de armazenamento dos dados
        final int reqID = this.folderManager.getReqID();
        
        // Armazena todos os possíveis tamanhos de requisição para cada valor de bitrate para a maior modulação possível nessa rota
        final int[] possibleSlotsByRoute = currentRoute.getAllReqSizes();

        // Encontra a lista de lacunas para a rota principal ao longo de todo o espectro
        List<MSCLApeture> allAperturesInMainRoute = MSCLApeture.genApetures(currentRoute, 0, numberMaxSlotsPerLink - 1);
        
        // Verifica se é possível alocar essa requisição dentro da rota principal
        boolean isPossibleToAlocateReq = false;
        for (int index = 0; index < allAperturesInMainRoute.size(); index++){
            if (allAperturesInMainRoute.get(index).getSize() >= reqNumbOfSlots){
                isPossibleToAlocateReq = true;
            }
            else {
                allAperturesInMainRoute.remove(allAperturesInMainRoute.get(index));
                index--;
            }
        }
        
        //Se não há recursos disponíveis para alocar a requisição então retorna um valor alto de perda de capacidade
        if (!isPossibleToAlocateReq){
            // Armazena uma estrutura vazia para a requisição
            this.slotsMSCL.add(new ArrayList<Integer>());
            
            return Double.MAX_VALUE * 0.8; 
        }

        // Armazena a lista com todas as rotas interferentes
        final List<Route> allConflictRoutes = currentRoute.getAllConflictRoutes();
        //TODO: Implementar os demais modos de rotas interferentes
        //TODO: Implementar os demais modos de ordenação das rotas interferentes

        // Inicializa a variável que armazena a melhor perda de capacidade
        double bestCapacityLoss = Double.MAX_VALUE * 0.8;
        
        // Inicializa a variável que armazena o melhor slot para alocar a requisição
        int bestIndexSlot = -1;

        // Percorre todas as lacunas da rota principal evitando percorrer os slots que não são possíveis alocar a requisição
        for (MSCLApeture apertureInMainRoute: allAperturesInMainRoute) {

            // Armazena a posição inicial e o tamanho da lacuna
            final int initPosInApertureInMainRoute = apertureInMainRoute.getInitPosition();
            final int sizeInApertureInMainRoute = apertureInMainRoute.getSize();

            // Inicializa a variável que armazena a perda de capacidade total para o slot 'indexSlot'
            double totalCapacityLoss = 0.0;
            double capacityLossBefore = 0.0;
            double capacityLossAfter = 0.0;

            // ** Capacidade antes da alocação na rota principal

            // * Rota principal

            for (int possibleReqSize: possibleSlotsByRoute){

                if (possibleReqSize > sizeInApertureInMainRoute){
                    break;
                }
                capacityLossBefore += (sizeInApertureInMainRoute - possibleReqSize + 1);
            }

            // * Conjunto de rotas interferentes

            // Percorre todas as rotas interferentes
            for (Route conflictRoute: allConflictRoutes) {

                // Incrementa uma unidade do ciclos MSCL
                this.cyclesMSCL++;

                // Busca o primeiro slot livre a esquerda antes da próxima lacuna. FIXME:
                //final int minSlot = this.findSlotInLeft(conflictRoute, startSlot);

                // Busca o primeiro slot livre a direita antes da próxima lacuna. FIXME:
                //final int maxSlot = this.findSlotInRight(conflictRoute, finalSlot);

                // Encontra as lacunas na rota interferente entre os slots 'minSlot' e 'maxSlot'
                //final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, minSlot, maxSlot);
                //final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, initPosInApertureInMainRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);
                final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, 0, numberMaxSlotsPerLink - 1);

                // Percorre as lacunas encontradas na rota interferente
                for (MSCLApeture apetureInConflictRoute : allApeturesAfectInConflictRoute) {

                    final int sizeInApetureInConflictRoute = apetureInConflictRoute.getSize();

                    for (int possibleReqSize: possibleSlotsByRoute){

                        if (possibleReqSize > sizeInApetureInConflictRoute){
                            break;
                        }
                        capacityLossBefore += (sizeInApetureInConflictRoute - possibleReqSize + 1);
                    }
                }
            }

            // Percorre todos os slots dentro da lacuna
            POINT_SLOT:for (int indexSlot = initPosInApertureInMainRoute; indexSlot < initPosInApertureInMainRoute + sizeInApertureInMainRoute; indexSlot++) {

                // Considerando uma requisição de tamanho 'reqNumbOfSlots', estabelece o slot inicial e o final
                int startSlot = indexSlot;
                int finalSlot = indexSlot + reqNumbOfSlots - 1;

                // Se o slot final estiver fora do espectro ou da lacuna, então para o laço
                if ((finalSlot >= numberMaxSlotsPerLink) || (finalSlot >= initPosInApertureInMainRoute + sizeInApertureInMainRoute)){
                    break;
                }

                // Verifica se é possível alocar a requisição iniciando em startSlot. 
                //FIXME: Essa parte pode ser removida para otimização. Se a lacuna somente tem slots livre e o começo e final devem está dentro dela, então é possível remover essa verificação.
                // for (int slot = startSlot; slot <= finalSlot; slot++){
                //     if (currentRoute.getSlotValue(slot) > 0){
                //         continue POINT_SLOT;
                //     }
                // }

                // *** Chegando aqui é possível fazer a alocação e começa o cálculo de capacidade para o slot 'indexSlot'

                // Cria uma lista dos slots fake necessários para alocar a requisição
                List<Integer> slotsReqFake = new ArrayList<Integer>();
                for (int s = startSlot; s <= finalSlot; s++){
                    slotsReqFake.add(s);
                }

                // Inicializa a variável que armazena a perda de capacidade total para o slot 'indexSlot'
                totalCapacityLoss = 0.0;
                capacityLossAfter = 0.0;

                // Realiza a alocação fake na rota principal
                currentRoute.incrementSlotsOcupy(slotsReqFake);
                
                // ** Capacidade depois da alocação na rota principal
                
                // Encontra os buracos formados após a alocação fake. 
                // FIXME: Verificar se os parâmetros informados a busca dos burracos estão corretos. Seria todos os buracos formados ou somente os laterais?
                final List<MSCLApeture> aperturesInMainRouteAfter = MSCLApeture.genApetures(currentRoute, initPosInApertureInMainRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);
                //final List<MSCLApeture> aperturesInMainRouteAfter = MSCLApeture.genApetures(currentRoute, 0, numberMaxSlotsPerLink - 1);

                // * Rota principal

                for (MSCLApeture apetureInApetureMainRouteAfter : aperturesInMainRouteAfter) {

                    final int sizeInapetureInApetureMainRouteAfter = apetureInApetureMainRouteAfter.getSize();

                    for (int possibleReqSize: possibleSlotsByRoute){
                        if (possibleReqSize > sizeInapetureInApetureMainRouteAfter){
                            break;
                        }
                        capacityLossAfter += (sizeInapetureInApetureMainRouteAfter - possibleReqSize + 1);
                    }
                }

                // * Conjunto de rotas interferentes

                // Percorre todas as rotas interferentes
                for (Route conflictRoute: allConflictRoutes) {

                    // Incrementa uma unidade do ciclos MSCL
                    this.cyclesMSCL++;

                    // Busca o primeiro slot livre a esquerda antes da próxima lacuna. FIXME:
                    //final int minSlot = this.findSlotInLeft(conflictRoute, startSlot);

                    // Busca o primeiro slot livre a direita antes da próxima lacuna. FIXME:
                    //final int maxSlot = this.findSlotInRight(conflictRoute, finalSlot);

                    // Encontra as lacunas na rota interferente entre os slots 'minSlot' e 'maxSlot'
                    //final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, minSlot, maxSlot);
                    //final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, initPosInApertureInMainRoute, initPosInApertureInMainRoute + sizeInApertureInMainRoute - 1);
                    final List<MSCLApeture> allApeturesAfectInConflictRoute = MSCLApeture.genApetures(conflictRoute, 0, numberMaxSlotsPerLink - 1);
  
                    // Percorre as lacunas encontradas na rota interferente
                    for (MSCLApeture apetureInConflictRoute : allApeturesAfectInConflictRoute) {

                        final int sizeInApetureInConflictRoute = apetureInConflictRoute.getSize();

                        for (int possibleReqSize: possibleSlotsByRoute){

                            if (possibleReqSize > sizeInApetureInConflictRoute){
                                break;
                            }
                            capacityLossAfter += (sizeInApetureInConflictRoute - possibleReqSize + 1);
                        }
                    }
                }

                // Realiza a desalocação fake na rota principal
                currentRoute.decreasesSlotsOcupy(slotsReqFake);

                // Calcula a perda de capacidade na rota principal e nas rotas interferentes para o slot 'indexSlot'
                totalCapacityLoss += capacityLossBefore - capacityLossAfter;

                // Verifica se a perda de capacidade para o slot 'indexSlot' é menor que zero e lança uma exceção
                if (totalCapacityLoss < 0){
                    throw new Exception("Erro: totalCapacityLoss < 0");
                }

                // if (reqID == 11) {
                //     System.out.println("Req = " + reqID + " - Slot = " + indexSlot + " - Capacity Loss = " + totalCapacityLoss);
                // }

                // Verifica se a perda de capacidade para o slot 'indexSlot' é menor que a melhor perda de capacidade e atualiza a melhor perda de capacidade
                if (totalCapacityLoss < bestCapacityLoss){
                    bestCapacityLoss = totalCapacityLoss;
                    bestIndexSlot = indexSlot;
                }
            }
        }

        // Cria uma lista com os slots necessários para alocar a requisição com a melhor perda de capacidade
        List<Integer> slotsReq = new ArrayList<Integer>();
        for (int s = bestIndexSlot; s <= bestIndexSlot + reqNumbOfSlots - 1; s++){
            slotsReq.add(s);
        }

        // Adiciona a lista de slots necessários para alocar a requisição com a melhor perda de capacidade
        this.slotsMSCL.add(slotsReq);


        //System.out.print("ReqID: " + reqID + " - Route Path: " + currentRoute.getPath() + " - Slots: " + slotsReq + " - Capacity Loss: " + bestCapacityLoss + "\n");

        return bestCapacityLoss;
    }

    public int findSlotInLeft(Route route, int initSlot) throws Exception{
        // Busca o mínimo a esquerda
        int minSlot = initSlot; // Força um erro
        for (int min = initSlot; min >= 0;min--){

            //this.cycles++;

            if (route.getSlotValue(min) == 0){
                minSlot = min;
            } else {
                break;
            }
        }

        if (minSlot == -1){
            throw new Exception("Erro: emptySlots = -1");
        }

        return minSlot;
    }

    public int findSlotInRight(Route route, int initSlot) throws Exception{
        // Busca o máximo a direita
        int maxSlot = initSlot; // Força um erro
        for (int max = initSlot; max < ParametersSimulation.getNumberOfSlotsPerLink(); max++){

            //this.cycles++;

            if (route.getSlotValue(max) == 0){
                maxSlot = max;
            } else {
                break;
            }
        }

        if (maxSlot == ParametersSimulation.getNumberOfSlotsPerLink() + 1){
            throw new Exception("Erro: emptySlots = -1");
        }

        return maxSlot;
    }

    /**
     * Método para retornar a rota encontrada após utilizar o algoritmo de roteamento.
     * 
     * @return Retorna a rota encontrada
     */
    public Route getRoute() {
        return this.route;
    }

    /**
     * Método para retornar o conjunto dos slots encontrado
     * 
     * @return O conjunto dos slots encontrado
     */
    public List<Integer> getSlotsIndex() {
        return this.fSlots;
    }

    /**
     * Método para retornar o valor do ciclosMSCL
     * 
     * @return O valor do CicloMSCL para a rota selecionada
     */
    public long getCyclesMSCL() {
        return this.cyclesMSCL;
    }
}