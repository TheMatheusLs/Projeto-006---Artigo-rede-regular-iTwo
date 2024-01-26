package Main;

import java.util.List;
import java.util.Random;

import CallRequests.CallRequest;
import CallRequests.CallRequestManager;
import Config.ConfigSimulator;
import Config.ParametersSimulation;
import GeneralClasses.ProbabilityFunctions;
import Manager.FolderManager;
import Manager.SimulationResults;
import Network.TopologyManager;
import RSA.RSAManager;
import RSA.Routing.Route;
import RSA.Routing.RoutesManager;
import Types.GeneralTypes.CallRequestType;
import Types.GeneralTypes.RandomGenerationType;
import Types.GeneralTypes.StopCriteriaType;

public class Enviroment {

    private FolderManager folderManager;
    private RoutesManager routesManager;
    private int[] seedsForLoad;
    private long currentRandomSeed;
    private Random randomGeneration;
    private TopologyManager topology;
    private RSAManager rsaManager;

    // Variáveis para o Reset
    private long geralInitTime;
    private double meanRateCallDur;
    private long numberMaxOfRequisitions;
    private CallRequestType callRequestType;
    private int[] possibleBitRates;
    private StopCriteriaType stopCriteria;

    private double timeSim = 0.0;
    private int iReq = 0;
    private boolean hasSlots, hasQoT;
    private int numBlockBySlots = 0;
    private int numBlockByQoT = 0;
    private long limitCallRequest = 0;
    private long cyclesMSCL = 0;
    private boolean isDone = false;

    private double networkLoad;

    private double currentReward;

    private CallRequestManager listOfCalls;

    public Enviroment(FolderManager folderManager) throws Exception {
        this.folderManager = folderManager;
        this.seedsForLoad = this.generateRandomSeeds();

        // Cria uma nova instância da topologia
        this.topology = new TopologyManager();
        this.topology.save(this.folderManager);

        // Cria uma nova instância de routing
        this.routesManager = new RoutesManager(this.topology);
        this.routesManager.save(folderManager);

        this.folderManager.setNetworkOpticalLinks(this.topology.getNetworkOpticalLinks());

        this.rsaManager = new RSAManager(this.routesManager, this.folderManager, this.randomGeneration);

        this.currentRandomSeed = this.seedsForLoad[0];

        this.listOfCalls = new CallRequestManager();

        this.reset();

    }

    /**
     * Método para inicializar a simulação
     * 
     * @throws Exception
     */
    public int reset() throws Exception {
        
        this.listOfCalls.desallocateAllRequests(); 

        this.listOfCalls.eraseCallList();

        this.topology.checkIfIsClean();

		// Verifica se todas as rotas estão limpas
        this.routesManager.checkIfIsClean();

        //this.topology.reset();
        //this.routesManager.reset();

        this.randomGeneration = new Random(this.currentRandomSeed);

        this.geralInitTime = System.currentTimeMillis();
        this.meanRateCallDur = ConfigSimulator.getMeanRateOfCallsDuration();
        this.numberMaxOfRequisitions = ParametersSimulation.getMaxNumberOfRequisitions();
        this.callRequestType = ParametersSimulation.getCallRequestType();
        this.possibleBitRates = ParametersSimulation.getTrafficOption();
        this.stopCriteria = ParametersSimulation.getStopCriteriaType();

        this.timeSim = 0.0;
        this.hasSlots = false;
        this.hasQoT = false;
        this.numBlockBySlots = 0;
        this.numBlockByQoT = 0;
        this.limitCallRequest = 0;
        this.cyclesMSCL = 0;

        this.iReq = 0;

        this.currentReward = 0.0;


        return this.getObservation();
    }

    private int getObservation() {
        return 0;
    }

    public boolean isDone() {
        return this.isDone;
    }

    public int findNewOrigin() {
        return (int) Math.floor(this.randomGeneration.nextDouble() * this.topology.getNumberOfNodes());
    }

    public int findNewDestination(int source) {
        int destination;
        do {
            destination = (int) Math.floor(randomGeneration.nextDouble() * this.topology.getNumberOfNodes());
        } while (source == destination);
        return destination;
    }

    /**
     * Método para executar um passo da simulação
     * 
     * @throws Exception
     */
    public int step(double networkLoad, int action_index, int source, int destination) throws Exception {

        this.networkLoad = networkLoad;

        this.folderManager.setReqID(this.iReq);

        // Apresenta o progresso para a simulação
        if ((this.iReq % 10000) == 0) {
            System.out.print(">");
        }

        this.hasQoT = false;
        this.hasSlots = false;

        this.isDone = false;

        // Remove as requisições espiradas
        this.listOfCalls.removeCallRequest(timeSim);

        this.timeSim += ProbabilityFunctions.exponentialDistribution(networkLoad, this.randomGeneration);

        final CallRequest callRequest = new CallRequest(iReq, source, destination, callRequestType, possibleBitRates,
                timeSim, meanRateCallDur, this.randomGeneration);

        // Executa o problema do RSA
        if (action_index == 0) {
            this.rsaManager.findRoutingSA(source, destination, callRequest);
        } else {
            this.rsaManager.findSARouting(source, destination, callRequest);
        }
        Route route = this.rsaManager.getRoute();
        List<Integer> fSlotsIndex = this.rsaManager.getSlotsIndex();
        this.cyclesMSCL += this.rsaManager.getCyclesMSCL();

        if (route != null) {

            if (!fSlotsIndex.isEmpty() && fSlotsIndex.size() == callRequest.getReqNumbOfSlots()) { // NOPMD by Andr� on
                                                                                                   // 13/06/17 13:12
                hasSlots = true;
            }

            hasQoT = route.isQoT();

            if (hasSlots && hasQoT) {
                callRequest.setFrequencySlots(fSlotsIndex);
                callRequest.setRoute(route);

                // Incrementar os slots que estão sendo utilizados pelas rotas
                route.incrementSlotsOcupy(fSlotsIndex);

                callRequest.allocate(topology.getListOfNodes());
                listOfCalls.addCall(callRequest);

                this.currentReward = +0.1;
            }
        }

        if (!hasSlots) {
            this.numBlockBySlots++;
            this.currentReward = -1;
        } else if (!hasQoT) {
            this.numBlockByQoT++;
            this.currentReward = -1;
        }

        if (stopCriteria == StopCriteriaType.BlockedCallRequest) {
            if ((numBlockBySlots + numBlockByQoT) >= ParametersSimulation.getMaxNumberOfBlockedRequests()) {
                this.isDone = true;
            }
        }

        this.iReq += 1;

        if (this.iReq >= this.numberMaxOfRequisitions) {
            this.isDone = true;
        }

        return this.getObservation();
    }

    public double getReward() {
        return this.currentReward;
    }

    public int getNumberOfNodes() {
        return this.topology.getNumberOfNodes();
    }

    public int getNumberOfActions() {
        return 2;
    }

    public int getNextAction(double[][][] qMatrix, int source, int destination, double epsilon) {
        if (this.randomGeneration.nextDouble() < epsilon) {
            return this.randomGeneration.nextInt(this.getNumberOfActions());
        } else {
            return this.getBestAction(qMatrix, source, destination);
        }
    }

    private int getBestAction(double[][][] qMatrix, int source, int destination) {
        int bestAction = 0;
        double bestValue = qMatrix[source][destination][0];
        for (int action = 1; action < this.getNumberOfActions(); action++) {
            if (qMatrix[source][destination][action] > bestValue) {
                bestValue = qMatrix[source][destination][action];
                bestAction = action;
            }
        }
        return bestAction;
    }

    public double getBestValue(double[] values) {
        double bestValue = values[0];
        for (int action = 1; action < this.getNumberOfActions(); action++) {
            if (values[action] > bestValue) {
                bestValue = values[action];
            }
        }
        return bestValue;
    }

    public SimulationResults results() {

        SimulationResults simulationResults = new SimulationResults(this.networkLoad, this.iReq,
                this.currentRandomSeed);

        final long geralTotalTime = System.currentTimeMillis() - geralInitTime;
        simulationResults.setExecutionTime(geralTotalTime);

        double PB = (double) (this.numBlockBySlots + this.numBlockByQoT) / this.iReq;
        simulationResults.setProbabilityBlocking(PB);
        simulationResults.setNumBlockBySlots(this.numBlockBySlots);
        simulationResults.setNumBlockByQoT(this.numBlockByQoT);
        simulationResults.setMSCLCycle(this.cyclesMSCL);

        return simulationResults;
    }

    /**
     * Cria as sementes aleatórias da rede conforme a métrica escolhida
     * 
     * @return Lista com as sementes
     */
    private int[] generateRandomSeeds() {

        int[] auxSeeds = new int[ParametersSimulation.getNumberOfSimulationsPerLoadNetwork()];

        if (ParametersSimulation.getRandomGeneration().equals(RandomGenerationType.PseudoRandomGeneration)) {
            Random randomAux = new Random(ParametersSimulation.getMainSeed());

            for (int nSim = 0; nSim < ParametersSimulation.getNumberOfSimulationsPerLoadNetwork(); nSim++) {
                auxSeeds[nSim] = randomAux.nextInt(Integer.MAX_VALUE);
            }

            this.randomGeneration = new Random(ParametersSimulation.getMainSeed());
        } else {
            if (ParametersSimulation.getRandomGeneration().equals(RandomGenerationType.SameRequestForAllPoints)) {

                Random randomAux = new Random(ParametersSimulation.getMainSeed());

                int seedFix = randomAux.nextInt(Integer.MAX_VALUE);

                for (int nSim = 0; nSim < ParametersSimulation.getNumberOfSimulationsPerLoadNetwork(); nSim++) {
                    auxSeeds[nSim] = seedFix;
                }

                this.randomGeneration = new Random(ParametersSimulation.getMainSeed());
            } else {
                if (ParametersSimulation.getRandomGeneration().equals(RandomGenerationType.RandomGeneration)) {
                    Random randomAux = new Random();

                    this.randomGeneration = new Random(randomAux.nextInt(Integer.MAX_VALUE));

                    for (int nSim = 0; nSim < ParametersSimulation.getNumberOfSimulationsPerLoadNetwork(); nSim++) {
                        auxSeeds[nSim] = randomAux.nextInt(Integer.MAX_VALUE);
                    }
                }
            }
        }

        return auxSeeds;
    }
}
