package Network.Topologies;

public class TopologyOneLinkOnly extends TopologyGeneral{
    /**
     * Quantidade de Nós presentes na topologia
     */
    private static int numberOfNodes = 2;

    public TopologyOneLinkOnly(){
        super(getLength());
    }

    /**
	 * Tamanho dos links da rede One Link
	 * 
     * @return
     */
    public static double[][] getLength(){
        //create network adjacency matrix
		double[][] lengths = new double[numberOfNodes][numberOfNodes];			
		for(int x = 0; x < numberOfNodes; x++){
			for(int y = 0 ; y < numberOfNodes; y++){
				lengths[x][y] = Double.MAX_VALUE;
			}
		}
		
		lengths[0][1]   = 300;
	
		return lengths;
    }
}
