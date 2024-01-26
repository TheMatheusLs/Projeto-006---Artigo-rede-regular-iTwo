package Network.Topologies;

public class TopologyIRoutes extends TopologyGeneral{
    /**
     * Quantidade de Nós presentes na topologia
     */
    private static int numberOfNodes = 8;

    public TopologyIRoutes(){
        super(getLength());
    }

    /**
	 * Tamanho dos links da rede NSFNet
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
		

		lengths[0][1]   = 300; // 1 2
		lengths[1][2]   = 300; // 2 3
		lengths[2][3]   = 300; // 3 4
		lengths[3][4]   = 300; // 4 5
		lengths[4][5]   = 300; // 5 6
		lengths[2][5]   = 300; // 3 6
		lengths[5][6]   = 300; // 6 7
		lengths[6][7]   = 300; // 7 8


		return lengths;
    }
}
