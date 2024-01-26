package Network.Topologies;

public class Topology6Simple extends TopologyGeneral{
    /**
     * Quantidade de Nós presentes na topologia
     */
    private static int numberOfNodes = 6;

    public Topology6Simple(){
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
		

		lengths[0][1]   = 300;
		lengths[1][0]   = 300;
		lengths[0][5]   = 300;
		lengths[5][0]   = 300;
		lengths[1][2]   = 300;
		lengths[2][1]   = 300;
		lengths[1][4]   = 300;
		lengths[4][1]   = 300;
		lengths[2][3]   = 300;
		lengths[3][2]   = 300;
		lengths[2][5]   = 300;
		lengths[5][2]   = 300;
		lengths[3][4]   = 300;
		lengths[4][3]   = 300;
		lengths[5][4]   = 300;
		lengths[4][5]   = 300;

		return lengths;
    }
}
