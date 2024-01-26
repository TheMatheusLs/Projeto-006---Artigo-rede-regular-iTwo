package RSA.Spectrum.Algorithms;

import java.util.ArrayList;
import java.util.List;

import Config.ParametersSimulation;
import Network.Structure.OpticalLink;
import RSA.Routing.Route;

public class FirstLastFit extends SpectrumAlgorithm{
    
    public FirstLastFit(){
    }

    /**
     * Método para encontrar o primeiro conjunto de slots disponível no espectro pra a rota. Algoritmo First-Fit
     * 
     * @param reqNumbOfSlots
	 * @param route
	 * @return O cojunto de índice dos slots em um lista de inteiros
     * @author André 
     */
    @Override
    public List<Integer> findFrequencySlots(final int reqNumbOfSlots, final Route route) throws Exception{

        List<OpticalLink> uplink = route.getUpLink();
        List<OpticalLink> downlink = route.getUpLink();

        int numberMaxFreqSlots = ParametersSimulation.getNumberOfSlotsPerLink();

        if(reqNumbOfSlots <= 0){
            throw new Exception("Required number of frequency slots is invalid");
        } else if (uplink.isEmpty()){
            throw new Exception("Routing solution is invalid");         
        }
       

        final boolean bidirectional = !uplink.isEmpty() && !downlink.isEmpty();

        final List<Integer> slots = new ArrayList<Integer>();
        final List<OpticalLink> links = new ArrayList<>(uplink);

        // For indo de 0, 320, 1, 319, 2, 318, ..., 159, 160
        fMaxSlots:for (int i = 0; i <= numberMaxFreqSlots; i++) {
            int iSlot;
            if (i % 2 == 0) {
                // First-Fit
                iSlot = (int)Math.floor(i / 2);

                boolean availableSlot = true;
                boolean availableLastSlot = true;
                int count = reqNumbOfSlots;

                // Percorre o primeiro slot de cada link buscando se todos estão disponíveis
                for (OpticalLink link : links) {
                    final boolean availSlotIn = link.isAvailableSlotAt(iSlot);
                    if(!availSlotIn){
                        continue fMaxSlots;
                    }
                }

                // Percorre o último slot de cada link buscando se todos estão disponíveis
                if (iSlot + reqNumbOfSlots - 1 < numberMaxFreqSlots) {
                    for (OpticalLink link : links) {
                        final boolean availSlotIn = link.isAvailableSlotAt(iSlot + reqNumbOfSlots - 1);
                        if(!availSlotIn){
                            availableLastSlot = false;
                            break;
                        }
                    }
                } else {
                    availableLastSlot = false;
                }
                
                if ((iSlot + reqNumbOfSlots -1 < numberMaxFreqSlots) && (availableLastSlot)){
                    slots.add(iSlot); //primeiro slot encontrado.
                    count--;

                    if(count == 0){ //Encontrou os slots necess�rios.
                        break fMaxSlots;
                    }

                    if(iSlot + reqNumbOfSlots - 1 > numberMaxFreqSlots){ //Se a soma for maior que o numberOfSlots, não há slots disponíveis na rota.
                        slots.clear();
                        break fMaxSlots;						
                    } else {
                        for(int r = iSlot + 1; r < (reqNumbOfSlots + iSlot) && r < numberMaxFreqSlots; r++){ //tenta encontrar o resto dos slots contiguos e continuos
                            for (OpticalLink link : links) {
                                final boolean availSlotIn = link.isAvailableSlotAt(r);
                                if(!availSlotIn){
                                    availableSlot = false;
                                    break;
                                }
                            } 

                            if(availableSlot){
                                slots.add(r);
                                count--;							
                            } else {
                                iSlot = r++;
                                slots.clear();
                                break;
                            }

                            if(count == 0){ //Encontrou o slots necess�rios.
                                break fMaxSlots;
                            }
                        }
                    }
                } else {
                    iSlot = iSlot + reqNumbOfSlots - 1;
                    slots.clear();
                }

            } else {
                // Last-Fit
                iSlot = numberMaxFreqSlots - 1 - (int)Math.floor(i / 2);

                boolean availableSlot = true;
                boolean availableLastSlot = true;
                int count = reqNumbOfSlots;

                // Percorre o primeiro slot de cada link buscando se todos estão disponíveis
                for (OpticalLink link : links) {
                    final boolean availSlotIn = link.isAvailableSlotAt(iSlot);
                    if(!availSlotIn){
                        continue fMaxSlots;
                    }
                }

                // Percorre o último slot de cada link buscando se todos estão disponíveis
                if (iSlot - reqNumbOfSlots + 1 < numberMaxFreqSlots) {
                    for (OpticalLink link : links) {
                        final boolean availSlotIn = link.isAvailableSlotAt(iSlot - reqNumbOfSlots + 1);
                        if(!availSlotIn){
                            availableLastSlot = false;
                            break;
                        }
                    }
                } else {
                    availableLastSlot = false;
                }
                
                if ((iSlot - reqNumbOfSlots + 1 >= 0) && (availableLastSlot)){
                    slots.add(iSlot); //primeiro slot encontrado.
                    count--;

                    if(count == 0){ //Encontrou os slots necess�rios.
                        break fMaxSlots;
                    }

                    if(iSlot - reqNumbOfSlots + 1 < 0){ //Se a soma for maior que o numberOfSlots, não há slots disponíveis na rota.
                        slots.clear();
                        break fMaxSlots;						
                    } else {
                        for(int r = iSlot - 1; r > (iSlot - reqNumbOfSlots) && r >= 0; r--){ //tenta encontrar o resto dos slots contiguos e continuos
                            for (OpticalLink link : links) {
                                final boolean availSlotIn = link.isAvailableSlotAt(r);
                                if(!availSlotIn){
                                    availableSlot = false;
                                    break;
                                }
                            } 

                            if(availableSlot){
                                slots.add(r);
                                count--;							
                            } else {
                                iSlot = r++;
                                slots.clear();
                                break;
                            }

                            if(count == 0){ //Encontrou o slots necess�rios.
                                break fMaxSlots;
                            }
                        }
                    }
                } else {
                    iSlot = iSlot + reqNumbOfSlots - 1;
                    slots.clear();
                }
            } 
        }

        return slots;
    }
}
