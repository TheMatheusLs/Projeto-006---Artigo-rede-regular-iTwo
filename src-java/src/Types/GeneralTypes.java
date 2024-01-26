package Types;

public class GeneralTypes {
    
    public enum RoutingAlgorithmType{
        Dijstra,
        YEN,
        MSCLSequencial,
        MSCLCombinado;
    }


    public enum CallRequestType{
        Unidirectional,
        Bidirectional;
    }


    public enum TopologyType{
        IRoutes,
        NSFNet,
        OneLink,
        Simple6,
        OneLinkOnly;
    }


    public enum SpectralAllocationAlgorithmType{
        FirstFit,
        FirstLastFit,
        MSCL,
        Random;
    }


    public enum LinkCostType{
        Hops,
        Length,
        LengthNormalized;
    }


    public enum StopCriteriaType{
        TotalCallRequest,
        BlockedCallRequest;
    }


    public enum RandomGenerationType{
        SameRequestForAllPoints,
        PseudoRandomGeneration,
        RandomGeneration;
    }


    public enum GainAlgorithmType{
        Basic;
    }


    public enum KSortedRoutesByType{
        MaxToMinOcupation,
        MinToMaxOcupation,
        None;
    }


    public enum PhysicalLayerOption{
        Disabled,
        Enabled;
    }

    
    public enum RSAOrderType{
        Routing_SA,
        SA_Routing,
        None;
    }
}
