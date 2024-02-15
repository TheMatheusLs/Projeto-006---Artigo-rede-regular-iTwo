
# Tipo de conexão entre os enlaces
IS_BIDIRECTIONAL = False 

# Tipo de custo para a alocação de slots # 'length' or 'hops'
IS_COST_TYPE = 'hops'

# Número de classes de slots
DEMANDS_CLASS = [2, 3, 6]

# Número de requisições
MAX_REQS = 100_000

# Duração média das chamadas (em Erlangs)
MEAN_CALL_DURATION = 1

# Número de slots por enlace
NUMBER_OF_SLOTS = 128

# Número de códigos
CLASS_TAG_RSA_FF = 0
CLASS_TAG_MSCL = 1
CLASS_TAG_LF = 2
CLASS_TAG_SAR = 3

RSA_CODE = 0
SAR_CODE = 1
MSCL_CODE = 2

# Carga da rede (em Erlangs)
LOAD = 300

# Número de rotas por par OD
K_ROUTES = 3