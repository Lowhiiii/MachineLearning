
from pathlib import Path

HERE = Path(__file__).parent
DATA_PATH = HERE / ".." / ".." / "data"
RAW_DATA_FILE = DATA_PATH / "raw/NF-UNSW-NB15-v2.csv"
PREPROCESSED_DATA_PATH = DATA_PATH / "preprocessed"

TRAIN_DATA_PATH = PREPROCESSED_DATA_PATH / "train.csv"
TEST_DATA_PATH = PREPROCESSED_DATA_PATH / "test.csv"
NUMERICAL_STATS_PATH = PREPROCESSED_DATA_PATH / "numerical_stats.csv"
CATEGORICAL_STATS_PATH = PREPROCESSED_DATA_PATH / "categorical_stats.json"


#AUTOENCODER CONFIG BELOW
BASE_FEATURES = [
    # Chosen according to Top 20 correlated features
    "Label",  # O =benign or 1 = malicious
    "MIN_TTL",  # Min flow TTL
    "MAX_TTL",  # Max flow TTL
    "MIN_IP_PKT_LEN",  # Len of the smallest flow IP packet observed
    "PROTOCOL",  # IP protocol identifier byte
    "SHORTEST_FLOW_PKT",  # Shortest packet (bytes) of the flow
    "SERVER_TCP_FLAGS",  # Cumulative of all server TCP flags
    "TCP_FLAGS",  # Cumulative of all TCP flags
    "CLIENT_TCP_FLAGS",  # Cumulative of all client TCP flags
    "L4_DST_PORT",  # IPv4 destination port number
    "NUM_PKTS_UP_TO_128_BYTES",  # Packets whose IP size <= 128
    "TCP_WIN_MAX_IN",  # Max TCP Window (src->dst)
    "MAX_IP_PKT_LEN",  # Len of the largest flow IP packet observed
    "LONGEST_FLOW_PKT",  # Longest packet (bytes) of the flow
    "DST_TO_SRC_AVG_THROUGHPUT",  # Dst to src average thpt (bps)
    "DNS_QUERY_TYPE",  # DNS query type (e.g., 1=A, 2=NS, etc.)
    "OUT_PKTS",  # Outgoing number of packets
    "L4_SRC_PORT",  # IPv4 source port number
    "FTP_COMMAND_RET_CODE",  # FTP client command return code
    "TCP_WIN_MAX_OUT",  # Max TCP Window (dst->src)

    # Other non-numerical features
    "IPV4_SRC_ADDR",  # IPv4 source address
    "IPV4_DST_ADDR",  # IPv4 destination address
    "Attack",  # Indicator of whether the flow is malicious
    "L7_PROTO"  # Layer 7 protocol (numeric)
]

BASE_CATEGORICAL_FEATURES = [
    "NETWORK_IPV4_SRC_ADDR",
    "HOST_IPV4_SRC_ADDR",
    "NETWORK_IPV4_DST_ADDR",
    "HOST_IPV4_DST_ADDR",
    "DNS_QUERY_TYPE",
    "L7_PROTO",
    "PROTOCOL",
    "L4_DST_PORT",
    "L4_SRC_PORT",
    "SERVER_TCP_FLAGS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "FTP_COMMAND_RET_CODE",
    "ICMP_TYPE",
    "DNS_QUERY_ID",
    "ICMP_IPV4_TYPE"
]



TEST_RATIO = 0.2
VAL_RATIO = 0.2

# AutoEncoder features
BINARY_LABEL_COLUMN = "Label"
MULTICLASS_LABEL_COLUMN = "Attack"
MULTIClASS_CLASS_NAMES= ['Benign', 'Fuzzers', 'Generic', 'Reconnaissance', 'Exploits', 'Analysis', 'Backdoor','DoS', 'Shellcode','Worms']
BINARY_CLASS_NAMES = ['Benign', 'Malicious']

NUMERICAL_FEATURES = [  # List of numerical features to compute stats for
    "MIN_TTL",
    "MAX_TTL",
    "MIN_IP_PKT_LEN",
    "SHORTEST_FLOW_PKT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "TCP_WIN_MAX_IN",
    "MAX_IP_PKT_LEN",
    "LONGEST_FLOW_PKT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "OUT_PKTS",
    "TCP_WIN_MAX_OUT"
]

CATEGORICAL_FEATURES = [  # List of categorical features to compute stats for
    "NETWORK_IPV4_SRC_ADDR",
    "HOST_IPV4_SRC_ADDR",
    "NETWORK_IPV4_DST_ADDR",
    "HOST_IPV4_DST_ADDR",
    "DNS_QUERY_TYPE",
    "L7_PROTO",
    "PROTOCOL",
    # "L4_DST_PORT",
    "CLIP_L4_DST_PORT",
    "CLIP_L4_SRC_PORT",
    # "L4_SRC_PORT",
    "SERVER_TCP_FLAGS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "FTP_COMMAND_RET_CODE",
]

SEED = 42

MAX_PLOT_POINTS = 100000

AUTOENCODER_INPUT_DIMS = 704 #Not calculated but obtained from autoencoder datamodule.py (main)


LOGS_DIR  = HERE / ".." / ".." / "logs"