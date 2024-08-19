DATA_SET = "CoLA"
DATA_PATH = "/home/aiservice/workspace/questionB/pre_data/%s/%s_train.json"%(DATA_SET, DATA_SET)
VAL_SET_SIZE = 1600
MODEL = "gemma"

MAX_LENGTH_Q = 374 - 1 
MAX_LENGTH_A = 10 - 1 
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 2 # 384

MODEL_PATH = "/home/aiservice/workspace/data/questionBData/model/Lucachen/gemma2b"