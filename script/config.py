import transformers
import torch

# Paths to datasets and model files
RAW_DATASET_PATH = '../nlu_dataset/nlu.csv'
ENTITY_RECOGNITION_DATASET_PATH  = '../nlu_dataset/er_dataset.csv'
INTENT_CLASSIFICATION_DATASET_PATH  = '../nlu_dataset/is_dataset.csv'
MODEL_SAVE_PATH = '/Volumes/My Passport 1/models/epoch50_best_model.pth' 
TRACE_MODEL_SAVE_PATH = '/Volumes/My Passport 1/models/epoch50_best_model_trace.pth' 
LOG_DIRECTORY = '/home2/ncwn67/A-Hackers-AI-Voice-Assistant/VoiceAssistant/nlu/tb_logs/run4'

# Additional Parameters
SAVE_MODEL_FLAG = True

# Hyperparameters
MAX_SEQUENCE_LENGTH = 63
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
NUMBER_OF_EPOCHS = 50

# Model selection
MODEL_NAME = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    MODEL_NAME,
    do_lower_case=True
)

# Device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
