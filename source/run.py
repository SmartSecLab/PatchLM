import torch

import source.utility as util
from source.codellama import CodeLlamaModel
from source.codet5 import CodeT5Model

dash_line = "=" * 50

# Setup logger
log = util.get_logger()
config = util.load_config()
log.info(dash_line)
log.info(f"Logging at: {util.log_filename}")
log.info(f"Config: {config}")
log.info(dash_line)


# Clears the memory cache
torch.cuda.empty_cache()

# Optionally, you can also use this to clear the CUDA memory allocator
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()


config["device"] = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu").type

log.info(f"Using available device: {config['device']}")

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # Load the CodeLlama model or the CodeT5 model
    if 'codellama' in str(config['base_model']).lower():
        config['model_type'] = 'codellama'
        code_llama = CodeLlamaModel(config, log)
        code_llama.run_codellama()

    elif 'codet5' in str(config['base_model']).lower():
        config['model_type'] = 'codet5'
        code_t5 = CodeT5Model(config, log)
        code_t5.run_codet5()
    else:
        raise ValueError(f"Invalid model name: {config['base_model']}")

    log.info(dash_line)
    log.info(f"Model loaded successfully: {config['base_model']}")
    log.info(dash_line)
