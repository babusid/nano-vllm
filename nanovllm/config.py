import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # Speculative decoding specific configs
    speculative_method: str = "medusa"     # TODO: add eagle  
    num_speculative_tokens: int = 4    
    medusa_num_heads: int = 0
    medusa_num_layers: int = 1         
    medusa_model_path: str = ""        

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        
        # medusa_num_heads takes priority if defined
        if self.medusa_num_heads == 0:
            self.medusa_num_heads = self.num_speculative_tokens
        else:
            self.num_speculative_tokens = self.medusa_num_heads
        if self.speculative_method:
            assert self.speculative_method in ("medusa",), \
                f"Unknown speculative_method '{self.speculative_method}'. Valid: 'medusa'"
