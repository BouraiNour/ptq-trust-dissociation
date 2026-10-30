import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LETTERS = "ABCD"

def load_model_and_tokenizer(model_id, quantization, device_map, base_model_id="Qwen/Qwen2.5-7B-Instruct"):
    tok = AutoTokenizer.from_pretrained(base_model_id)
    
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True, "device_map": device_map}
    
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_compute_dtype=torch.float16
        )
        
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    
    # Extract single-token IDs for A, B, C, D (with and without spaces/case)
    letter_ids = {
        L: sorted(list(set([
            tok.encode(s, add_special_tokens=False)[0] 
            for s in (L, " "+L, L.lower(), " "+L.lower()) 
            if len(tok.encode(s, add_special_tokens=False)) == 1
        ]))) 
        for L in LETTERS
    }
    
    return model, tok, letter_ids
