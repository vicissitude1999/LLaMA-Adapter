torchrun --nproc_per_node 1 example.py \
         --ckpt_dir models/LLaMA-7B/ \
         --tokenizer_path models/LLaMA-7B/tokenizer.model \
         --adapter_path ckpt/adapter_adapter_len10_layer30_epoch5_alpaca_math.pth