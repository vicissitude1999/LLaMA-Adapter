import json

import torch
import functools
from llama import ModelArgs, Tokenizer, Transformer


def Llama7B_adapter(args, add_bias, add_scale, train_norm, **kwargs):

    llama_model_path = args.llama_model_path
    model_name = ""

    checkpoint = torch.load(llama_model_path + model_name + "/consolidated.00.pth", map_location="cpu")
    print(llama_model_path + model_name + "/consolidated.00.pth")

    with open(llama_model_path + model_name + "/params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        adapter_len=args.adapter_len,
        adapter_layer=args.adapter_layer,
        add_bias=add_bias,
        add_scale=add_scale,
        **params
    )
    tokenizer = Tokenizer(model_path=llama_model_path + "/tokenizer.model")

    model_args.vocab_size = tokenizer.n_words
    
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_adapter = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model_llama_adapter.load_state_dict(checkpoint, strict=False)
    
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # model_llama_adapter = Transformer(model_args)
    # torch.set_default_tensor_type(torch.FloatTensor)
    # model_llama_adapter.load_state_dict(checkpoint, strict=False)

    # for name, param in model_llama_adapter.named_parameters():
    #     if "adapter" not in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    #         param.data = param.data.float()

    # for name, param in model_llama_adapter.layers[-1 * args.adapter_layer :].named_parameters():
    #     if "gate" in name or "adapter" in name:
    #         param.data = param.data.float()
    #         param.requires_grad = True
    
    
    print(model_args.n_layers)
    # adapter_layer = 0
    # for i in range(model_args.n_layers):
    #     if i < model_args.n_layers - adapter_layer:
    #         del model_llama_adapter.layers[i].attention.gate

    for name, param in model_llama_adapter.named_parameters():
        # requires_grad = (
        #     name.endswith(".gate")
        #     or name == "adapter_query"
        #     or (train_norm and "_norm." in name)
        #     or name.endswith(".added_bias")
        #     or name.endswith(".added_scale")
        # )
        # if requires_grad:
        #     param.data = param.data.float()
        #     param.requires_grad_(True)
        # else:
        #     param.requires_grad_(False)
        
        if "adapter" in name or "bias" in name or "scale" in name or (train_norm and "_norm." in name):
            print(name)
            param.data = param.data.float() # the params to be optimizes are converted to float32, fixed weights are float16
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


    return model_llama_adapter


# set recommended archs
Llama7B_adapter = functools.partial(
    Llama7B_adapter, model_name="", add_bias=True, add_scale=False, train_norm=True
)

