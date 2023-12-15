import torch

model = torch.load("./checkpoint_math/checkpoint-4.pth", map_location="cpu")
new_model = dict()
weight_list = ["layers." + str(i) + ".attention.gate" for i in range(32)]
old_weight_list = ["layers." + str(i) + ".attention.gate" for i in range(32)]

w1_bias_weight_list = ["layers." + str(i) + ".feed_forward.w1_bias" for i in range(32)]
w2_bias_weight_list = ["layers." + str(i) + ".feed_forward.w2_bias" for i in range(32)]
w3_bias_weight_list = ["layers." + str(i) + ".feed_forward.w3_bias" for i in range(32)]
wk_bias_weight_list = ["layers." + str(i) + ".attention.wk_bias" for i in range(32)]
wv_bias_weight_list = ["layers." + str(i) + ".attention.wv_bias" for i in range(32)]
wo_bias_weight_list = ["layers." + str(i) + ".attention.wo_bias" for i in range(32)]
attention_norm_weight_list = ["layers." + str(i) + ".attention_norm.weight" for i in range(32)]
ffn_norm_weight_list = ["layers." + str(i) + ".ffn_norm.weight" for i in range(32)]

weight_list = weight_list + w1_bias_weight_list + w2_bias_weight_list + w3_bias_weight_list + \
    wk_bias_weight_list + wv_bias_weight_list + wo_bias_weight_list + attention_norm_weight_list + ffn_norm_weight_list + ["adapter_query.weight"]

print(weight_list)
print(model["model"]["adapter_query.weight"].shape)

for i in range(len(weight_list)):
    new_model[weight_list[i]] = model["model"][weight_list[i]]

torch.save(new_model, "../ckpt/adapter_adapter_len10_layer30_epoch5_bt_bias_norm.pth")
