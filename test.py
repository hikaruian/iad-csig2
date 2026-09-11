from src.model import build_model

model = build_model(
    encoder_name="dinov2reg_vit_large_14",
    inp_num=12,
    decoder_depth=8,
    residual_strength=0.20,
    encoder_source="auto",
)

state = model.state_dict()

for k in [
    "encoder.model.reg_token",
    "encoder.model.register_tokens",
    "encoder.model.mask_token",
    "encoder.model.pos_embed",
]:
    if k in state:
        print(k, tuple(state[k].shape))
    else:
        print(k, "MISSING")

