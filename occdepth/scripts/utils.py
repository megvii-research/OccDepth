def load_pretrain_model(model, params):
    filter_params = {}
    for k, v in params.items():
        if "backbone.net_rgb" in k:
            filter_params[k.replace("backbone.", "")] = v
    print("Load Pretrain model keys:", list(filter_params.keys()))
    model.load_state_dict(filter_params, strict=False)
    return model
