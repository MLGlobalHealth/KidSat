import torch

from . import models_vit_temporal, models_vit
from .util import interpolate_pos_embed


def build_satmae_temporal_finetune(args):
    model = models_vit_temporal.__dict__[args.satmae_type](
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    if args.pretrained_model is not None:
        ckpt = torch.load(args.pretrained_model, map_location="cpu")
        print("Load pre-trained checkpoint from: %s" % args.pretrained_model)
        checkpoint_model = ckpt["model"]
        state_dict = model.state_dict()
    
        # if 'patch_embed.proj.weight' in checkpoint_model and 'patch_embed.proj.weight' in state_dict:
        #     ckpt_patch_embed_weight = checkpoint_model['patch_embed.proj.weight']
        #     model_patch_embed_weight = state_dict['patch_embed.proj.weight']
        #     if ckpt_patch_embed_weight.shape[1] != model_patch_embed_weight.shape[1]:
        #         print('Using 3 channels of ckpt patch_embed')
        #         model.patch_embed.proj.weight.data[:, :3, :, :] = ckpt_patch_embed_weight.data[:, :3, :, :]
    
        # TODO: Do something smarter?
        # checkpoint_model = {k:v for k, v in checkpoint_model if "decoder" not in k}
        for k in [
            "pos_embed",
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "head.weight",
            "head.bias",
        ]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
    
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
    
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    
        try:
            torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)
        except:
            pass

    return model

def build_satmae_finetune(args):
    model = models_vit.__dict__[args.satmae_type](
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    if args.pretrained_model is not None:
        ckpt = torch.load(args.pretrained_model, map_location="cpu")
        print("Load pre-trained checkpoint from: %s" % args.pretrained_model)
        checkpoint_model = ckpt["model"]
        state_dict = model.state_dict()

        # if 'patch_embed.proj.weight' in checkpoint_model and 'patch_embed.proj.weight' in state_dict:
        #     ckpt_patch_embed_weight = checkpoint_model['patch_embed.proj.weight']
        #     model_patch_embed_weight = state_dict['patch_embed.proj.weight']
        #     if ckpt_patch_embed_weight.shape[1] != model_patch_embed_weight.shape[1]:
        #         print('Using 3 channels of ckpt patch_embed')
        #         model.patch_embed.proj.weight.data[:, :3, :, :] = ckpt_patch_embed_weight.data[:, :3, :, :]
    
        # TODO: Do something smarter?
        for k in [
            "pos_embed",
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "head.weight",
            "head.bias",
        ]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
    
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
    
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        try:
            torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)
        except:
            pass

    return model