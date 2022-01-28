from .cls_hrnet import get_cls_net

def cls_net(cfg, **kwargs):
    from .default import _C as config
    config.defrost()
    config.merge_from_file(cfg)
    config.freeze()
    
    return get_cls_net(config, **kwargs)
