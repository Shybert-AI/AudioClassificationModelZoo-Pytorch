import dataclasses
import logging
import os
from copy import deepcopy
from typing import Optional, Dict, Callable, Any, Tuple

from torch import nn as nn
from torch.hub import load_state_dict_from_url

from audio_model_timm.models._features import FeatureListNet, FeatureHookNet
from audio_model_timm.models._features_fx import FeatureGraphNet
from audio_model_timm.models._helpers import load_state_dict
from audio_model_timm.models._hub import has_hf_hub, download_cached_file, check_cached_file, load_state_dict_from_hf
from audio_model_timm.models._manipulate import adapt_input_conv
from audio_model_timm.models._pretrained import PretrainedCfg
from audio_model_timm.models._prune import adapt_model_from_file
from audio_model_timm.models._registry import get_pretrained_cfg

_logger = logging.getLogger(__name__)

# Global variables for rarely used pretrained checkpoint download progress and hash check.
# Use set_pretrained_download_progress / set_pretrained_check_hash functions to toggle.
_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False


__all__ = ['set_pretrained_download_progress', 'set_pretrained_check_hash', 'load_custom_pretrained', 'load_pretrained',
           'pretrained_cfg_for_features', 'resolve_pretrained_cfg', 'build_model_with_cfg']


def _resolve_pretrained_source(pretrained_cfg):
    cfg_source = pretrained_cfg.get('source', '')
    pretrained_url = pretrained_cfg.get('url', None)
    pretrained_file = pretrained_cfg.get('file', None)
    hf_hub_id = pretrained_cfg.get('hf_hub_id', None)

    # resolve where to load pretrained weights from
    load_from = ''
    pretrained_loc = ''
    if cfg_source == 'hf-hub' and has_hf_hub(necessary=True):
        # hf-hub specified as source via model identifier
        load_from = 'hf-hub'
        assert hf_hub_id
        pretrained_loc = hf_hub_id
    else:
        # default source == timm or unspecified
        if pretrained_file:
            # file load override is the highest priority if set
            load_from = 'file'
            pretrained_loc = pretrained_file
        else:
            # next, HF hub is prioritized unless a valid cached version of weights exists already
            cached_url_valid = check_cached_file(pretrained_url) if pretrained_url else False
            if hf_hub_id and has_hf_hub(necessary=True) and not cached_url_valid:
                # hf-hub available as alternate weight source in default_cfg
                load_from = 'hf-hub'
                pretrained_loc = hf_hub_id
            elif pretrained_url:
                load_from = 'url'
                pretrained_loc = pretrained_url

    if load_from == 'hf-hub' and pretrained_cfg.get('hf_hub_filename', None):
        # if a filename override is set, return tuple for location w/ (hub_id, filename)
        pretrained_loc = pretrained_loc, pretrained_cfg['hf_hub_filename']
    return load_from, pretrained_loc


def set_pretrained_download_progress(enable=True):
    """ Set download progress for pretrained weights on/off (globally). """
    global _DOWNLOAD_PROGRESS
    _DOWNLOAD_PROGRESS = enable


def set_pretrained_check_hash(enable=True):
    """ Set hash checking for pretrained weights on/off (globally). """
    global _CHECK_HASH
    _CHECK_HASH = enable


def load_custom_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        load_fn: Optional[Callable] = None,
):
    r"""Loads a custom (read non .pth) weight file

    Downloads checkpoint file into cache-dir like torch.hub based loaders, but calls
    a passed in custom load fun, or the `load_pretrained` model member fn.

    If the object is already present in `model_dir`, it's deserialized and returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        model: The instantiated model to load weights into
        pretrained_cfg (dict): Default pretrained model cfg
        load_fn: An external standalone fn that loads weights into provided model, otherwise a fn named
            'laod_pretrained' on the model will be called if it exists
    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        _logger.warning("Invalid pretrained config, cannot load weights.")
        return

    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if not load_from:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    if load_from == 'hf-hub':  # FIXME
        _logger.warning("Hugging Face hub not currently supported for custom load pretrained models.")
    elif load_from == 'url':
        pretrained_loc = download_cached_file(
            pretrained_loc,
            check_hash=_CHECK_HASH,
            progress=_DOWNLOAD_PROGRESS,
        )

    if load_fn is not None:
        load_fn(model, pretrained_loc)
    elif hasattr(model, 'load_pretrained'):
        model.load_pretrained(pretrained_loc)
    else:
        _logger.warning("Valid function to load pretrained weights is not available, using random initialization.")


def load_pretrained_apr22(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        num_classes: int = 1000,
        in_chans: int = 3,
        filter_fn: Optional[Callable] = None,
        strict: bool = True,
):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        pretrained_cfg (Optional[Dict]): configuration for pretrained weights / target dataset
        num_classes (int): num_classes for target model
        in_chans (int): in_chans for target model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint

    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        _logger.warning("Invalid pretrained config, cannot load weights.")
        return

    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    print("\n load_from ", load_from, pretrained_loc)
    if load_from == 'file':
        _logger.info(f'Loading pretrained weights from file ({pretrained_loc})')
        state_dict = load_state_dict(pretrained_loc)
    elif load_from == 'url':
        _logger.info(f'Loading pretrained weights from url ({pretrained_loc})')
        if pretrained_cfg.get('custom_load', False):
            pretrained_loc = download_cached_file(
                pretrained_loc,
                progress=_DOWNLOAD_PROGRESS,
                check_hash=_CHECK_HASH,
            )
            model.load_pretrained(pretrained_loc)
            return
        else:
            state_dict = load_state_dict_from_url(
                pretrained_loc,
                map_location='cpu',
                progress=_DOWNLOAD_PROGRESS,
                check_hash=_CHECK_HASH,
            )
    elif load_from == 'hf-hub':
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        if isinstance(pretrained_loc, (list, tuple)):
            state_dict = load_state_dict_from_hf(*pretrained_loc)
        else:
            state_dict = load_state_dict_from_hf(pretrained_loc) ## linear attn(attencl format)
            # print([(k,v.shape) for k,v in state_dict.items() if 'qkv' in k][:4]) ## attentioncl here
    else:
        _logger.warning("No pretrained weights exist or were found for this model. Using random initialization.")
        return

    if filter_fn is not None:
        try:
            state_dict = filter_fn(state_dict, model)
        except TypeError as e:
            # for backwards compat with filter fn that take one arg
            state_dict = filter_fn(state_dict) ## change liner attn weight to  conv atten weight

    input_convs = pretrained_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3: ## not using this because im explcitly doing it later
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = pretrained_cfg.get('classifier', None)
    label_offset = pretrained_cfg.get('label_offset', 0)
    
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                state_dict.pop(classifier_name + '.weight', None)
                state_dict.pop(classifier_name + '.bias', None)
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]


    ### stem to 1 channel  
    import torch
    import copy
    state_dict_new = copy.deepcopy(state_dict)


    ### atten2d vs attencl which is usd in pretrained weight

    # exit('donr')


    ### conv filter 3 channel to 1 channel

    for k,v in state_dict_new.items():
        if 'stem' in k:
            if v.shape == (64, 3, 3, 3):
                state_dict[k] = torch.mean(v,dim=1, keepdim=True)

    # ### time freq blcoks
    # state_dict_new1 = copy.deepcopy(state_dict)

    # for k,v in state_dict.items():
    #     if '.attn_block.' in k:
    #         old_k_t = '.attn_block_time.'.join(k.split('.attn_block.'))
    #         old_k_f = '.attn_block_freq.'.join(k.split('.attn_block.'))

    #         print(f"{k} replaced with {old_k_t,old_k_f}")

    #         state_dict_new1[old_k_t] = v
    #         state_dict_new1[old_k_f] = v
    #         state_dict_new1.pop(k,None)

    # state_dict = state_dict_new1
    ### time freq conv blcoks in stem
    RANDOM_INIT_CONV = False
    state_dict_new1 = copy.deepcopy(state_dict)

    for k,v in state_dict.items():
        if 'stem.' in k:

            if 'stem.conv1.' in k:
                old_k_t = '.conv1_t.'.join(k.split('.conv1.'))
                old_k_f = '.conv1_f.'.join(k.split('.conv1.'))
            elif 'stem.conv2.' in k:
                old_k_t = '.conv2_t.'.join(k.split('.conv2.'))
                old_k_f = '.conv2_f.'.join(k.split('.conv2.'))
            elif 'stem.norm1.' in k:
                old_k_t = '.norm1_t.'.join(k.split('.norm1.'))
                old_k_f = '.norm1_f.'.join(k.split('.norm1.'))
            elif 'stem.t_f_weight' in k:
                continue ## already initialized
            else:
                exit(f"error key {k}")
            if not old_k_f in model.state_dict().keys(): 
                print(f"error {old_k_f} not in model")
                exit('error')
            ## reshaping kernal
            v_t=copy.deepcopy(v)
            v_f=copy.deepcopy(v)
            v_t_shape = model.state_dict()[old_k_t].shape[-2:]

            if v_t_shape==v.shape[-2:]:
                state_dict_new1[old_k_t] = v_t
                state_dict_new1[old_k_f] = v_f
                continue

            v_t = nn.functional.interpolate(v_t, size=v_t_shape, mode='bilinear', align_corners=False)

            v_f_shape = model.state_dict()[old_k_f].shape[-2:]
            v_f = nn.functional.interpolate(v_f, size=v_f_shape, mode='bilinear', align_corners=False)

            if RANDOM_INIT_CONV:
                print(f"\nRunning with for Stem RANDOM_INIT_CONV {RANDOM_INIT_CONV}\n")
                v_t = model.state_dict()[old_k_t]
                v_t = nn.init.xavier_uniform_(v_t)

                v_f = model.state_dict()[old_k_f]
                v_f = nn.init.xavier_uniform_(v_f)

            state_dict_new1[old_k_t] = v_t
            state_dict_new1[old_k_f] = v_f
    state_dict = state_dict_new1
    #######################


 
    ### time freq conv blcoks in MBConv
    RANDOM_INIT_CONV=False
    state_dict_new1 = copy.deepcopy(state_dict)

    for k,v in state_dict.items():
            
        if '.conv2_kxk.' in k:
            old_k_t = '.conv2_kxk_t.'.join(k.split('.conv2_kxk.'))
            old_k_f = '.conv2_kxk_f.'.join(k.split('.conv2_kxk.'))


            if not old_k_f in model.state_dict().keys():
                exit('no such case')

            ## reshaping kernal
            v_t=copy.deepcopy(v)
            v_f=copy.deepcopy(v)
            v_t_shape = model.state_dict()[old_k_t].shape[-2:]

            if v_t_shape==v.shape[-2:]: ## no change after t f reconfig eg bias
                state_dict_new1[old_k_t] = v_t
                state_dict_new1[old_k_f] = v_f
                continue

            v_t_shape = model.state_dict()[old_k_t].shape[-2:]
            v_t = nn.functional.interpolate(v_t, size=v_t_shape, mode='bilinear', align_corners=False)

            v_f_shape = model.state_dict()[old_k_f].shape[-2:]
            v_f = nn.functional.interpolate(v_f, size=v_f_shape, mode='bilinear', align_corners=False)
            
            if RANDOM_INIT_CONV:
                print(f"\nRunning with for MBConv RANDOM_INIT_CONV {RANDOM_INIT_CONV}\n")
                v_t = model.state_dict()[old_k_t]
                v_t = nn.init.xavier_uniform_(v_t)

                v_f = model.state_dict()[old_k_f]
                v_f = nn.init.xavier_uniform_(v_f)

            state_dict_new1[old_k_t] = v_t
            state_dict_new1[old_k_f] = v_f


    state_dict = state_dict_new1

    #######################
    #######################
    ### add t_f_weight
    state_dict_new1 = copy.deepcopy(state_dict)

    for k,v in model.state_dict().items():
        if 't_f_weight' in k:
            state_dict[k]=torch.tensor(0.5)

    relative_pos_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
    for k in relative_pos_keys:
        state_dict.pop(k,None)

    for k in model.state_dict().keys():
        if k not in state_dict.keys():
            if (not 'relative_position_bias_table' in k)&(not 'head.fc' in k):
                exit(f" some key {k} other than relative pos head.fc missing")
    model.load_state_dict(state_dict, strict=strict)

def load_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        num_classes: int = 1000,
        in_chans: int = 3,
        filter_fn: Optional[Callable] = None,
        strict: bool = True,
):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        pretrained_cfg (Optional[Dict]): configuration for pretrained weights / target dataset
        num_classes (int): num_classes for target model
        in_chans (int): in_chans for target model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint

    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        _logger.warning("Invalid pretrained config, cannot load weights.")
        return

    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if load_from == 'file':
        _logger.info(f'Loading pretrained weights from file ({pretrained_loc})')
        state_dict = load_state_dict(pretrained_loc)
    elif load_from == 'url':
        _logger.info(f'Loading pretrained weights from url ({pretrained_loc})')
        if pretrained_cfg.get('custom_load', False):
            pretrained_loc = download_cached_file(
                pretrained_loc,
                progress=_DOWNLOAD_PROGRESS,
                check_hash=_CHECK_HASH,
            )
            model.load_pretrained(pretrained_loc)
            return
        else:
            state_dict = load_state_dict_from_url(
                pretrained_loc,
                map_location='cpu',
                progress=_DOWNLOAD_PROGRESS,
                check_hash=_CHECK_HASH,
            )
    elif load_from == 'hf-hub':
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        if isinstance(pretrained_loc, (list, tuple)):
            state_dict = load_state_dict_from_hf(*pretrained_loc)
        else:
            state_dict = load_state_dict_from_hf(pretrained_loc)
    else:
        _logger.warning("No pretrained weights exist or were found for this model. Using random initialization.")
        return

    if filter_fn is not None:
        try:
            state_dict = filter_fn(state_dict, model)
        except TypeError as e:
            # for backwards compat with filter fn that take one arg
            state_dict = filter_fn(state_dict)

    input_convs = pretrained_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = pretrained_cfg.get('classifier', None)
    label_offset = pretrained_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                state_dict.pop(classifier_name + '.weight', None)
                state_dict.pop(classifier_name + '.bias', None)
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=strict)


def pretrained_cfg_for_features(pretrained_cfg):
    pretrained_cfg = deepcopy(pretrained_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'classifier', 'global_pool')  # add default final pool size?
    for tr in to_remove:
        pretrained_cfg.pop(tr, None)
    return pretrained_cfg


def _filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model

    Args:
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)

    for n in default_kwarg_names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # pretrained_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])

    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    _filter_kwargs(kwargs, names=kwargs_filter)


def resolve_pretrained_cfg(
        variant: str,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
) -> PretrainedCfg:
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            # pretrained_cfg dict passed as arg, validate by converting to PretrainedCfg
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None

    # fallback to looking up pretrained cfg in model registry by variant identifier
    if not pretrained_cfg:
        if pretrained_tag:
            model_with_tag = '.'.join([variant, pretrained_tag])
        pretrained_cfg = get_pretrained_cfg(model_with_tag)

    if not pretrained_cfg:
        _logger.warning(
            f"No pretrained configuration specified for {model_with_tag} model. Using a default."
            f" Please add a config to the model pretrained_cfg registry or pass explicitly.")
        pretrained_cfg = PretrainedCfg()  # instance with defaults

    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault('architecture', variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)

    return pretrained_cfg


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        pretrained_cfg: Optional[Dict] = None,
        pretrained_cfg_overlay: Optional[Dict] = None,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[Dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs,
):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        pretrained_cfg (dict): model's pretrained weight/task config
        model_cfg (Optional[Dict]): model's architecture config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        pretrained_strict (bool): load pretrained weights strictly
        pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
        kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay
    )

    # FIXME converting back to dict, PretrainedCfg use should be propagated further, but not into model
    pretrained_cfg = pretrained_cfg.to_dict()

    _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    # Instantiate the model
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
        # print("model_cls",model_cls,"\nmodel fg " ,model_cfg, "\n args ",kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    if pruned:
        model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        load_pretrained(
            model,
            pretrained_cfg=pretrained_cfg,
            num_classes=num_classes_pretrained,
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=pretrained_filter_fn,
            strict=pretrained_strict,
        )
        print("pretrained \n")
        print("num_classes_pretrained ",num_classes_pretrained)
        print("pretrained_cfg ",pretrained_cfg)
        print("pretrained_strict ",pretrained_strict)



    # Wrap the model in a feature extraction module if enabled
    if features:
        feature_cls = FeatureListNet
        output_fmt = getattr(model, 'output_fmt', None)
        if output_fmt is not None:
            feature_cfg.setdefault('output_fmt', output_fmt)
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                elif feature_cls == 'fx':
                    feature_cls = FeatureGraphNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)  # add back pretrained cfg
        model.default_cfg = model.pretrained_cfg  # alias for rename backwards compat (default_cfg -> pretrained_cfg)

    return model
