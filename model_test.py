import os
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
import math
import cv2
from PIL import Image
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import time
import torch
import sys
from torch.utils.data import DataLoader
import argparse
import warnings
from data_loader import AudioDataset,AudioDataset_Feature,AudioDataset_test
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import runner
from model import losses
from model import metrics
from model import resnet_se,EfficientNet,tdnn,mobilenetv4,ast_models,campplus,htsat,eres2net,panns,res2net,ecapa_tdnn,soundnet
sys.path.append("model/DTFAT/src")
sys.path.append("model/EfficientAT")
from models.mn.model import get_model as get_mn
from models.dymn.model import get_model as get_dymn
from model.DTFAT.src import the_new_audio_model
from augment import parse_transforms
from thop import profile
from torchsummary import summary
from easydict import EasyDict


def model_test(args, model, device):

    testset = AudioDataset_test(args.train_list, num_classes=args.classes,audio_path=args.audio_path,fre_len=args.fre_len)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)
    with open(args.label_list_path) as f:
        labels = f.readlines()
    id2label = {i:label.strip() for i,label in enumerate(labels)}
    print(id2label)
    for data,wavpath,num in test_loader:
        for i in range(num):
            x = data[:,:,:,args.fre_len*i:args.fre_len*(i+1)].to(device)
            output = model(x)
            result = torch.nn.functional.softmax(output, dim=-1)[0]
            result = result.data.cpu().numpy()
            # 最大概率的label
            lab = np.argsort(result)[-1]
            score = result[lab]
            label = id2label[lab],
            score = round(float(score), 5)
            # return self.class_labels[lab], round(float(score), 5)
            print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model_type = args.model_type  # mn10_as   dymn10_as

    if model_type == "ResNetSE":
        model = resnet_se.ResNetSE(num_class=10, input_size=64)
    elif model_type == "ResNetSE_GRU":
        model = resnet_se.ResNetSE_GRU(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")
    elif model_type == "EffNetAttention":
        model = EfficientNet.EffNetAttention(pretrain=False, b=2, head_num=1, label_dim=10)
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")
        params = 0
        for p in model.parameters():
            if p.requires_grad:
                params += p.numel()
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")
    elif model_type == "TDNN":
        model = tdnn.TDNN(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")
    elif model_type == "TDNN_GRU_SE":
        model = tdnn.TDNN_GRU_SE(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "MobileNetv4":
        model = mobilenetv4.mobilenetv4_conv_small(num_classes=10, c=1)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "DTFAT":
        model = the_new_audio_model.get_timm_pretrained_model(n_classes=10, imgnet=True)
        print(summary(model, (1, 256, 128), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 256, 128),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "CAMPPlus":
        model = campplus.CAMPPlus(num_class=10, input_size=64)
        print(summary(model, (1, 64, 128), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 128),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "AST-model":
        model = ast_models.ASTModel(input_tdim=64, label_dim=10)
        print(summary(model, (1, 64, 128), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 128),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "htsat":
        # for htsat hyperparamater
        config = EasyDict({"htsat_window_size": 8,
                           "htsat_spec_size": 256,
                           "htsat_patch_size": 4,
                           "htsat_stride": (4, 4),
                           "htsat_num_head": [4, 8, 16, 32],
                           "htsat_dim": 96,
                           "htsat_depth": [2, 2, 6, 2],
                           "classes_num": 10,
                           "window_size": 1024,
                           "hop_size": 320,
                           "sample_rate": 16000,
                           "mel_bins": 100,
                           "fmin": 50,
                           "fmax": 14000,
                           "enable_tscam": True,
                           "enable_repeat_mode": False,
                           "htsat_attn_heatmap": False,
                           "loss_type": "clip_bce"
                           })

        model = htsat.HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config=config,
            depths=config.htsat_depth,
            embed_dim=config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )

    elif model_type == "ERes2Net":

        model = eres2net.ERes2Net(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "ERes2NetV2":

        model = eres2net.ERes2NetV2(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "PANNS_CNN6":
        model = panns.PANNS_CNN6(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "PANNS_CNN10":
        model = panns.PANNS_CNN10(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "PANNS_CNN14":
        model = panns.PANNS_CNN14(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "Res2Net":
        model = res2net.Res2Net(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "EcapaTdnn":
        model = ecapa_tdnn.EcapaTdnn(num_class=10, input_size=64)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "mn10_as":

        model = get_mn(pretrained_name=model_type, num_classes=10)
        x = torch.randn([8, 1, 64, 100])
        x = model(x)[0]
        print(x.shape)
        print(summary(model, (1, 64, 100), device="cpu"))

        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "dymn10_as":
        model = get_dymn(pretrained_name=model_type, num_classes=10)
        print(summary(model, (1, 64, 100), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "EAT-M-Transformer":
        model = soundnet.SoundNetRaw(nf=32,
                                     dim_feedforward=2048,
                                     clip_length=100,
                                     embed_dim=64,
                                     n_layers=6,
                                     nhead=16,
                                     n_classes=10,
                                     factors=[4, 4, 4, 4],
                                     )
        # print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    else:
        print("This model is not supported, please check and confirm")
    criterion = losses.BCELoss()
    model_name = f"Audio_classification_{model.name}_s{args.seed}_{criterion.name}"
    model_path = os.path.join(args.save_model, f"{model_name}.pth")
    if not  os.path.isfile(model_path):
        print("The pre trained model does not exist")
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    # test model
    model_test(args, model, device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--seed', default=2025)
    parser.add_argument('--audio_path', default="example_audio/7383-3-0-1.wav")
    parser.add_argument('--model_type', default="EAT-M-Transformer",
                        choices=["TDNN", "TDNN_GRU_SE", "ResNetSE", "ResNetSE_GRU", "EffilecentNet_B2", \
                                 "MobileNetV4", "DTFAT", "AST", "CAMPPlus", "ERes2Net", "ERes2NetV2", "PANNS_CNN6",
                                 "PANNS_CNN10", \
                                 "PANNS_CNN14", "Res2Net", "EcapaTdnn", "HTS-AT", "mn10_as", "dymn10_as",
                                 "EAT-M-Transformer"])
    parser.add_argument('--save_model', default="model")
    parser.add_argument('--test_batch_size', default=1)
    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--classes', default=10)
    parser.add_argument('--train_list', default="dataset/train_list.txt")
    parser.add_argument('--test_list', default="dataset/test_list.txt")
    parser.add_argument('--fre_len', default=100)
    parser.add_argument('--label_list_path', default="dataset/label_list.txt")
    args = parser.parse_args()

    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)


