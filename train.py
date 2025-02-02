import numpy as np
import time
import torch
import sys
from torch.utils.data import DataLoader
import argparse
import warnings
from data_loader import AudioDataset,AudioDataset_Feature
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

warnings.filterwarnings("ignore")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def data_loader(args):
    transform = parse_transforms(["freqmask", "timemask", "shift"])
    if os.path.isfile("features/Urbansound8K_train.h5"):
        trainset = AudioDataset_Feature(args.train_list,feature_map_list="features/Urbansound8K_train.h5",mode='train',num_classes=args.classes,transform=None)
        validset = AudioDataset_Feature(args.test_list,feature_map_list="features/Urbansound8K_test.h5", mode='test',num_classes=args.classes)
    else:
        trainset = AudioDataset(args.train_list, mode='train', num_classes=args.classes)
        validset = AudioDataset(args.test_list, mode='test', num_classes=args.classes)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(validset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, valid_loader

    
def train_model(args, model, optimizer, criterion, metric, device):
    # get dataset loaders
    train_data_loader, val_data_loader = data_loader(args)
    
    # create folder to save model
    os.makedirs(args.save_model, exist_ok=True)
    model_name = f"Audio_classification_{model.name}_s{args.seed}_{criterion.name}"

    max_score = 0
    train_hist = []
    valid_hist = []

    warmup_lr = 0.00001
    cosine_max_lr = 0.2
    warmup_epochs = 5
    cosine_epochs = 50
    total_epochs = warmup_epochs + cosine_epochs

    # 定义学习率调度器，结合 warm-up 和余弦退火
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=0)

    for epoch in range(args.n_epochs):
        # 更新学习率
        if epoch < warmup_epochs:
            new_lr = warmup_lr + (args.learning_rate - warmup_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            lr_scheduler.step()
        print(f"\nEpoch: {epoch + 1},lr ={lr_scheduler.get_lr()[0]}")

        logs_train = runner.train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            dataloader=train_data_loader,
            device=device,
        )

        logs_valid = runner.valid_epoch(
            model=model,
            criterion=criterion,
            metric=metric,
            dataloader=val_data_loader,
            device=device,
        )

        train_hist.append(logs_train)
        valid_hist.append(logs_valid)
        score = logs_valid[metric.name[0]]

        if max_score < score:
            max_score = score
            torch.save(model.state_dict(), os.path.join(args.save_model, f"{model_name}.pth"))
            print("Model saved in the folder : ", args.save_model)
            print("Model name is : ", model_name)
        # torch.save(model.state_dict(), os.path.join(args.save_model, f"{model_name}_{epoch}.pth"))
        # print("Model saved in the folder : ", args.save_model)
        # print("Model name is : ", model_name)
     
            
def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_type = args.model_type   # mn10_as   dymn10_as

    if model_type == "ResNetSE":
        model = resnet_se.ResNetSE(num_class=10,input_size=64)
    elif model_type == "ResNetSE_GRU":
        model = resnet_se.ResNetSE_GRU(num_class=10, input_size=64)
        print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
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
        print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")
    elif model_type == "TDNN_GRU_SE":
        model = tdnn.TDNN_GRU_SE(num_class=10, input_size=64)
        print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "MobileNetv4":
        model = mobilenetv4.mobilenetv4_conv_small(num_classes=10,c = 1)
        print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "DTFAT":
        model = the_new_audio_model.get_timm_pretrained_model(n_classes=10, imgnet=True)
        print(summary(model, (1, 256, 128), device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 256, 128),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "CAMPPlus":
        model = campplus.CAMPPlus(num_class=10,input_size=64)
        print(summary(model,(1,64,128),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1,64,128),),verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "AST-model":
        model = ast_models.ASTModel(input_tdim=64,label_dim=10)
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
        print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "PANNS_CNN10":
        model = panns.PANNS_CNN10(num_class=10, input_size=64)
        print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "PANNS_CNN14":
        model = panns.PANNS_CNN14(num_class=10, input_size=64)
        print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1, 64, 100),), verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "Res2Net":
        model = res2net.Res2Net(num_class=10, input_size=64)
        print(summary(model,(1,64,100),device="cpu"))
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

        model = get_mn(pretrained_name=model_type,num_classes=10)
        x = torch.randn([8, 1, 64, 100])
        x = model(x)[0]
        print(x.shape)
        print(summary(model,(1,64,100),device="cpu"))

        flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")

    elif model_type == "dymn10_as":
        model = get_dymn(pretrained_name=model_type,num_classes=10)
        print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
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
        #print(summary(model,(1,64,100),device="cpu"))
        flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
        # 输出 FLOPs 和参数数量，增加描述文字
        print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")


    # count parameters
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters: ", params)

    criterion = losses.BCELoss()
    metric = metrics.Accuracy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(
            [dict(params=model.module.parameters(), lr=args.learning_rate)]
        )
    
    print("Number of epochs   :", args.n_epochs)
    print("Number of classes  :", args.classes)
    print("Batch size         :", args.batch_size)
    print("Device             :", device)
               
    # training model
    train_model(args, model, optimizer, criterion, metric, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--seed', default=2025)
    parser.add_argument('--n_epochs', default=200)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--test_batch_size', default=64)
    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--model_type', default="EAT-M-Transformer",choices=["TDNN","TDNN_GRU_SE","ResNetSE","ResNetSE_GRU","EffilecentNet_B2",\
                                                                      "MobileNetV4","DTFAT","AST","CAMPPlus","ERes2Net","ERes2NetV2","PANNS_CNN6","PANNS_CNN10",\
                                                                      "PANNS_CNN14","Res2Net","EcapaTdnn","HTS-AT","mn10_as","dymn10_as","EAT-M-Transformer"])
    parser.add_argument('--learning_rate', default=0.0001)  
    parser.add_argument('--classes', default=10)
    parser.add_argument('--train_list', default="dataset/train_list.txt")
    parser.add_argument('--test_list', default="dataset/test_list.txt")
    parser.add_argument('--save_model', default="model") 
    parser.add_argument('--save_results', default="results")
    args = parser.parse_args()
    
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
