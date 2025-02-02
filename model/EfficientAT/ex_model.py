import torch
from thop import profile
from torchsummary import summary
from models.mn.model import get_model as get_mn
from models.dymn.model import get_model as get_dymn


if __name__ == "__main__":
    x = torch.randn([8,1,64,100])
    model = get_mn(pretrained_name="mn10_as",num_classes=10)
    x = torch.randn([8, 1, 64, 100])
    x = model(x)
    print(x.shape)
    #print(summary(model,(1,64,100),device="cpu"))

    flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
    # 输出 FLOPs 和参数数量，增加描述文字
    print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")


    model = get_dymn(pretrained_name="dymn10_as",num_classes=10)
    x = torch.randn([8, 1, 64, 100])
    x = model(x)
    print(x.shape)
    #print(summary(model,(1,64,100),device="cpu"))
    flops, params = profile(model, inputs=(torch.randn(1, 1,64,100),),verbose=False)
    # 输出 FLOPs 和参数数量，增加描述文字
    print(f"FLOPs: {flops / 1e9:.2f} GFlops Parameters: {params / 1e6:.2f} M")