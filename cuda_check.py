import torch

if __name__ == '__main__':
    print(f"cuda is available: {torch.cuda.is_available()}")
    print(torch.randn(1).cuda())
