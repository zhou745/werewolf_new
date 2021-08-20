import torch
import torch.nn.functional as F
import numpy as np

def main():
    #size N d
    N = 1024
    d = 16

    M = 1024*1024
    sample = 1

    device = 0 if torch.cuda.is_available() else torch.device('cpu')

    A = np.random.uniform(-10,10.,(N,d))
    B = np.random.uniform(-10, 10., (N, d))

    A_cuda = torch.tensor(A,dtype = torch.float32).to(device)
    B_cuda = torch.tensor(B,dtype = torch.float32).to(device)
    # B_cuda = A_cuda
    #true atten value
    Atten_softmax = F.softmax(torch.matmul(A_cuda,B_cuda.transpose(0,1))/np.sqrt(d),dim=-1)

    #compute with kernel functions
    kernal_w = []
    kernal_b = []

    sigma_square = np.sqrt(d)

    for iter in range(sample):
        kernal_w.append(np.random.normal(0.,sigma_square,(M,d)))
        kernal_b.append(np.random.uniform(0., 2*np.pi, (M, 1)))

    kernel_w_cuda = torch.tensor(kernal_w,dtype= torch.float32).to(device)
    kernel_b_cuda = torch.tensor(kernal_b,dtype = torch.float32).to(device)


    Q_value =  torch.matmul(kernel_w_cuda,A_cuda.transpose(0,1)) + kernel_b_cuda
    K_value = torch.matmul(kernel_w_cuda, B_cuda.transpose(0, 1)) + kernel_b_cuda

    Q_kernel = (torch.cos(Q_value)*np.sqrt(2/M)).view(-1,N)
    K_kernel = (torch.cos(K_value) * np.sqrt(2 / M)).view(-1,N)

    Q_norm = torch.diag(A_cuda.norm(dim=1)/(2*np.sqrt(d)))
    K_norm = torch.diag(B_cuda.norm(dim=1) /(2 * np.sqrt(d)))

    Atten_kenel = torch.matmul(Q_kernel.transpose(0,1),K_kernel)
    Atten_kenel = torch.matmul(Q_norm,Atten_kenel)
    Atten_kenel = torch.matmul(Atten_kenel,K_norm)

    D = torch.diag(1/Atten_kenel.sum(dim=-1))
    Atten_kenel = torch.matmul(D ,Atten_kenel)

    abs_value = torch.abs(Atten_softmax)+1e-7
    print(Atten_softmax)
    print(Atten_kenel)
    print((Atten_softmax-Atten_kenel)/abs_value)
    print(Atten_softmax-Atten_kenel)

if __name__ == "__main__":
    main()