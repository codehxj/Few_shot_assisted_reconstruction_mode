import  model.recon_normalizaiton_model
import utils.common_utils
import assist_normalization as AN

ANM = AN.AN_model_stru().type(dtype)
ANM.load_state_dict(torch.load('ANM.pth'))

recon_norm_Net = skip(3, 3,num_channels_down=[8, 16, 32, 64, 128], num_channels_up=[8, 16, 32, 64, 128],num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',  need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

optimizerRN = optim.Adam(recon_norm_Net.parameters(), 0.01)
net_input = get_noise(3, 'noise', (128, 128))

Fname = 'data/MultiPIE_test/input/008_01_01_051_03.png'
imgF = Image.open(Fname)
imgF_np = pil_to_np(imgF)
imgF_torch = np_to_torch(imgF_np)


net_input = imgF_torch
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
reg_noise_std = 1. / 30.

out_norm = ANM(imgF_torch)

def closure(k):
    net_input = net_input_saved
    if reg_noise_std > 0:
       net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    out = recon_norm_Net(net_input)
    optimizerRN.zero_grad()
    loss = mse(out_norm,out) + mse(imgF_torch,out)
    loss.backward(retain_variables=True )
    optimizerRN.step()
    print('loss:',loss.item())

    return out



for k in range(10000):
    out = closure()
    if k % 500 == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1), np.clip(imgF_np, 0, 1)], nrow=1)
        torch.save(recon_norm_Net.state_dict(), 'recon_norm_Net.pth')

def closure(net_input_saved,noise,reg_noise_std):
    net_input = net_input_saved
    if reg_noise_std > 0:
       net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    out = netG(net_input).to(torch.device('cuda'))
    out_norm = netSingG(net_input).to(torch.device('cuda'))
    optimizerG.zero_grad()
    #style_loss = mse(out_norm,out)*20 + mse(imgF_torch,out)*80
    style_loss = mse(out_norm,out)*50 + mse(imgF_torch,out)*50
    style_loss.backward(retain_graph=True)
    optimizerG.step()
    return out

dirname='/content/dat/dat/YaleB/'
files = os.listdir(dirname)

for img_name in files:
  imgF = Image.open(dirname+img_name)
  imgF = ImageEnhance.Brightness(imgF).enhance(2)
  imgF_np = pil_to_np(imgF)
  imgF_torch = np_to_torch(imgF_np).to(torch.device('cuda'))
  #imgF_torch = imgF_torch.expand(1,3,128,128)
  net_input = imgF_torch
  noise = net_input.detach().clone().to(torch.device('cuda'))
  reg_noise_std = 1. / 30.
  net_input_saved = net_input.detach().clone().to(torch.device('cuda'))
  for k in range(2000):
    out = closure(net_input_saved,noise,reg_noise_std)
  #torch.save(netG.state_dict(), '/content/drive/My Drive/net_gray/'+img_name+'G_change.pth')
  out_np = torch_to_np(out)
  plot_image_grid([np.clip(out_np, 0, 1), np.clip(imgF_np, 0, 1)], nrow=1)