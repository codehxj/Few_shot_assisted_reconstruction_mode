import  model.assisit_normalizaiton_model as AN_model
import utils.common_utils
from PIL import Image
import os
import random


def AN_model_stru():
    netSingG = AN_model.GeneratorConcatSkip2CleanAdd()
    netSingG.apply(AN_model.weights_init)
    return netSingG

# The default database is PIE
def training_ANM(dirname,ANM,optimizerANM,k):
    files = os.listdir(dirname)
    random.shuffle(files)
    for img_name in files:
        if img_name[-6:-4]=='07':
            continue
        else:
            img1 = Image.open(dirname+img_name)
            print('img_name:',img_name)
            img_np1 = pil_to_np(img1)
            img_torch1 = np_to_torch(img_np1).type(dtype)

            #filp
            img1_flip = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img_np1_flip = pil_to_np(img1_flip)
            img_torch1_flip = np_to_torch(img_np1_flip).type(dtype)

            img_name2 = img_name[:-6]+'07.png'
            if  not os.path.exists(dirname + img_name2):
                continue
            img2 = Image.open(dirname + img_name2)
            img_np2 = pil_to_np(img2)
            img_torch2 = np_to_torch(img_np2).type(dtype)

            img2_flip = img2.transpose(Image.FLIP_LEFT_RIGHT)
            img_np2_flip = pil_to_np(img2_flip)
            img_torch2_flip = np_to_torch(img_np2_flip).type(dtype)

            optimizerANM.zero_grad()
            output1 = ANM(img_torch1)
            loss1 = L1_loos(output1, img_torch2)
            loss1.backward(retain_graph=True)
            output2 = ANM(img_torch1_flip)
            loss2 = L1_loos(output2,img_torch2_flip)
            loss2.backward(retain_graph=True)
            optimizerANM.step()

    if k % 200 ==0:
        torch.save(ANM.state_dict(), 'ANM.pth')
        plot_image_grid([np.clip(torch_to_np(output1), 0, 1), np.clip(img_np1, 0, 1)], nrow=2)

def training_YaleB():
  for file_name in files:
    file_path = os.listdir(dirname+file_name)
    if int(file_name[-2:]) > 10:
      continue
    for img_name in file_path:
        if img_name[-3:]=='pgm':
            img1 = Image.open(dirname+file_name+'/'+img_name)
            img1 = img1.resize((128,128),Image.BILINEAR)
            img_np1 = pil_to_np(img1)
            img_torch1 = np_to_torch(img_np1).to(torch.device('cuda'))
            #img_torch1 = img_torch1.expand(1,3,128,128)

            #filp
            img1_flip = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img_np1_flip = pil_to_np(img1_flip)
            img_torch1_flip = np_to_torch(img_np1_flip).to(torch.device('cuda'))
            #img_torch1_flip = img_torch1_flip.expand(1,3,128,128)

            img_name2 = img_name[:7]+'_P00A+000E+00.pgm'
            img2 = Image.open(dirname+file_name+'/'+ img_name2)
            img2 = img2.resize((128,128),Image.BILINEAR)
            img_np2 = pil_to_np(img2)
            img_torch2 = np_to_torch(img_np2).to(torch.device('cuda'))
            #img_torch2 = img_torch2.expand(1,3,128,128)

            img2_flip = img2.transpose(Image.FLIP_LEFT_RIGHT)
            img_np2_flip = pil_to_np(img2_flip)
            img_torch2_flip = np_to_torch(img_np2_flip).to(torch.device('cuda'))
            #img_torch2_flip = img_torch2_flip.expand(1,3,128,128)

            optimizernetSingG.zero_grad()
            output3 = netSingG(img_torch1)
            loss3 = L1_loos(output3, img_torch2) *1000
            loss3.backward(retain_graph=True)
            output4 = netSingG(img_torch1_flip)
            loss4 = L1_loos(output4,img_torch2_flip) * 1000
            loss4.backward(retain_graph=True)
            optimizernetSingG.step()



dirname= 'PIE'
dtype = is_cuda('True')
ANM = AN_model_stru().type(dtype)
optimizerANM = optim.Adam(ANM.parameters(), 0.01,betas=(0.5, 0.999))

for k in range(8000):
    training_ANM(dirname,ANM,optimizerANM,dtype,k)