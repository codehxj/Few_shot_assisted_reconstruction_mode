fiiqaWeight = 'model/illumination_net/97_160_2.pth'
illumination_net = ShuffleNetV2(200).type(dtype)
if colab_cuda =='True':
  checkpoint = torch.load(fiiqaWeight)
else:
  checkpoint = torch.load(fiiqaWeight,map_location='cpu')

illumination_net.load_state_dict(checkpoint['net'])