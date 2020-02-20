from model.face_parsing.model import BiSeNet

faceParsNet = BiSeNet(n_classes=19)
faceParsNet.load_state_dict(torch.load('model/face_parsing/79999_iter.pth'))

Fname = 'data/MultiPIE_test/input/008_01_01_051_03.png'
imgF = Image.open(Fname)
F_resize = imgF.resize((512, 512), Image.BILINEAR)
F_resize_np = pil_to_np(F_resize)
F_resize_torch = np_to_torch(F_resize_np)
#F_resize_torch = F_resize_torch.expand(1,3,512,512)
#F_resize_torch = torch.nn.functional.interpolate(imgF_torch, scale_factor=4, mode='bilinear')
F_parsOut = faceParsNet(F_resize_torch)[0]