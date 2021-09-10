import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import math
import glob
import numpy as np
from PIL import Image
from torchgeometry.core import warp_affine

from sos_model import SOSNet

sos_ = SOSNet().cuda()

def describe_torch(model, img, kps, N=32, mag_factor=3, use_gpu=True):
    kpx, kpy = kps[0], kps[1]
    s, a = mag_factor * 100 / N, -1
    cos = math.cos(a * math.pi / 180.0)
    sin = math.sin(a * math.pi / 180.0)

    x_t = kpx + (-s * cos + s * sin) * N / 2.0
    y_t = kpy + (-s * sin - s * cos) * N / 2.0

    M = torch.ones(len(x_t), 3, 3)
    M[:, 0, 2], M[:, 1, 2] = x_t, y_t
    M[:, 0, 0] = M[:, 1, 1] = +s * cos * M[:, 0, 0]
    M[:, 0, 1], M[:, 1, 0] = -s * sin * M[:, 0, 1], +s * sin * M[:, 1, 0]
    M[:, 2, 0], M[:, 2, 1] = 0 * M[:, 2, 0], 0 * M[:, 2, 1]

    M_inv = torch.linalg.inv(M)
    patches = warp_affine(img.repeat(len(M),1,1).unsqueeze(1).float(),\
                          torch.tensor(M_inv)[:,:2],(N,N))
    if use_gpu:
        patches = patches.cuda()

    descrs = model(patches)
    return descrs

def find_maxima(imgss, window_size, stride, octave):
    _x, _y = None, None

    for img_n, imgs in enumerate(imgss): # no batch
        for i in range(len(imgs)):
            m = nn.Threshold(np.mean(imgs[i].cpu().detach().numpy()) * 5, 0)
            img = m(imgs[i])

            unfold = torch.nn.Unfold(kernel_size=(window_size, window_size), stride=stride)
            k = unfold(img.unsqueeze(0).unsqueeze(0))[0].permute(1,0)

            id = torch.argmax(k, dim=1)
            coord = (id == ((window_size ** 2) // 2))
            new = (coord == True).nonzero(as_tuple=False)
            row = new[:] // (496 // stride) * stride + window_size // 2
            col = new[:] % (496 // stride) * stride + window_size // 2

            if _x==None:
                _x = row.squeeze().unsqueeze(0) * octave
                _y = col.squeeze().unsqueeze(0) * octave
                continue
            try :
                _x = torch.cat((_x, row.squeeze().unsqueeze(0) * octave),dim=1)
                _y = torch.cat((_y, col.squeeze().unsqueeze(0) * octave),dim=1)
            except RuntimeError:
                continue

    return _x, _y

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lambdas = nn.Parameter(torch.rand(3, requires_grad=True, dtype=torch.float).unsqueeze(0) * np.pi)
        self.sigmas = nn.Parameter(torch.rand(3, requires_grad=True, dtype=torch.float).unsqueeze(0) * 10)

        self.gammas = torch.Tensor([0.5]).unsqueeze(0).cuda()
        self.psis = torch.Tensor([np.pi / 2]).unsqueeze(0).cuda()
        self.thetas = torch.Tensor([np.pi/8,np.pi/4,np.pi*3/8,np.pi/2,\
                                    np.pi*5/8,np.pi*6/8,np.pi*7/8,np.pi]).unsqueeze(0).cuda()

        self.in_channels = 1
        self.kernel_s = 5
        indices = torch.arange(self.kernel_s, dtype=torch.float32) - (self.kernel_s - 1) / 2
        self.register_buffer('indices', indices)

        # number of channels after the conv
        self._n_channels_post_conv = self.in_channels * self.sigmas.shape[1] * \
                                     self.lambdas.shape[1] * self.gammas.shape[1] * \
                                     self.psis.shape[1] * self.thetas.shape[1]
    def gabor(self, image):
            sigmas = self.sigmas
            lambdas = self.lambdas
            gammas = self.gammas
            psis = self.psis
            thetas = self.thetas
            y = self.indices
            x = self.indices

            in_channels = sigmas.shape[0]
            assert in_channels == lambdas.shape[0]
            assert in_channels == gammas.shape[0]

            kernel_size = y.shape[0], x.shape[0]

            sigmas = sigmas.view(in_channels, sigmas.shape[1], 1, 1, 1, 1, 1, 1)
            lambdas = lambdas.view(in_channels, 1, lambdas.shape[1], 1, 1, 1, 1, 1)
            gammas = gammas.view(in_channels, 1, 1, gammas.shape[1], 1, 1, 1, 1)
            psis = psis.view(in_channels, 1, 1, 1, psis.shape[1], 1, 1, 1)
            thetas = thetas.view(in_channels, 1, 1, 1, 1, thetas.shape[1], 1, 1)

            y = y.view(1, 1, 1, 1, 1, 1, y.shape[0], 1)
            x = x.view(1, 1, 1, 1, 1, 1, 1, x.shape[0])

            sigma_x = sigmas
            sigma_y = sigmas / gammas

            sin_t = torch.sin(thetas)
            cos_t = torch.cos(thetas)
            y_theta = -x * sin_t + y * cos_t
            x_theta = x * cos_t + y * sin_t

            gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
                 * torch.cos(2.0 * math.pi * x_theta / lambdas + psis)

            # gb = gb.view(self._n_channels_post_conv, 1, 31, 31)
            gb = gb.view(-1, 1,  kernel_size[0], kernel_size[1])

            res = nn.functional.conv2d(input=image, weight=gb,
                                       padding=self.kernel_s//2, groups=1)
            if 1:
                res = res.view(1, 1, -1, 8, image.shape[2], image.shape[3])
                res, _ = res.max(dim=3)
            res = res.view(1, -1, image.shape[2], image.shape[3])

            return res
    def forward(self, img):
        out = self.gabor(img)
        _x, _y = find_maxima(out, window_size, stride, octave=1)

        for oct in range(2,5,2):
            out = self.gabor(img//oct)
            x, y = find_maxima(out, window_size, stride, octave=1)
            _x, _y = torch.cat((_x,x), dim=1), torch.cat((_y,y), dim=1)

        return torch.stack((_x,_y)).squeeze(1)

if __name__ =='__main__':

    model = Net().cuda()
    model.load_state_dict(torch.load('./save_param_0910.pth'))
    print("init: ",model.state_dict())
    model.eval()
    once = 3
    for fl in sorted(glob.glob('hpatches-release/*')):

        strin = fl[17:]

        if strin[0]=='v':
            continue
        if once >= 1:
            once -= 1
            continue
        print(strin)
        img_o = cv2.imread('hpatches-release/' + strin + '/1.ppm')
        img_g = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
        img = torch.tensor(img_g)
        img = img.unsqueeze(0).unsqueeze(0).float().cuda()

        window_size, stride = img.shape[2] // 100, 2
        # import skimage.data
        # from skimage.color import rgb2gray
        # astronaut = skimage.data.astronaut()
        # astronaut = rgb2gray(astronaut)
        # # astronaut[...,0] = astronaut[...,0].T
        #
        # astronaut = np.moveaxis(astronaut, -1, 0)[None, ...]
        # astronaut = astronaut[np.newaxis, ...]
        # astronaut = torch.from_numpy(astronaut).float().cuda()


        kps = model(img)
        des = describe_torch(sos_, img, kps)

        # img_d = cv2.drawKeypoints(img_g, newkp, img_o)
        # plt.imshow(img_d)
        # plt.show()
        # # kp1, des1 = sift_cv2.compute(img, newkp)
        # # kp1, des1 = newkp, des.cpu().detach().numpy()
        # print(len(newkp))

        ##################################################
        for idx in range(2,7):
            img2_ = cv2.imread('hpatches-release/' + strin + '/'+str(idx)+'.ppm')
            img2_g = cv2.cvtColor(img2_, cv2.COLOR_BGR2GRAY)

            hsv = cv2.cvtColor(img2_, cv2.COLOR_BGR2HSV)
            mean = hsv[..., 2].mean()
            mid = 0.5
            gamma = math.log(mid * 255) / math.log(mean)

            img2 = np.array(Image.fromarray(img2_g).convert("L"))
            img2 = adjust_gamma(img2, gamma=gamma)
            img2 = torch.Tensor(img2)
            img2 = img2.unsqueeze(0).unsqueeze(0).float().cuda()

            kps2 = model(img2)
            des2 = describe_torch(sos_, img2, kps2)
            # img2_d = cv2.drawKeypoints(img2_g, newkp2, img2_)
            # plt.imshow(img2_d)
            # plt.show()

            # kp2, des2 = newkp2,des2.cpu().detach().numpy() #sift_cv2.compute(img2, newkp)
            ####################################################
            # dis, idxs = K.feature.match_nn(des, des2)
            # print(dis[:10])
            # good = 0
            # a = np.loadtxt('./hpatches-release/' + strin + '/H_1_' + str(idx))
            # total = 0
            # for ind, (i_1, i_2 )in enumerate(idxs):
            #     if dis[ind] >0.95:
            #         continue
            #     total += 1
            #     x_, y_, _ = np.dot(a, np.array([kpx[i_1].cpu().detach().numpy(), kpy[i_1].cpu().detach().numpy(), 1]))
            #     if abs(x_-kpx2[i_2].cpu().detach().numpy())<20 and abs(y_-kpy2[i_2].cpu().detach().numpy())<20:
            #         good += 1

            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des.cpu().detach().numpy(), des2.cpu().detach().numpy(), k=2)
            arrx1, arrx2, arry1, arry2 = [], [], [], []
            matchesMask = [[0, 0] for i in range(len(matches))]
            good = 0

            imgO1 = img
            imgO2 = img2
            a = np.loadtxt('./hpatches-release/' + strin + '/H_1_' + str(idx))
            # a = Util.find_homo(kp1,kp2,des1,des2)
            choo = 0
            for i, (m, n) in enumerate(matches):
                if 1:  # m.distance < 0.9 * n.distance:
                    x, y = newkp[m.queryIdx].pt
                    # k = (imgO1.shape[0] // 30) // (2)  # ** kp1[m.queryIdx].octave)
                    # x, y = int(x), int(y)
                    # if x - k < 0 or x + k > imgO1.shape[0] or y - k < 0 or y + k > imgO1.shape[1]:
                    #     continue
                    # patch = imgO1[x - k:x + k, y - k:y + k]
                    # # plt.imsave("tmp.png", patch)
                    # # patch_ = cv2.imread("tmp.png")
                    # # patch_ = Image.fromarray(patch_).resize((90, 90))

                    x2, y2 = newkp2[m.trainIdx].pt
                    x_, y_, _ = np.dot(a, np.array([x, y, 1]))
                    x__, y__ = abs(x2 - x_), abs(y2 - y_)

                    if x__ < 20 and y__ < 20:
                        matchesMask[i] = [1, 0]
                        good += 1
                        arrx1.append(x)
                        arry1.append(y)
                    else:
                        arrx2.append(x)
                        arry2.append(y)
                    draw_params = dict(matchColor=(0, 255, 0),
                                       singlePointColor=(255, 0, 0),
                                       matchesMask=matchesMask,
                                       flags=0)

            print(good / (len(matches) - choo))
