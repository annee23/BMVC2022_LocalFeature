import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

import cv2
import math
import time
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import kornia as K
import sos_model

device = 'cuda:0'
sos_ = sos_model.SOSNet().cuda()

def describe_opencv(model, img, kpts, N=32, mag_factor=3, use_gpu=True):
    patches = []
    for kp in kpts:
        x, y = kp.pt
        s = kp.size
        a = kp.angle

        s = mag_factor * s / N
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)


        M = np.matrix([
            [+s * cos, -s * sin, (-s * cos + s * sin) * N / 2.0 + x],
            [+s * sin, +s * cos, (-s * sin - s * cos) * N / 2.0 + y]])

        patch = cv2.warpAffine(img, M, (N, N),
                               flags=cv2.WARP_INVERSE_MAP + \
                                     cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)
        patches.append(patch)

    patches = torch.from_numpy(np.asarray(patches)).float()
    patches = torch.unsqueeze(patches, 1)
    if use_gpu:
        patches = patches.cuda()
    descrs = model(patches)
    return descrs

def find_maxima(imgss, window_size, stride, kps_x, kps_y, octave):
    for img_n, imgs in enumerate(imgss):
        for i in range(len(imgs)):
            kps_x.append([])
            kps_y.append([])

            m = nn.Threshold(np.mean(imgs[i].cpu().detach().numpy()) * 5, 0)
            img = m(imgs[i])

            unfold = torch.nn.Unfold(kernel_size=(window_size, window_size), stride=stride)
            k = unfold(img.unsqueeze(0).unsqueeze(0))[0].permute(1,0)

            id = torch.argmax(k, dim=1)
            coord = (id == ((window_size ** 2) // 2))
            new = (coord == True).nonzero(as_tuple=False)
            row = new[:] // (496 // stride) * stride + window_size // 2
            col = new[:] % (496 // stride) * stride + window_size // 2

            kps_x[i].append(row * octave)
            kps_y[i].append(col * octave)

    return kps_x, kps_y

def describe_torch(model, img, kps_x, kps_y, octave, N=32, mag_factor=3, use_gpu=True):
    patches = None

    for im_id,im in enumerate(img[0]):
        for idx in range(len(kps_x)):
            try :
                x, y = kps_x[im_id][idx].cpu()//octave, kps_y[im_id][idx].cpu()//octave
            except IndexError:
                print(im_id,idx,len(kps_y))
            m = nn.ZeroPad2d(16)
            pad_img = m(im)
            patch = pad_img[x:x+32,y:y+32]
            patch = patch.unsqueeze(0).unsqueeze(0)

            if torch.count_nonzero(patch)==0:
                continue
            if patches!=None:
                patches = torch.cat((patches,patch),dim=0)
            else: patches = patch

    if use_gpu:
        patches = patches.cuda()
    descrs = model(patches)
    return descrs

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
        kps_x, kps_y = [], []
        out = self.gabor(img)
        end = time.time()
        print(f"{end - start:.5f} sec_gabor")
        kps_x, kps_y = find_maxima(out, window_size, stride, kps_x, kps_y, octave=1)
        end = time.time()
        print(f"{end - start:.5f} sec_findmax")
        # des = describe_torch(sos_, out, kps_x, kps_y, octave=1)
        # for oct in range(2,5,2):
        #     out = self.gabor(img//oct)
        #     kps_x, kps_y = find_maxima(out, window_size, stride, kps_x, kps_y, octave=oct)
        #     # des_ = describe_torch(sos_, out, kps_x, kps_y, octave=oct)
        #     # torch.cat((des,des_))
        return kps_x,kps_y

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

class ImageFolder(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))

    def sal_map(self, image):
        #image = image.cpu().detach().numpy()
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(image)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        map1 = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                             cv2.THRESH_TOZERO_INV | cv2.THRESH_OTSU)[1]
        map2 = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                             cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]
        map3 = cv2.threshold(map1.astype("uint8"), 127, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        map4 = cv2.threshold(map2.astype("uint8"), 127, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        map5 = cv2.threshold(map1.astype("uint8"), 127, 255,
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return map5 - map4,map3,map4
        # return torch.from_numpy(map5 - map4), torch.from_numpy(map3), torch.from_numpy(map4)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path).resize((500,500)))
        # ######## gamma transform part ############
        # part1, part2, part3 = self.sal_map(img)
        # cv2.imshow("o",img)
        # cv2.imshow("1",part1)
        # cv2.imshow("2", part2)
        # cv2.imshow("3", part3)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mean = hsv[..., 2].mean()
        # mid = 0.5
        # gamma = math.log(mid * 255) / math.log(mean)
        #
        img = np.array(Image.fromarray(img).convert("L"))
        # img = adjust_gamma(img, gamma=gamma)
        # ########## perspective part #################
        # a = np.random.normal([[9.89480085e-01, 1.69214700e-02, 8.80583236e+01], \
        #                       [7.47685288e-02, 1.10552544e+00, -7.56167770e+01], \
        #                       [2.13061082e-04, 5.15216757e-05, 9.99938407e-01]], \
        #                      [[1.386122201417242, 0.042968759212886655, 64407.697788545185], \
        #                       [0.11747808251085476, 1.1111895252671184, 163409.32527545939], \
        #                       [7.832456464080216e-07, 9.40111368610622e-08, 5.544239439179068e-05]])
        # img2 = cv2.warpPerspective(img, a, (img.shape[1], img.shape[0]))


        ########## bluring part ############
        k1 = np.zeros((5,9,9))
        x, y = np.mgrid[0:9:1, 0:9:1]
        pos = np.dstack((x, y))

        k_0 = multivariate_normal(mean=[4, 4], cov=[[1, 0],[0, 1]])
        k1[0] = k_0.pdf(pos)
        k_1 = multivariate_normal(mean=[4, 4], cov=[[1, 0], [0, 3]])
        k1[1] = k_1.pdf(pos)
        k_2 = multivariate_normal(mean=[4, 4], cov=[[3, 0], [0, 1]])
        k1[2] = k_2.pdf(pos)
        k_3 = multivariate_normal(mean=[4, 4], cov=[[3, 1.5], [1.5, 3]])
        k1[3] = k_3.pdf(pos)
        k_4 = multivariate_normal(mean=[4, 4], cov=[[3, -1.5], [-1.5, 3]])
        k1[4] = k_4.pdf(pos)

        img2 = cv2.filter2D(img, -1, k1[np.random.randint(0,4)])

        return img, img2

    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    start = time.time()

    dataset = ImageFolder(folder_path='PS-dataset/')
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8)
    end = time.time()
    print(f"{end - start:.5f} sec_0")
    model = Net().to(device)
    print(model.state_dict())

    optimizer = optim.SGD(model.parameters(), lr=1)


    model.train()
    for epoch in range(50):
        for img, img2 in train_loader:
            optimizer.zero_grad()

            img_ = img.unsqueeze(1).float().cuda()
            img2_ = img2.unsqueeze(1).float().cuda()

            window_size,stride  = img_.shape[2] // 100, 2

            kpx,kpy = model(img_)
            end = time.time()
            print(f"{end - start:.5f} sec_model1")
            kpx2,kpy2 = model(img2_)


            newkp = []
            for i in range(9):
                for id in range(len(kpx[i][0])):
                    k_p = cv2.KeyPoint(float(kpy[i][0][id].cpu().detach()), float(kpx[i][0][id].cpu().detach()), 100)
                    newkp.append(k_p)
            des = describe_opencv(sos_, img.cpu().detach().numpy()[0], newkp)
            end = time.time()
            print(f"{end - start:.5f} sec_makekp")
            newkp2 = []
            for i in range(9):
                for id in range(len(kpx2[i][0])):
                    k_p = cv2.KeyPoint(float(kpy2[i][0][id].cpu().detach()), float(kpx2[i][0][id].cpu().detach()), 100)
                    newkp2.append(k_p)
            des2 = describe_opencv(sos_, img2.cpu().detach().numpy()[0],newkp2)
            end = time.time()
            print(f"{end - start:.5f} sec_5")

            img_d = cv2.drawKeypoints(img.cpu().detach().numpy()[0], newkp, img.cpu().detach().numpy()[0])
            plt.imshow(img_d)
            plt.show()

            dis, idxs = K.feature.match_nn(des,des2)
            end = time.time()
            print(f"{end - start:.5f} sec_6")
            dis.mean().backward()

            optimizer.step()

            torch.save(model.state_dict(), './save_param6.pth')
            # if epoch%50==0:
            print(dis.mean(), model.thetas, model.lambdas, model.sigmas)
