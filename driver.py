import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage.segmentation._slic import _enforce_label_connectivity_cython
import kornia


# CODE IS BASED ON ss-with-RIM (Suzuki,2020): https://github.com/DensoITLab/ss-with-RIM


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def conv_in_relu(in_c, out_c, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.ReflectionPad2d(kernel_size // 2),
        nn.Conv2d(in_c, out_c, kernel_size, stride=stride, bias=False),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.ReLU(inplace=True)
    )



class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=100):
        super(ASPP, self).__init__()
        self.out_channels = out_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Encoder(nn.Module):
    def __init__(self, in_c=5, n_filters=32, n_layers=5):
        super().__init__()
        self.original_in_c = in_c
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(conv_in_relu(in_c, n_filters << i))
            in_c = n_filters << i

        self.layers = nn.Sequential(*self.layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        return x


class ASPP_SuperPix(nn.Module):
    def __init__(self, img_size=256, img_channel=1, atrous_rates=[2, 4, 8], n_spix=100, n_filters=32, n_layers=4,
                 use_cood_input=False, use_recons_output=True,
                 use_last_inorm=True, use_edge_loss=True, use_recon_loss=True, use_spix_recons_for_loss=True, use_TV=True,):
        super(ASPP_SuperPix, self).__init__()
        self.img_size = img_size
        self.use_cood_input = use_cood_input
        if use_cood_input:
            in_channels = img_channel + 2
        else:
            in_channels = img_channel
        self.in_channels = in_channels  # img_channel == 3 for RGB, img_channel == 1 for gray, + 2 for pixel location
        self.atrous_rates = atrous_rates  # dilation rate
        self.n_spix = n_spix  # number of superpixel
        self.enc_out_channel = n_filters * (2 ** (n_layers - 2))
        self.use_recons_output = use_recons_output
        if use_recons_output:
            self.out_channels = n_spix + 3  # + RGB
        else:
            self.out_channels = n_spix  # number of superpixel, also output channel

        self.use_last_inorm = use_last_inorm
        if use_last_inorm:
            self.norm = nn.InstanceNorm2d(n_spix, affine=True)
        self.use_edge_loss = use_edge_loss
        self.use_recon_loss = use_recon_loss
        self.use_spix_recons_for_loss = use_spix_recons_for_loss
        self.use_TV = use_TV

        self.encoder = Encoder(in_channels, n_filters, n_layers)
        self.decoder = ASPP(in_channels=2 * self.enc_out_channel, atrous_rates=atrous_rates,
                            out_channels=self.enc_out_channel)
        self.final = nn.Sequential(
            nn.Conv2d(self.enc_out_channel, self.enc_out_channel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.enc_out_channel, self.out_channels, 1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.encoder(x)
        lap_x = kornia.filters.laplacian(x, kernel_size=3)
        x = torch.cat([x, lap_x], dim=1)
        x = self.decoder(x)
        x = self.final(x)
        if self.use_recons_output:
            recons, spix = x[:, :3], x[:, 3:]
        else:
            spix = x.clone()
            recons = None
        if self.use_last_inorm:
            spix = self.norm(spix)

        return spix, recons

    def mutual_information(self, logits, coeff):
        prob = logits.softmax(1)
        pixel_wise_ent = - (prob * F.log_softmax(logits, 1)).sum(1).mean()
        marginal_prob = prob.mean((2, 3))
        marginal_ent = - (marginal_prob * torch.log(marginal_prob + 1e-16)).sum(1).mean()
        return pixel_wise_ent - coeff * marginal_ent

    def TV_smoothness(self, logits, image):
        prob = logits.softmax(1)
        dp_dx = prob[..., :-1] - prob[..., 1:]
        dp_dy = prob[..., :-1, :] - prob[..., 1:, :]
        di_dx = image[..., :-1] - image[..., 1:]
        di_dy = image[..., :-1, :] - image[..., 1:, :]
        di_dx2 = 0.5 * ((di_dx[:, :, :-1, :] ** 2) + (di_dx[:, :, 1:, :] ** 2))
        di_dy2 = 0.5 * ((di_dy[:, :, :, :-1] ** 2) + (di_dy[:, :, :, 1:] ** 2))

        dp_dx2 = 0.5 * ((dp_dx[:, :, :-1, :] ** 2) + (dp_dx[:, :, 1:, :] ** 2))
        dp_dy2 = 0.5 * ((dp_dy[:, :, :, :-1] ** 2) + (dp_dy[:, :, :, 1:] ** 2))

        TV_image = torch.sqrt(di_dx2 + di_dy2 + 1e-8).sum(1)
        TV_prob = torch.sqrt(dp_dx2 + dp_dy2 + 1e-8).sum(1)
        TV = (TV_image * TV_prob).mean()
        return TV

    def smoothness(self, logits, image):
        prob = logits.softmax(1)
        dp_dx = prob[..., :-1] - prob[..., 1:]
        dp_dy = prob[..., :-1, :] - prob[..., 1:, :]
        di_dx = image[..., :-1] - image[..., 1:]
        di_dy = image[..., :-1, :] - image[..., 1:, :]

        return (dp_dx.abs().sum(1) * (-di_dx.pow(2).sum(1) / 8).exp()).mean() + \
               (dp_dy.abs().sum(1) * (-di_dy.pow(2).sum(1) / 8).exp()).mean()

    def ContourLoss(self, mean_img, image):
        lap_mean_img = kornia.filters.laplacian(mean_img, 3)
        lap_mean_img = F.softmax(lap_mean_img / lap_mean_img.abs().max())

        lap_img = kornia.filters.laplacian(image, 3)
        lap_img = F.softmax(lap_img / lap_img.abs().max())

        return F.kl_div(lap_mean_img, lap_img)

    def reconstruction(self, recons, image):
        return F.mse_loss(recons, image)

    def __preprocess(self, image, device="cuda"):
        image = image.permute(2, 0, 1).float()[None]
        h, w = image.shape[-2:]
        coord = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w))).float()[None]
        input = torch.cat([image, coord], 1).to(device)
        data_mean = input.mean((2, 3), keepdim=False)
        data_std = input.std((2, 3), keepdim=False)
        input = (input - input.mean((2, 3), keepdim=True)) / input.std((2, 3), keepdim=True)
        return input, data_mean, data_std

    def imageFromSpix(self, spix, input):
        probs = F.softmax(spix, 1)
        input_img = input[:, :3, :, :]
        probs = probs.unsqueeze(1)
        votes = input_img[:, :, None, :, :] * probs
        vals = (votes.sum((3, 4)) / probs.sum((3, 4))).unsqueeze(-1).unsqueeze(-1)
        mean_img = (vals * probs[:, None, :, :, :]).sum(3).squeeze(0)

        return mean_img

    def imageFromHardSpix(self, spix, image):
        mean_img = np.zeros_like(image)
        for ii in np.unique(spix):
            mean_val = image[spix == ii, :].mean(dim=0)
            mean_img[spix == ii] = mean_val

        return mean_img

    def optimize(self, image, n_iter=500, lr=1e-2, lam=2, alpha=2, beta=2, eta=1, device="cuda"):
        input, data_mean, data_std = self.__preprocess(image, device)
        optimizer = optim.Adam(self.parameters(), lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
        input_image = input[:, :3].clone()

        for i in range(n_iter):
            optimizer.zero_grad()
            if self.use_cood_input:
                spix, recons = self.forward(input)
            else:
                spix, recons = self.forward(input[:, :3, :, :])
            mean_img = self.imageFromSpix(spix, input)

            # clustering loss
            loss_clustering = self.mutual_information(spix, lam)

            # smoothness loss, only this loss takes location input
            if self.use_TV:
                loss_smooth = self.TV_smoothness(spix, input)
            else:
                loss_smooth = self.smoothness(spix, input)

            # edge loss
            if self.use_edge_loss:
                if self.use_spix_recons_for_loss:
                    loss_edge = 0.5 * (self.ContourLoss(mean_img, input_image) + self.ContourLoss(recons, input_image))
                else:
                    loss_edge = 0.5 * self.ContourLoss(recons, input_image)

            # recon loss
            if self.use_recon_loss:
                if self.use_spix_recons_for_loss:
                    loss_recon = self.reconstruction(mean_img, input_image) + self.reconstruction(recons, input_image)
                else:
                    loss_recon = self.reconstruction(recons, input_image)

            loss = loss_clustering + alpha * loss_smooth + beta * loss_recon + eta * loss_edge

            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f"[{i + 1}/{n_iter}] loss {loss.item()}, "
                  f"loss_clustering {loss_clustering.item()}, "
                  f"loss_smooth {loss_smooth.item()}, "
                  f"loss_edge {loss_edge.item()}",
                  f"loss_recon {loss_recon.item()}",
                  flush=True)

        return self.calc_spixel(image, device),\
               mean_img.detach().cpu(), \
               input_image.detach().cpu(), \
               data_mean[0, :3].cpu(), \
               data_std[0, :3].cpu(), \
               recons.detach().cpu()

    def calc_spixel(self, image, device="cuda"):
        input, _, _ = self.__preprocess(image, device)
        if self.use_cood_input:
            spix, recons = self.forward(input)
        else:
            spix, recons = self.forward(input[:, :3, :, :])

        spix = spix.argmax(1).squeeze().to("cpu").detach().numpy()

        segment_size = spix.size / self.n_spix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        spix = _enforce_label_connectivity_cython(
            spix[None], min_size, max_size)[0]

        return spix


if __name__ == "__main__":
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="./sample/GT_01350.png", type=str, help="/path/to/image")
    parser.add_argument("--img_size", default=256, type=int, help="resolution of image")
    parser.add_argument("--img_channel", default=3, type=int, help="3 for RGB, 1 for gray")

    parser.add_argument("--atrous_rates", default=[2, 4, 8], type=int, help="dilation rate setting")
    parser.add_argument("--n_spix", default=128, type=int, help="number of superpixels")
    parser.add_argument("--n_filters", default=32, type=int, help="number of convolution filters")
    parser.add_argument("--n_layers", default=3, type=int, help="number of convolution layers")

    parser.add_argument("--lam", default=2, type=float, help="coefficient of marginal entropy")
    parser.add_argument("--alpha", default=2, type=float, help="coefficient of smoothness loss")
    parser.add_argument("--beta", default=2, type=float, help="coefficient of reconstruction loss")
    parser.add_argument("--eta", default=1, type=float, help="coefficient of edge loss")

    parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
    parser.add_argument("--n_iter", default=500, type=int, help="number of iterations")
    parser.add_argument("--out_dir", default="./", type=str, help="output directory")

    parser.add_argument("--use_cood_input", default=True, type=bool, help="if to input the location (C+2)")
    parser.add_argument("--use_recons_output", default=True, type=bool, help="if to optimize also for reconstruction")
    parser.add_argument("--use_last_inorm", default=True, type=bool, help="if to use last instance norm")
    parser.add_argument("--use_edge_loss", default=True, type=bool, help="if to use edge loss")
    parser.add_argument("--use_recon_loss", default=True, type=bool, help="if to use recon loss")
    parser.add_argument("--use_spix_recons_for_loss", default=True, type=bool,
                        help="if previous is true, then can choose to recons the spix")
    parser.add_argument("--use_TV", default=False, type=bool,
                        help="if to use TV smoothness (if false use L2 smoothness)")
    args = parser.parse_args()

    # device info
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = ASPP_SuperPix(img_size=args.img_size,
                          img_channel=args.img_channel,
                          atrous_rates=args.atrous_rates,
                          n_spix=args.n_spix,
                          n_filters=args.n_filters,
                          n_layers=args.n_layers,
                          use_cood_input=args.use_cood_input,
                          use_recons_output=args.use_recons_output,
                          use_last_inorm=args.use_last_inorm,
                          use_edge_loss=args.use_edge_loss,
                          use_recon_loss=args.use_recon_loss,
                          use_spix_recons_for_loss=args.use_spix_recons_for_loss,
                          use_TV=args.use_TV).to(device)

    # load image
    if args.image is None:  # load sample image from scipy
        import scipy.misc
        img = scipy.misc.face()
    else:
        img = plt.imread(args.image)
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img.float(), size=args.img_size, mode='bilinear', align_corners=False)
    img = img.squeeze().permute(1, 2, 0)

    spix, mean_img, input_tensor, mean, std, recons = model.optimize(img,
                                                                     args.n_iter,
                                                                     args.lr,
                                                                     args.lam,
                                                                     args.alpha,
                                                                     args.beta,
                                                                     args.eta,
                                                                     device)

    input_tensor = UnNormalize(mean, std)(input_tensor).squeeze().transpose(0, 2).transpose(0, 1)  #
    mean_img = UnNormalize(mean, std)(mean_img).squeeze().transpose(0, 2).transpose(0, 1)
    recons_img = UnNormalize(mean, std)(recons).squeeze().transpose(0, 2).transpose(0, 1)

    mean_img = model.imageFromHardSpix(spix, img)
    fig, ax = plt.subplots(1, 3, )
    ax[0].imshow(mark_boundaries(img.numpy(), spix))
    ax[0].set_title('Spix')
    ax[1].imshow(mean_img)
    ax[1].set_title('Spix Image')
    ax[2].imshow(input_tensor)
    ax[2].set_title('GT Image')
    plt.show()
