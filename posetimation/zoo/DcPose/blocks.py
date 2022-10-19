import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import warnings


class FlowLayer(nn.Module):

    def __init__(self, channels=17, bath_size=64, n_iter=10):
        super(FlowLayer, self).__init__()
        # self.bottleneck = nn.Conv3d(channels, bottleneck, stride=1, padding=0, bias=False, kernel_size=1)
        # self.unbottleneck = nn.Conv3d(bottleneck*2, channels, stride=1, padding=0, bias=False, kernel_size=1)
        # self.bn = nn.BatchNorm3d(channels)
        # channels = bottleneck

        # transpose() 는 딱 2개의 차원을 맞교환할 수 있다.
        # permute() 는 모든 차원들을 맞교환할 수 있다.
        # x = torch.rand(16, 32, 3)
        # y = x.tranpose(0, 2)  # [3, 32, 16]
        # z = x.permute(2, 1, 0)  # [3, 32, 16]

        # (batch, channel, height, width)

        self.n_iter = n_iter

        # h 방향, w 방향으로 각각의 gradient를 계산하기 위해서로 보인다.
        # params의 차이는 - require_grad 를 반영할건지, 반영 안할 건지에 대한 부분이다.

        # torch.Size([1, 1, 1, 3])
        # conv 연산을 통해서 gradient를 구하는 부분
        self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5, 0, 0.5]]]]).repeat(channels, channels, 1, 1))
        # torch.Size([1, 1, 3, 1]) -> transpoe(3,2)를 통해 변경
        # conv 연산을 통해서 gradient를 구하는 부분
        self.img_grad2 = nn.Parameter(
            torch.FloatTensor([[[[-0.5, 0, 0.5]]]]).transpose(3, 2).repeat(channels, channels, 1, 1))

        self.f_grad = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
        self.f_grad2 = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
        self.div = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
        self.div2 = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))

        self.channels = channels

        self.t = 0.3
        self.l = 0.15
        self.a = 0.25

        self.t = nn.Parameter(torch.FloatTensor([self.t]))
        self.l = nn.Parameter(torch.FloatTensor([self.l]))
        self.a = nn.Parameter(torch.FloatTensor([self.a]))

        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0)

        # 수정
        self.bn = nn.BatchNorm2d(channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def norm_img(self, x):
        mx = torch.max(x)
        mn = torch.min(x)
        x = 255 * (x - mn) / (mn - mx)
        return x

    # (batch, channel, height, width)
    def forward_grad(self, x):
        grad_x = F.conv2d(F.pad(x, (0, 0, 0, 1)), self.f_grad)  # , groups=self.channels)
        grad_x[:, :, -1, :] = 0

        grad_y = F.conv2d(F.pad(x, (0, 0, 0, 1)), self.f_grad2)  # , groups=self.channels)
        grad_y[:, :, -1, :] = 0
        return grad_x, grad_y

    # divergence : 발산
    # 왜 있는가?
    # (batch, channel, height, width)
    # torch.nn.functional.pad(input, pad, mode='constant', value=0) => 원하는대로 padding을 줄 수 있는 함수!!.
    def divergence(self, x, y):
        # -1 부분 있으면 마지막 부분을 하나를 제외한다는 의미이다.
        # to pad the last 2 dimensions of the input tensor, then use (left, right, top, bottom);
        # 첫 행 부분에 패딩을 추가하고, 마지막 부분의 값은 제외한다는 의미이다.
        tx = F.pad(x[:, :, :-1, :], (0, 0, 1, 0))
        ty = F.pad(y[:, :, :-1, :], (0, 0, 1, 0))

        # 해당 부분의 코드의미도 tx ty 관련해서 행 마지막 부분에 패딩을 추가한다는 의미이다.
        # 해당 부분을 통해서 gradient를 각각 계산하게 되는데 ................
        # 왜 구지 처음 x,y 값에 대한 첫줄을 패딩으로 추가해주는 것일까? 그냥 이미지에서 마지막 bottom 부분에만 패딩을 추가해주면 될 것 같은데 ......

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # self.div = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,channels,1,1))
        grad_x = F.conv2d(F.pad(tx, (0, 0, 0, 1)), self.div)  # , groups=self.channels)
        grad_y = F.conv2d(F.pad(ty, (0, 0, 0, 1)), self.div2)  # , groups=self.channels)
        return grad_x + grad_y

    def forward(self, current, next):
        # flowlayer를 통과하게 되면 n프레임을 입력하게 되면 n-1개의 feature vector가 나오게 됨.
        # 즉 ... 최정적으로 feature vector를 구할 때 기존 프레임의 값을 합치는 것이 성능이 잘 나오기 때문에 ... residual 부분에서 마지막 프레임을 제외하고 더해주는 것이다.!!
        # residual = x[:,:,:-1]

        #
        # x = self.bottleneck(x)

        # b,c,t,h,w

        # x라는 것은 전체 비디오 n프레임이 있을 경우 => 1 ~ n-1 프레임까지를 말한다.
        # x = inp[:,:,:-1]

        # y라는 것은 전체 비디오 n프레임 중에서 => 2 ~ n프레임까지를 말한다.
        # y라는 값은 x 기준으로 보았을 때 한 프레임정도 앞에 있는 것들 말한다.
        # y = inp[:,:,1:]

        # => x,y 를 각각 flowlayer의 전프레임과 후프레임으로 입력하여 각 레이어에 대한 flow 값을 도출한다.

        # b,c,t,h,w = x.size()
        # x = x.permute(0,2,1,3,4).contiguous().view(b*t,c,h,w)
        # y = y.permute(0,2,1,3,4).contiguous().view(b*t,c,h,w)

        x = self.norm_img(current)
        y = self.norm_img(next)

        u1 = torch.zeros_like(x)
        u2 = torch.zeros_like(y)
        l_t = self.l * self.t
        taut = self.a / self.t

        # (batch, channel, height, width)
        # padding : (left, right, top, bottom);
        # 왜 여기서는 left와 right에 대한 패딩을 한 이유는 gradient를 구할 경우, 3개의 픽셀씩 묶어서 작업을 하기 때문에
        # 패딩을 하지 않을 경우, w 너비 부분의 크기가 줄어둘기 때문이다.
        grad2_x = F.conv2d(F.pad(y, (1, 1, 0, 0)), self.img_grad, padding=0, stride=1)  # , groups=self.channels)
        grad2_x[:, :, :, 0] = 0.5 * (x[:, :, :, 1] - x[:, :, :, 0])
        grad2_x[:, :, :, -1] = 0.5 * (x[:, :, :, -1] - x[:, :, :, -2])

        # (batch, channel, height, width)
        # padding : (left, right, top, bottom);
        grad2_y = F.conv2d(F.pad(y, (0, 0, 1, 1)), self.img_grad2, padding=0, stride=1)  # , groups=self.channels)
        grad2_y[:, :, 0, :] = 0.5 * (x[:, :, 1, :] - x[:, :, 0, :])
        grad2_y[:, :, -1, :] = 0.5 * (x[:, :, -1, :] - x[:, :, -2, :])

        p11 = torch.zeros_like(x.data)
        p12 = torch.zeros_like(x.data)
        p21 = torch.zeros_like(x.data)
        p22 = torch.zeros_like(x.data)

        gsqx = grad2_x ** 2
        gsqy = grad2_y ** 2
        grad = gsqx + gsqy + 1e-12

        rho_c = y - grad2_x * u1 - grad2_y * u2 - x

        for i in range(self.n_iter):
            rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12

            # 왜? v1, v2 두개의 값이 있는가??
            # v1과 v2는 각각 x와 y의 방향에 대한 크기이다.
            v1 = torch.zeros_like(x.data)
            v2 = torch.zeros_like(x.data)
            mask1 = (rho < -l_t * grad).detach()
            v1[mask1] = (l_t * grad2_x)[mask1]
            v2[mask1] = (l_t * grad2_y)[mask1]

            # 해당 mask2는아래 연산을 통해서 각 인덱싱부분에 값을 넣을지 안 넣을지를 결정해주는 부분이다.
            # 문법적 이해 체크하도록!!
            mask2 = (rho > l_t * grad).detach()
            v1[mask2] = (-l_t * grad2_x)[mask2]
            v2[mask2] = (-l_t * grad2_y)[mask2]

            # ^1 ^2 의 의미는 무엇인가?
            mask3 = ((mask1 == False) & (mask2 == False) & (grad > 1e-12)).detach()
            v1[mask3] = ((-rho / grad) * grad2_x)[mask3]
            v2[mask3] = ((-rho / grad) * grad2_y)[mask3]

            # delete a Tensor in GPU
            del rho
            del mask1
            del mask2
            del mask3

            v1 += u1
            v2 += u2

            # u1은 x방향 u2는 y방향을 의미한다.
            u1 = v1 + self.t * self.divergence(p11, p12)
            u2 = v2 + self.t * self.divergence(p21, p22)

            # delete a Tensor in GPU
            del v1
            del v2
            u1 = u1
            u2 = u2

            u1x, u1y = self.forward_grad(u1)
            u2x, u2y = self.forward_grad(u2)

            p11 = (p11 + taut * u1x) / (1. + taut * torch.sqrt(u1x ** 2 + u1y ** 2 + 1e-12))
            p12 = (p12 + taut * u1y) / (1. + taut * torch.sqrt(u1x ** 2 + u1y ** 2 + 1e-12))
            p21 = (p21 + taut * u2x) / (1. + taut * torch.sqrt(u2x ** 2 + u2y ** 2 + 1e-12))
            p22 = (p22 + taut * u2y) / (1. + taut * torch.sqrt(u2x ** 2 + u2y ** 2 + 1e-12))

            # delete a Tensor in GPU
            del u1x
            del u1y
            del u2x
            del u2y

        flow = torch.cat([u1, u2], dim=1)
        # flow = self.conv(flow)
        flow = self.bn(flow)
        # flow = self.relu(flow)

        # print(u1)
        # print(u2)

        return u1, u2
        # 조인트별로 x방향과 y방향 motion estimator이다.
        # bath, channel
class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.float(),
                size=T // self.stride,
                mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.float()

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.to(out_conv.device)
        # out_conv = out_conv * out_mask[:, None, :].to(out_conv.device)
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            n_head,  # number of attention heads
            n_ds_strides=(1, 1),  # downsampling strides for q & x, k & v
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # dimension of the hidden layer in MLP
            act_layer=nn.GELU,  # nonlinear activation used in MLP, default GELU
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # drop path rate
            mha_win_size=-1,  # > 0 to use window mha
            use_rel_pe=False  # if to add rel position encoding to attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        # specify the attention module
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe  # only valid for local attention
            )
        else:
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, pos_embd=None):
        # def forward(self, x, mask, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        # out = self.attn(self.ln1(x), mask)
        out = self.attn(self.ln1(x))
        # out_mask_float = out_mask.float()
        out = self.pool_skip(x) + self.drop_path_attn(out)
        # out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)))
        # out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd
            # out += pos_embd * out_mask_float
        return out
        # return out, out_mask


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            n_qx_stride=1,  # dowsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        # self.query_conv = MaskedConv1D(
        #     self.n_embd, self.n_embd, kernel_size,
        #     stride=stride, padding=padding, groups=self.n_embd, bias=False
        # )
        self.query_conv = nn.Conv1d(self.n_embd, self.n_embd, kernel_size,
                                    stride=stride, padding=padding, groups=self.n_embd, bias=False,
                                    padding_mode='zeros')
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = nn.Conv1d(self.n_embd, self.n_embd, kernel_size,
                                  stride=stride, padding=padding, groups=self.n_embd, bias=False, padding_mode='zeros')
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = nn.Conv1d(self.n_embd, self.n_embd, kernel_size,
                                    stride=stride, padding=padding, groups=self.n_embd, bias=False,
                                    padding_mode='zeros')
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)
        # self.self_attn = nn.MultiheadAttention(self.n_embd, n_head, dropout=attn_pdrop)

    # def forward(self, x, mask):
    def forward(self, x):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q = self.query_conv(x)
        # q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k = self.key_conv(x)
        # k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v = self.value_conv(x)
        # v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # # move head forward to be the batch dim
        # # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        # k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1)
        q = q.view(B, self.n_head, self.n_channels, -1)
        v = v.view(B, self.n_head, self.n_channels, -1)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        # att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        # out = att @ (v * kv_mask[:, :, :, None].float())
        out = att @ v
        # out, att_map = self.self_attn(q, k, v,
        #                               attn_mask=None,
        #                               key_padding_mask=None)
        # re-assemble all head outputs side by side

        # out = out.view(B, C, -1)
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out))
        # out = self.proj_drop(self.proj(out)) * qx_mask.float()
        return out
        # return out, qx_mask


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            window_size,  # size of the local attention window
            n_qx_stride=1,  # dowsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
            use_rel_pe=False  # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        # self.query_conv = MaskedConv1D(
        #     self.n_embd, self.n_embd, kernel_size,
        #     stride=stride, padding=padding, groups=self.n_embd, bias=False
        # )
        self.query_conv = nn.Conv1d(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        # self.key_conv = MaskedConv1D(
        #     self.n_embd, self.n_embd, kernel_size,
        #     stride=stride, padding=padding, groups=self.n_embd, bias=False
        # )
        self.key_conv = nn.Conv1d(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        # self.value_conv = MaskedConv1D(
        #     self.n_embd, self.n_embd, kernel_size,
        #     stride=stride, padding=padding, groups=self.n_embd, bias=False
        # )
        self.value_conv = nn.Conv1d(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # relative position encoding
        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd) ** 0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
            shift every row 1 step right, converting columns into diagonals.
            Example::
                  chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                           -1.8348,  0.7672,  0.2986,  0.0285,
                                           -0.7584,  0.4206, -0.0405,  0.1599,
                                           2.0514, -1.1600,  0.5372,  0.2629 ]
                  window_overlap = num_rows = 4
                 (pad & diagonalize) =>
                 [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
                   0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
                   0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
                   0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
            """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
            self, query, key, num_heads, window_overlap
    ):
        """
            Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
            """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                                :, :, :window_overlap, : window_overlap + 1
                                                                ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                               :, -1, window_overlap:, : window_overlap + 1
                                                               ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                               :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                               ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                              :, 0, : window_overlap - 1,
                                                                              1 - window_overlap:
                                                                              ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
            self, attn_probs, value, num_heads, window_overlap
    ):
        """
            Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
            same shape as `attn_probs`
            """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    # def forward(self, x, mask):
    def forward(self, x):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        # q, qx_mask = self.query_conv(x, mask)
        q = self.query_conv(x)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k = self.key_conv(x)
        k = self.key_norm(k)
        # v, _ = self.value_conv(x, mask)
        v = self.value_conv(x)
        v = self.value_norm(v)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # # kv_mask -> B, T'', 1
        # inverse_kv_mask = torch.logical_not(
        #     kv_mask[:, :, :, None].view(B, -1, 1))
        # # 0 for valid slot, -inf for masked ones
        # float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
        #     inverse_kv_mask, -1e4)
        # # compute the diagonal mask (for each local window)
        # diagonal_mask = self._sliding_chunks_query_key_matmul(
        #     float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
        #     float_inverse_kv_mask,
        #     1,
        #     self.window_overlap
        # )
        # att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        # att = att.masked_fill(
        #     torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out))
        # out = self.proj_drop(self.proj(out)) * qx_mask.float()
        return out
        # return out, qx_mask
