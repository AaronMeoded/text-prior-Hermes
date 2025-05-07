import torch
import torch.nn as nn
import torch.nn.functional as F
from .hermes_unet_utils import inconv, down_block, up_block, PriorInitFusionLayer
from .hermes_utils import HierarchyPriorClassifier, ModalityClassifier
from .utils import get_block, get_norm
import pdb

class Hermes_UNet(nn.Module):
    def __init__(
        self,
        in_ch,
        base_ch,
        scale=[2,2,2,2],
        kernel_size=[3,3,3,3],
        block='BasicBlock',
        num_block=[2,2,2,2],
        pool=True,
        norm='in',
        tn=72,
        mn=6,
        embed_dim=1536,
        text_prior_num=4
    ):
        super().__init__()
        # store parameters for forward use
        self.base_ch = base_ch
        self.embed_dim = embed_dim
        self.text_prior_num = text_prior_num

        # resolve block and norm functions
        block_fn = get_block(block)
        norm_fn = get_norm(norm)

        # U-Net encoder
        self.inc = inconv(in_ch, base_ch, block=block_fn, kernel_size=kernel_size[0], norm=norm_fn)
        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block[0], block=block_fn,
                                pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm_fn)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block[1], block=block_fn,
                                pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm_fn)
        self.prior_init_fuse_2 = PriorInitFusionLayer(
            4*base_ch, 4*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn
        )
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block[2], block=block_fn,
                                pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm_fn)
        self.prior_init_fuse_3 = PriorInitFusionLayer(
            8*base_ch, 8*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn
        )
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block[3], block=block_fn,
                                pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm_fn)
        self.prior_init_fuse_4 = PriorInitFusionLayer(
            10*base_ch, 10*base_ch, block_num=4, task_prior_num=tn, modality_prior_num=mn
        )

        # U-Net decoder
        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block[2], block=block_fn,
                            up_scale=scale[3], kernel_size=kernel_size[3], norm=norm_fn)
        self.prior_fuse_5 = PriorInitFusionLayer(
            8*base_ch, 8*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn
        )
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block[1], block=block_fn,
                            up_scale=scale[2], kernel_size=kernel_size[2], norm=norm_fn)
        self.prior_fuse_6 = PriorInitFusionLayer(
            4*base_ch, 4*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn
        )
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block[0], block=block_fn,
                            up_scale=scale[1], kernel_size=kernel_size[1], norm=norm_fn)
        self.up4 = up_block(2*base_ch, base_ch, num_block=2, block=block_fn,
                            up_scale=scale[0], kernel_size=kernel_size[0], norm=norm_fn)

        # classification heads
        self.out = HierarchyPriorClassifier(34*base_ch, base_ch)
        self.mod_out = ModalityClassifier(34*base_ch, mn)

        # text embedding linear heads: embed_dim -> text_prior_num * prior_dim_per_scale
        self.text_heads = nn.ModuleList([
            nn.Linear(embed_dim, text_prior_num * (4 * base_ch)),
            nn.Linear(embed_dim, text_prior_num * (8 * base_ch)),
            nn.Linear(embed_dim, text_prior_num * (10 * base_ch)),
            nn.Linear(embed_dim, text_prior_num * (8 * base_ch)),
            nn.Linear(embed_dim, text_prior_num * (4 * base_ch)),
        ])

    def forward(self, x, tgt_idx, mod_idx, raw_text_embeds):
        # x: (B, in_ch, D, H, W)
        # raw_text_embeds: (B, embed_dim)
        tn = tgt_idx.shape[1]
        mn = mod_idx.shape[1]
        B = raw_text_embeds.size(0)

        # build text_prior tokens for each scale
        prior_dims = [4 * self.base_ch, 8 * self.base_ch,
                      10 * self.base_ch, 8 * self.base_ch,
                      4 * self.base_ch]
        text_priors = []
        for head, pd in zip(self.text_heads, prior_dims):
            flat = head(raw_text_embeds)  # (B, text_prior_num * pd)
            text_priors.append(flat.view(B, self.text_prior_num, pd))

        # encoder + fusion 2
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3, priors_2 = self.prior_init_fuse_2(x3, tgt_idx, mod_idx, text_priors[0])

        # encoder + fusion 3
        x4 = self.down3(x3)
        x4, priors_3 = self.prior_init_fuse_3(x4, tgt_idx, mod_idx, text_priors[1])

        # encoder + fusion 4
        x5 = self.down4(x4)
        x5, priors_4 = self.prior_init_fuse_4(x5, tgt_idx, mod_idx, text_priors[2])

        # decoder + fusion 5
        out = self.up1(x5, x4)
        out, priors_5 = self.prior_fuse_5(out, tgt_idx, mod_idx, text_priors[3])

        # decoder + fusion 6
        out = self.up2(out, x3)
        out, priors_6 = self.prior_fuse_6(out, tgt_idx, mod_idx, text_priors[4])

        # finish decoding
        out = self.up3(out, x2)
        out = self.up4(out, x1)

        # segmentation head: use task priors
        task_priors = [p[:, :tn, :] for p in [priors_2, priors_3, priors_4, priors_5, priors_6]]
        seg_out = self.out(out, task_priors)

        # modality head: use modality priors
        mod_priors = [p[:, tn:tn+mn, :] for p in [priors_2, priors_3, priors_4, priors_5, priors_6]]
        mod_out = self.mod_out(mod_priors)

        return seg_out, mod_out