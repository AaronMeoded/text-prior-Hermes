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
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: list indicating downsample scale per level
            kernel_size: 3D kernel sizes per level
            block: block type
            num_block: number of blocks per stage
            pool: pooling vs strided conv
            norm: normalization type
            tn: number of task priors
            mn: number of modality priors
            embed_dim: dimension of input text embeddings
            text_prior_num: number of text-prior tokens per scale
        '''
        # store for forward
        self.base_ch = base_ch
        self.embed_dim = embed_dim
        self.text_prior_num = text_prior_num

        block = get_block(block)
        norm = get_norm(norm)
    
        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)
        
        self.down1 = down_block(
            base_ch, 2*base_ch, num_block=num_block[0], block=block,
            pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm
        )
        
        self.down2 = down_block(
            2*base_ch, 4*base_ch, num_block=num_block[1], block=block,
            pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm
        )
        self.prior_init_fuse_2 = PriorInitFusionLayer(4*base_ch, 4*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn)
        
        self.down3 = down_block(
            4*base_ch, 8*base_ch, num_block=num_block[2], block=block,
            pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm
        )
        self.prior_init_fuse_3 = PriorInitFusionLayer(8*base_ch, 8*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn)
        
        self.down4 = down_block(
            8*base_ch, 10*base_ch, num_block=num_block[3], block=block,
            pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm
        )
        self.prior_init_fuse_4 = PriorInitFusionLayer(10*base_ch, 10*base_ch, block_num=4, task_prior_num=tn, modality_prior_num=mn)

        self.up1 = up_block(
            10*base_ch, 8*base_ch, num_block=num_block[2], block=block,
            up_scale=scale[3], kernel_size=kernel_size[3], norm=norm
        )
        self.prior_fuse_5 = PriorInitFusionLayer(8*base_ch, 8*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn)
        
        self.up2 = up_block(
            8*base_ch, 4*base_ch, num_block=num_block[1], block=block,
            up_scale=scale[2], kernel_size=kernel_size[2], norm=norm
        )
        self.prior_fuse_6 = PriorInitFusionLayer(4*base_ch, 4*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn)

        self.up3 = up_block(
            4*base_ch, 2*base_ch, num_block=num_block[0], block=block,
            up_scale=scale[1], kernel_size=kernel_size[1], norm=norm
        )
        self.up4 = up_block(
            2*base_ch, base_ch, num_block=2, block=block,
            up_scale=scale[0], kernel_size=kernel_size[0], norm=norm
        )
    
        self.out = HierarchyPriorClassifier(34*base_ch, base_ch)
        self.mod_out = ModalityClassifier(34*base_ch, mn)

        # OpenAI text embedding projections: one head per fusion stage
        self.text_heads = nn.ModuleList([
            nn.Linear(embed_dim, text_prior_num * (4*base_ch)),
            nn.Linear(embed_dim, text_prior_num * (8*base_ch)),
            nn.Linear(embed_dim, text_prior_num * (10*base_ch)),
            nn.Linear(embed_dim, text_prior_num * (8*base_ch)),
            nn.Linear(embed_dim, text_prior_num * (4*base_ch)),
        ])

    def forward(self, x, tgt_idx, mod_idx, raw_text_embeds):
        # raw_text_embeds: (B, embed_dim)
        tn = tgt_idx.shape[1]
        mn = mod_idx.shape[1]

        # Prepare dynamic text priors for each fusion scale
        B = raw_text_embeds.size(0)
        prior_dims = [4*self.base_ch, 8*self.base_ch, 10*self.base_ch, 8*self.base_ch, 4*self.base_ch]
        text_priors = []
        for head, pd in zip(self.text_heads, prior_dims):
            flat = head(raw_text_embeds)             # (B, text_prior_num * pd)
            text_priors.append(flat.view(B, self.text_prior_num, pd))

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3, priors_2 = self.prior_init_fuse_2(x3, tgt_idx, mod_idx, text_priors[0])

        x4 = self.down3(x3)
        x4, priors_3 = self.prior_init_fuse_3(x4, tgt_idx, mod_idx, text_priors[1])

        x5 = self.down4(x4)
        x5, priors_4 = self.prior_init_fuse_4(x5, tgt_idx, mod_idx, text_priors[2])

        out = self.up1(x5, x4)
        out, priors_5 = self.prior_fuse_5(out, tgt_idx, mod_idx, text_priors[3])

        out = self.up2(out, x3)
        out, priors_6 = self.prior_fuse_6(out, tgt_idx, mod_idx, text_priors[4])

        out = self.up3(out, x2)
        out = self.up4(out, x1)

        # Task priors for segmentation
        task_priors_2 = priors_2[:, :tn, :]
        task_priors_3 = priors_3[:, :tn, :]
        task_priors_4 = priors_4[:, :tn, :]
        task_priors_5 = priors_5[:, :tn, :]
        task_priors_6 = priors_6[:, :tn, :]
        task_list = [task_priors_2, task_priors_3, task_priors_4, task_priors_5, task_priors_6]
        out = self.out(out, task_list)

        # Modality priors for modality classification
        mod_priors_2 = priors_2[:, tn:tn+mn, :]
        mod_priors_3 = priors_3[:, tn:tn+mn, :]
        mod_priors_4 = priors_4[:, tn:tn+mn, :]
        mod_priors_5 = priors_5[:, tn:tn+mn, :]
        mod_priors_6 = priors_6[:, tn:tn+mn, :]
        mod_list = [mod_priors_2, mod_priors_3, mod_priors_4, mod_priors_5, mod_priors_6]
        mod_out = self.mod_out(mod_list)

        return out, mod_out