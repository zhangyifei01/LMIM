# References:
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F


class MAEEncoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 img_size=(32, 128),
                 patch_size=4,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans,
                                      embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.img_size = img_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # ---------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**.5),
            cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))


        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) # noqa
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]  # p = 8
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def random_multicol_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking on columns.
        x: [N, L, D], where L = 8 * 32.
        """
    
        N, L, D = x.shape
        num_columns = 32
        num_rows = L // num_columns  # Should be 8
        num_blocks = 16  # 32/2

        len_keep = int(num_blocks * (1 - mask_ratio))

        # Reshape x to [N, num_rows, num_columns, D]
        x = x.view(N, num_rows, num_columns, D)
        x = x.view(N, num_rows, num_blocks, 2, D)

        # Generate noise for each column
        noise = torch.rand(N, num_blocks, device=x.device)

        # Sort noise to shuffle columns
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_restore = ids_restore.unsqueeze(1).unsqueeze(3).expand(-1, num_rows, -1, 2)

        # Keep the first subset of columns
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(1).unsqueeze(3).unsqueeze(-1).expand(-1, num_rows, -1, 2, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, num_rows, num_blocks, 2], device=x.device)
        mask[:, :, :len_keep, :] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore)

        return x_masked, mask, ids_restore
    

    def forward_encoder(self, x, mask_ratio=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio is not None:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if mask_ratio is not None:
            return x, mask, ids_restore
        else:
            return x, None, None
        
    def forward_encoder_no_masking(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs, mask_ratio=None):
        
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        return latent, mask, ids_restore
    

class LMIMDecoder(nn.Module):
    """ SA-CA-FFN Decoder
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.GELU,
                 layer_norm_eps=1e-6, batch_first=True, norm_first=False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation()

    # SA
    def _selfatt(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    # CA
    def _crossatt(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)

    # FFN
    def _ffn(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,):
        x = tgt
        x = self.norm1(x + self._selfatt(x, tgt_mask, tgt_key_padding_mask))
        x = self.norm2(x + self._crossatt(x, memory, memory_mask, memory_key_padding_mask))
        x = self.norm3(x + self._ffn(x))
        return x

    
class LMIMViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 img_size=(32, 128),
                 patch_size=4,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.model_maeenc = MAEEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, norm_layer)
        num_patches = self.model_maeenc.patch_embed.num_patches
        # ---------------------------------------------------------------------
        self.model_maeenc_momentum = MAEEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, norm_layer)

        for param_b, param_m in zip(self.model_maeenc.parameters(), self.model_maeenc_momentum.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        # ---------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed2 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     Block(
        #         decoder_embed_dim,
        #         decoder_num_heads,
        #         mlp_ratio,
        #         qkv_bias=True,
        #         qk_scale=None,
        #         norm_layer=norm_layer) for i in range(decoder_depth)
        # ])
        self.decoder_blocks = nn.ModuleList([
            LMIMDecoder(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, embed_dim,
            bias=True)  # decoder to patch
        # ---------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.model_maeenc.parameters(), self.model_maeenc_momentum.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.model_maeenc.patch_embed.num_patches**.5),
            cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) # noqa
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.model_maeenc.patch_embed.patch_size[0]  # p = 8
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.model_maeenc.patch_size
        h = self.model_maeenc.img_size[0] // p
        w = self.model_maeenc.img_size[1] // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    
    def _apply_decoder_blocks(self, q, kv):
        for blk in self.decoder_blocks:
            q = blk(q, kv)
        return q

    def forward_decoder(self, x, ids_restore, x_no):
        # embed tokens
        x = self.decoder_embed(x)
        x_no = self.decoder_embed2(x_no)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1,
                                                   x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        # cross decoder
        x = self._apply_decoder_blocks(x, x_no)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, target, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        mask = mask.view(mask.shape[0], -1)
        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, imgs_nomask, mask_ratio=0.75):
        latent, mask, ids_restore = self.model_maeenc(imgs, mask_ratio)
        mask_cls_token = latent[:, 0, :]
        latent_nomask = self.model_maeenc.forward_encoder_no_masking(imgs_nomask)
        nomask_cls_token = latent_nomask[:, 0, :]
        pred = self.forward_decoder(latent, ids_restore, latent_nomask)  # [N, L, p*p*3]
        with torch.no_grad(): # no gradient
            target, mask_t, ids_restore_t = self.model_maeenc_momentum(imgs)
            target = target[:, 1:, :]
            target = F.layer_norm(target, (target.size(-1),))  # normalize over feature-dim
            
        loss_cls = (mask_cls_token - nomask_cls_token)**2
        loss_cls = loss_cls.mean()

        loss = self.forward_loss(imgs, pred, target, mask)
        loss = loss + loss_cls
        return loss, pred, mask, imgs


def mae_vit_small_patch4_dec512d8b(**kwargs):
    model = LMIMViT(
        img_size=(32, 128),
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def mae_vit_base_patch4_dec512d8b(**kwargs):
    model = LMIMViT(
        img_size=(32, 128),
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


# set recommended archs
mae_vit_small_patch4 = mae_vit_small_patch4_dec512d8b  # decoder: 512 dim, 8 blocks # noqa
mae_vit_base_patch4 = mae_vit_base_patch4_dec512d8b  # decoder: 512 dim, 8 blocks # noqa
