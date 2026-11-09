import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
import cv2


class BitwiseSelfCorrection(object):
    def __init__(self, vae, args):
        self.noise_apply_layers = args.noise_apply_layers
        self.noise_apply_requant = args.noise_apply_requant
        self.noise_apply_strength = args.noise_apply_strength
        self.apply_spatial_patchify = args.apply_spatial_patchify
        self.vae = vae
        self.debug_bsc = args.debug_bsc
        self.flip_mode = args.flip_mode

        # single-boundary flip:
        # randomly sample k in [0, 12].
        # scale <= k: clean GT prefix
        # scale == k + 1: apply one-shot flip
        # scale > k + 1: no extra flip, only natural error propagation
        self.random_prefix_min = getattr(args, "random_prefix_min", 0)
        self.random_prefix_max = getattr(args, "random_prefix_max", 12)

        # flip upper bound for the first predicted scale k+1
        self.boundary_flip_prob = getattr(args, "boundary_flip_prob", 0.05)
                

    def flip_requant(
        self,
        vae_scale_schedule,
        inp_B3HW,
        raw_features,
        device,
        entropy_maps=None,          # 新增：每层每bit的翻转概率图，shape要和bit_indices一致
        disable_noise=False,        # 新增：第一次probe时设True，不做任何flip
    ):
        with torch.amp.autocast('cuda', enabled=False):
            B = raw_features.shape[0]

            # ------------------------------------------------------------
            # For single-boundary flip:
            # randomly sample transmitted prefix boundary k.
            # k in [0, 12] for 13 scales.
            # If k == 12, all scales are regarded as transmitted, so no flip.
            # ------------------------------------------------------------
            if self.flip_mode == "single_boundary":
                total_scales = len(vae_scale_schedule)

                k_min = max(0, int(self.random_prefix_min))
                k_max = min(int(self.random_prefix_max), total_scales - 1)

                sampled_prefix_k = np.random.randint(k_min, k_max + 1)



            else:
                sampled_prefix_k = -1

            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2)   # [B, d, 1, H, W]
            else:
                codes_out = raw_features

            cum_var_input = 0
            gt_all_bit_indices = []
            pred_all_bit_indices = []
            x_BLC_wo_prefix = []

            for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                # 1) 当前scale residual
                residual = codes_out - cum_var_input

                if si != len(vae_scale_schedule) - 1:
                    residual = F.interpolate(
                        residual,
                        size=vae_scale_schedule[si],
                        mode=self.vae.quantizer.z_interplote_down
                    ).contiguous()

                # 2) 当前scale量化
                quantized, _, bit_indices, loss = self.vae.quantizer.lfq(residual)
                # quantized:   [B, d, 1, h, w]
                # bit_indices: [B, 1, h, w, d]

                gt_all_bit_indices.append(bit_indices)

                # 3) 决定当前scale是否做flip
                pred_bit_indices = bit_indices

                if (not disable_noise) and (si < self.noise_apply_layers):
                    if self.flip_mode == "random":
                        noise_apply_strength = np.random.randint(
                            0, int(100 * self.noise_apply_strength) + 1
                        ) * 0.01

                        mask = torch.rand(*bit_indices.shape, device=device) < noise_apply_strength


                    elif self.flip_mode == "scale_only":
                        active_scales = vae_scale_schedule[:self.noise_apply_layers]

                        cur_tokens = pt * ph * pw
                        max_tokens = max(_pt * _ph * _pw for (_pt, _ph, _pw) in active_scales)

                        scale_ratio = float(cur_tokens) / float(max_tokens)
                        s_min = 0.2
                        beta = 0.5
                        scale_factor = s_min + (1.0 - s_min) * (scale_ratio ** beta)


                        p_max_global = self.noise_apply_strength
                        flip_prob_upper = p_max_global * scale_factor

                        # 在 [0, flip_prob_upper] 之间随机采样当前 scale 的实际 flip 概率
                        flip_prob_value = np.random.uniform(0.0, flip_prob_upper)

                        flip_prob = torch.full(
                            bit_indices.shape,
                            fill_value=flip_prob_value,
                            device=device,
                            dtype=torch.float32,
                        )

                        rand_map = torch.rand_like(flip_prob)
                        mask = rand_map < flip_prob

                    elif self.flip_mode == "entropy_with_scale":
                        if entropy_maps is None:
                            raise ValueError("flip_mode='entropy_with_scale' requires entropy_maps, but got None.")

                        entropy_norm = entropy_maps[si].to(device=device, dtype=torch.float32)
                        entropy_norm = entropy_norm.clamp_(0.0, 1.0)

                        active_scales = vae_scale_schedule[:self.noise_apply_layers]

                        cur_tokens = pt * ph * pw
                        max_tokens = max(_pt * _ph * _pw for (_pt, _ph, _pw) in active_scales)

                        scale_ratio = float(cur_tokens) / float(max_tokens)
                        s_min = 0.2
                        beta = 0.5
                        scale_factor = s_min + (1.0 - s_min) * (scale_ratio ** beta)

                        gamma = 1.5
                        entropy_factor = entropy_norm ** gamma

                        p_max_global = self.noise_apply_strength
                        flip_prob = p_max_global * scale_factor * entropy_factor
                        flip_prob = flip_prob.clamp_(0.0, p_max_global)

                        rand_map = torch.rand_like(flip_prob)
                        mask = rand_map < flip_prob


                    elif self.flip_mode == "single_boundary":
                        """
                        Random single-boundary flip.

                        Randomly sample k from [0, 12].
                        scale <= k:
                            clean GT prefix, no flip.
                        scale == k + 1:
                            apply one-shot bit flip.
                        scale > k + 1:
                            no extra artificial flip, but it is computed based on
                            contaminated cum_var_input if k+1 was flipped.
                        """
                        k = int(sampled_prefix_k)

                        # 当前 scale 属于哪种状态
                        if si <= k:
                            scale_state = "TRANSMITTED_GT_PREFIX"
                        elif si == k + 1:
                            scale_state = "FIRST_PREDICTED_FLIP"
                        else:
                            scale_state = "FOLLOWING_PREDICTED_NO_EXTRA_FLIP"

                        # Only flip the first untransmitted / first predicted scale.
                        # If k == 12, then k + 1 does not exist, so all scales are clean.
                        if si == k + 1:
                            flip_prob_upper = float(self.noise_apply_strength) * float(self.boundary_flip_prob)

                            # bounded random flip probability
                            flip_prob_value = np.random.uniform(0.0, flip_prob_upper)

                            flip_prob = torch.full(
                                bit_indices.shape,
                                fill_value=flip_prob_value,
                                device=device,
                                dtype=torch.float32,
                            )

                            rand_map = torch.rand_like(flip_prob)
                            mask = rand_map < flip_prob

                            actual_flip_rate = mask.float().mean().item()
                            flipped_bits = int(mask.sum().item())
                            total_bits = int(mask.numel())

                        else:
                            mask = torch.zeros_like(bit_indices, dtype=torch.bool, device=device)




                    else:
                        raise ValueError(f"Unknown flip_mode: {self.flip_mode}")


                    pred_bit_indices = bit_indices.clone()
                    pred_bit_indices[mask] = 1 - pred_bit_indices[mask]

                    if self.noise_apply_requant and mask.any():
                        # if self.flip_mode == "single_boundary":
                        #     print(
                        #         f"[Single-boundary Requant] "
                        #         f"scale={si:02d}, "
                        #         f"requant=True, "
                        #         f"flipped_bits={int(mask.sum().item())}, "
                        #         f"quantized will be replaced by flipped-code and enter cum_var_input"
                        #     )

                        quantized = self.vae.quantizer.lfq.indices_to_codes(
                            pred_bit_indices, label_type='bit_label'
                        )
                    # else:
                    #     if self.flip_mode == "single_boundary":
                    #         print(
                    #             f"[Single-boundary Requant] "
                    #             f"scale={si:02d}, "
                    #             f"requant=False, "
                    #             f"mask_any={bool(mask.any().item()) if torch.is_tensor(mask) else False}, "
                    #             f"use clean quantized code"
                    #         )
                else:
                    pred_bit_indices = bit_indices

                pred_all_bit_indices.append(pred_bit_indices)

                # 4) 更新累计前缀（连续code空间）
                cum_var_input = cum_var_input + F.interpolate(
                    quantized,
                    size=vae_scale_schedule[-1],
                    mode=self.vae.quantizer.z_interplote_up
                ).contiguous()

                if self.flip_mode == "single_boundary":
                    with torch.no_grad():
                        cur_norm = quantized.detach().float().norm().item()
                        cum_norm = cum_var_input.detach().float().norm().item()

                    # print(
                    #     f"[Single-boundary CumUpdate] "
                    #     f"scale={si:02d}, "
                    #     f"quantized_norm={cur_norm:.4f}, "
                    #     f"cum_var_input_norm={cum_norm:.4f}"
                    # )


                # 5) 构造给“下一层”用的prefix输入
                if si < len(vae_scale_schedule) - 1:
                    this_scale_input = F.interpolate(
                        cum_var_input,
                        size=vae_scale_schedule[si + 1],
                        mode=self.vae.quantizer.z_interplote_up
                    ).contiguous()

                    if self.apply_spatial_patchify:
                        # (B,d,1,H,W) -> (B,d,H,W) -> (B,4d,H/2,W/2)
                        this_scale_input = torch.nn.functional.pixel_unshuffle(
                            this_scale_input.squeeze(-3), 2
                        )

                    # -> [B, L, C]
                    x_BLC_wo_prefix.append(
                        this_scale_input.reshape(*this_scale_input.shape[:2], -1).permute(0, 2, 1)
                    )

            # 6) GT bit labels整理成训练目标格式
            if self.apply_spatial_patchify:
                gt_ms_idx_Bl = []
                for item in gt_all_bit_indices:
                    # item: [B,1,H,W,d]
                    item = item.squeeze(1).permute(0, 3, 1, 2).contiguous()   # [B,d,H,W]
                    item = torch.nn.functional.pixel_unshuffle(item, 2)       # [B,4d,H/2,W/2]
                    item = item.permute(0, 2, 3, 1).reshape(B, -1, 4 * self.vae.codebook_dim)
                    gt_ms_idx_Bl.append(item)
            else:
                gt_ms_idx_Bl = [item.reshape(B, -1, self.vae.codebook_dim) for item in gt_all_bit_indices]

            x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, 1)

            if self.debug_bsc:
                self.visualize(vae_scale_schedule, inp_B3HW, gt_all_bit_indices, pred_all_bit_indices)

        return x_BLC_wo_prefix, gt_ms_idx_Bl

    def visualize(self, vae_scale_schedule, inp_B3HW, gt_all_bit_indices, pred_all_bit_indices):
        gt_img = (inp_B3HW.squeeze(-3) + 1) / 2 * 255
        gt_img = gt_img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)[:, :, ::-1]
        recons_img_2 = labels2image(gt_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        recons_img_3 = labels2image(pred_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        cat_image = np.concatenate([gt_img, recons_img_2, recons_img_3], axis=1)
        save_path = osp.abspath('non_teacher_force.jpg')
        cv2.imwrite(save_path, cat_image)
        print(f'Save to {save_path}')
        import pdb; pdb.set_trace()
        print(cat_image.shape)