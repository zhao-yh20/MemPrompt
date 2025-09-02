import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import torch.nn.functional as F
import src.modules.vision_transformer_prompts as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from src.modules import heads, objectives, vilt_utils


class PromptMemory(nn.Module):
    def __init__(self, config, is_shared=False):
        super(PromptMemory, self).__init__()
        self.emb_dim = config["hidden_size"]
        self.key_dim = config.get("route_hidden_size", self.emb_dim)

        # Prompt memory parameters
        self.memory_size = config["mem_size"]  # Number of prompt candidates
        self.prompt_length = config["shared_prompt_length" if is_shared else "gen_prompt_length"]
        self.position = config["prompt_type"]
        self.top_k = config["top_k"]  # Number of selected prompts

        if self.key_dim != self.emb_dim:
            self.x_proj = nn.Linear(self.emb_dim, self.key_dim, bias=False)

        # Initialize prompt and keys
        self.prompt_memory = nn.Parameter(torch.empty(self.memory_size, self.prompt_length, self.emb_dim))
        nn.init.kaiming_normal_(self.prompt_memory, a=math.sqrt(5))

        self.prompt_keys = nn.Parameter(torch.empty(self.memory_size, self.key_dim))
        nn.init.kaiming_normal_(self.prompt_keys, a=math.sqrt(5))


    def forward(self, x_query):
        if self.key_dim != self.emb_dim:
            x_proj = self.x_proj(x_query)
        else:
            x_proj = x_query

        # Normalize
        x_proj_norm = nn.functional.normalize(x_proj, dim=-1)
        key_norm = nn.functional.normalize(self.prompt_keys, dim=-1)

        # retrieve top-k prompts
        similarity_scores = torch.matmul(x_proj_norm, key_norm.T)
        top_k = min(self.top_k, self.memory_size)
        top_k_scores, top_k_indices = torch.topk(similarity_scores, k=top_k, dim=1, largest=True, sorted=False)

        # Gather top-k prompts and keys
        selected_prompts = self.prompt_memory[top_k_indices]
        selected_keys = self.prompt_keys[top_k_indices]  # [batch_size, top_k, key_dim]

        x_proj_norm = x_proj_norm.unsqueeze(1)
        selected_keys_norm = nn.functional.normalize(selected_keys, dim=2)
        refined_scores = torch.sum(x_proj_norm * selected_keys_norm, dim=2)
        weights = nn.functional.softmax(refined_scores, dim=1)
        weights = weights.unsqueeze(2).unsqueeze(3)

        prompt_output = torch.sum(selected_prompts * weights, dim=1)  # [batch_size, prompt_len, emb_dim]
        return prompt_output



class MemPromptTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)


        self.prompt_type = self.hparams.config["prompt_type"]
        self.gen_prompt_length = self.hparams.config["gen_prompt_length"]
        self.gen_prompt_layers = self.hparams.config["gen_prompt_layers"]
        self.shared_prompt_layers = self.hparams.config["shared_prompt_layers"]
        self.shared_prompt_length = self.hparams.config["shared_prompt_length"]
        self.hidden_size = config["hidden_size"]
        self.bottleneck_size = self.hidden_size // 16

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
                self.hparams.config["load_path"] != ""
                and not self.hparams.config["test_only"]
                and not self.hparams.config["finetune_first"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # since the pre-trained max_text_len is 40,
            # we upsample the weight of position embedding to determined max_text_len
            if config["max_text_len"] != 40:
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1, -1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1, 1, 40, 768), size=(config["max_text_len"], 768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]

        # Initialize task-specific classifiers
        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.hatememes_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.food101_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)

        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:  ##   finetune_first = False
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print("use pre-finetune model")



        ####################################################################################################################################
        #  ## Initialize PromptMemory
        self.gen_v_prompt_memory = nn.ModuleList(
            [PromptMemory(config, is_shared=False) for _ in range(len(self.gen_prompt_layers))])
        self.gen_t_prompt_memory = nn.ModuleList(
            [PromptMemory(config, is_shared=False) for _ in range(len(self.gen_prompt_layers))])
        self.gen_v_prompt_memory.apply(objectives.init_weights)
        self.gen_t_prompt_memory.apply(objectives.init_weights)

        self.shared_missing_t = nn.ParameterList(
            nn.Parameter(nn.init.normal_(torch.empty(self.shared_prompt_length, self.hidden_size), std=0.02))
            for _ in range(len(self.shared_prompt_layers)))
        self.shared_missing_v = nn.ParameterList(
            nn.Parameter(nn.init.normal_(torch.empty(self.shared_prompt_length, self.hidden_size), std=0.02))
            for _ in range(len(self.shared_prompt_layers)))

        self.bottleneck_texts = nn.ModuleList(nn.Sequential(
                nn.Linear(self.hidden_size, self.bottleneck_size),
                nn.GELU(),
                nn.Linear(self.bottleneck_size, self.hidden_size))
                for _ in range(len(self.shared_prompt_layers)))
        self.bottleneck_images = nn.ModuleList(nn.Sequential(
                nn.Linear(self.hidden_size, self.bottleneck_size),
                nn.GELU(),
                nn.Linear(self.bottleneck_size, self.hidden_size))
                for _ in range(len(self.shared_prompt_layers)))

        self.bottleneck_images.apply(objectives.init_weights)
        self.bottleneck_texts.apply(objectives.init_weights)

        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.text_embeddings.parameters():
            param.requires_grad = False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        self.records = {}


    def infer(self, batch, mask_text=False, mask_image=False, image_token_type_idx=1, image_embeds=None, image_masks=None, is_train=None, ):

        imgkey = f"image_{image_token_type_idx - 1}" if f"image_{image_token_type_idx - 1}" in batch else "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        img = batch[imgkey][0]

        if image_embeds is None and image_masks is None:
            (image_embeds, image_masks, patch_index, image_labels) = self.transformer.visual_embed(
                img, max_image_len=self.hparams.config["max_image_len"], mask_it=mask_image,
            )

        else:
            patch_index, image_labels = (None, None,)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)
        masks = co_masks
        x = co_embeds

        batch_size = x.size(0)
        device = x.device
        emb_dim = x.size(-1)
        d_prompt_t_len = 0
        d_prompt_v_len = 0
        shared_prompt_t_len = 0
        shared_prompt_v_len = 0
        orthogonality_loss = 0.0

        missing_type = batch["missing_type"]
        if not isinstance(missing_type, torch.Tensor):
            missing_type = torch.tensor(missing_type, device=device)
        else:
            missing_type = missing_type.to(device)

        text_len = text_embeds.shape[1]
        image_len = image_embeds.shape[1]

        for layer_idx, blk in enumerate(self.transformer.blocks):

            text_start = d_prompt_t_len + shared_prompt_t_len
            text_end = d_prompt_t_len + shared_prompt_t_len + text_len
            image_start = text_end + d_prompt_v_len + shared_prompt_v_len
            image_end = text_end + d_prompt_v_len + shared_prompt_v_len + image_len

            if layer_idx in self.gen_prompt_layers:
                text_feats = torch.max(x[:, text_start:text_end, :], dim=1)[0]
                image_feats = torch.max(x[:, image_start:image_end, :], dim=1)[0]

                t_prompts = self.gen_t_prompt_memory[layer_idx](text_feats)
                v_prompts = self.gen_v_prompt_memory[layer_idx](image_feats)
                v_len = v_prompts.size(1)
                t_len = t_prompts.size(1)

            else:
                t_prompts, v_prompts = None, None
                v_len, t_len = 0, 0

            if layer_idx in self.shared_prompt_layers:
                shared_prompts_t, shared_prompts_v = None, None
                bottleneck_text = self.bottleneck_texts[layer_idx]
                bottleneck_image = self.bottleneck_images[layer_idx]

                for i in range(len(missing_type)):

                    if missing_type[i] == 0:
                        # s_prompt = self.shared_complete[layer_idx]
                        s_prompt = self.shared_missing_t[layer_idx] + self.shared_missing_v[layer_idx]
                        s_prompt_t = bottleneck_text(s_prompt)
                        s_prompt_v = bottleneck_image(s_prompt)

                    if missing_type[i] == 1:
                        s_prompt = self.shared_missing_t[layer_idx]
                        s_prompt_t = bottleneck_text(s_prompt)
                        s_prompt_v = bottleneck_image(s_prompt)
                    if missing_type[i] == 2:
                        s_prompt = self.shared_missing_v[layer_idx]
                        s_prompt_t = bottleneck_text(s_prompt)
                        s_prompt_v = bottleneck_image(s_prompt)

                    if shared_prompts_t is None and shared_prompts_v is None:
                        shared_prompts_t = s_prompt_t.unsqueeze(0)
                        shared_prompts_v = s_prompt_v.unsqueeze(0)
                    else:
                        shared_prompts_t = torch.cat((shared_prompts_t, s_prompt_t.unsqueeze(0)), dim=0)
                        shared_prompts_v = torch.cat((shared_prompts_v, s_prompt_v.unsqueeze(0)), dim=0)

                shared_len_t = shared_prompts_t.size(1)
                shared_len_v = shared_prompts_v.size(1)

            else:
                shared_prompts_t, shared_prompts_v = None, None
                shared_len_t, shared_len_v = 0, 0

            prompt_shared_t_masks = torch.ones(batch_size, shared_len_t, dtype=masks.dtype, device=device)
            prompt_shared_v_masks = torch.ones(batch_size, shared_len_v, dtype=masks.dtype, device=device)
            prompt_text_masks = torch.ones(batch_size, t_len, dtype=masks.dtype, device=device)
            prompt_image_masks = torch.ones(batch_size, v_len, dtype=masks.dtype, device=device)

            # # update x
            if t_prompts is not None and v_prompts is not None and shared_prompts_t is not None and shared_prompts_v is not None:
                masks = torch.cat([prompt_shared_t_masks, prompt_text_masks,
                                   masks[:, : d_prompt_t_len + shared_prompt_t_len + text_len],
                                   prompt_shared_v_masks, prompt_image_masks,
                                   masks[:, d_prompt_t_len + shared_prompt_t_len + text_len:]], dim=1)
                x = torch.cat(
                    [shared_prompts_t, t_prompts, x[:, : d_prompt_t_len + shared_prompt_t_len + text_len, :],
                     shared_prompts_v, v_prompts, x[:, d_prompt_t_len + shared_prompt_t_len + text_len:, :]], dim=1)

            elif t_prompts is not None and v_prompts is not None and shared_prompts_t is None and shared_prompts_v is None:
                masks = torch.cat([masks[:, :shared_prompt_t_len],
                                   prompt_text_masks,
                                   masks[:, shared_prompt_t_len:d_prompt_t_len + shared_prompt_t_len + text_len + shared_prompt_v_len],
                                   prompt_image_masks,
                                   masks[:, d_prompt_t_len + shared_prompt_t_len + text_len + shared_prompt_v_len:]], dim=1)
                x = torch.cat([x[:, :shared_prompt_t_len],
                               t_prompts,
                               x[:, shared_prompt_t_len:d_prompt_t_len + shared_prompt_t_len + text_len, :],
                               x[:, d_prompt_t_len + shared_prompt_t_len + text_len:d_prompt_t_len + shared_prompt_t_len + text_len + shared_prompt_v_len:, :],
                               v_prompts,
                               x[:, d_prompt_t_len + shared_prompt_t_len + text_len + shared_prompt_v_len:, :]], dim=1)

            elif t_prompts is None and v_prompts is None and shared_prompts_t is not None and shared_prompts_v is not None:
                masks = torch.cat([prompt_shared_t_masks, masks[:, : d_prompt_t_len + shared_prompt_t_len + text_len],
                                   prompt_shared_v_masks, masks[:, d_prompt_t_len + shared_prompt_t_len + text_len:]], dim=1)
                x = torch.cat(
                    [shared_prompts_t, x[:, : d_prompt_t_len + shared_prompt_t_len + text_len, :],
                     shared_prompts_v, x[:, d_prompt_t_len + shared_prompt_t_len + text_len:, :]], dim=1)
            else:
                masks = masks
                x = x

            d_prompt_t_len += t_len
            d_prompt_v_len += v_len
            shared_prompt_t_len += shared_len_t
            shared_prompt_v_len += shared_len_v

            ## transformer block
            x, _attn = blk(x, mask=masks, prompts=None, prompt_type=self.prompt_type)

        x = self.transformer.norm(x)

        text_feats = x[:, d_prompt_t_len + shared_prompt_t_len:d_prompt_t_len + shared_prompt_t_len + text_len]
        image_feats = x[:, d_prompt_t_len + shared_prompt_t_len + text_len + d_prompt_v_len + shared_prompt_v_len :]

        if self.prompt_type == 'input':
            cls_feats = self.pooler(x[:, d_prompt_t_len + shared_prompt_t_len:d_prompt_t_len + shared_prompt_t_len + 1])
        else:
            cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
            "orthogonality_loss": orthogonality_loss,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))

        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))

        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)  # 执行forward
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
