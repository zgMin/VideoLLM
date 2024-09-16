from typing import List, Optional, Tuple, Union
import torch
import math
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import torch.nn.functional as F
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
# 1, 3136, 1024
class MMSelfAttention(nn.Module):
    def __init__(self, hidden_size, attention_probs_dropout_prob=0.1, num_attention_heads = 1, qkv_bias = False) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        
        # self.tokens = nn.Parameter(torch.randn(num_tokens, hidden_size))
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states,tokens, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        batch_size = hidden_states.size(0)
        tokens_expanded = tokens
        # print(hidden_states.shape,tokens_expanded.shape)
        new_hidden_states = torch.cat([hidden_states,tokens_expanded],dim=1)
        mixed_query_layer = self.query(tokens_expanded)

        key_layer = self.transpose_for_scores(self.key(new_hidden_states))
        value_layer = self.transpose_for_scores(self.value(new_hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
    
class MMAgentCrossAttention(nn.Module):
    def __init__(self, dim, dim2, num_tokens=5, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.,
                  num_frame = 16, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_frame = num_frame
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.agent = nn.Linear(dim2, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)


        self.tokens = nn.Parameter(torch.randn(num_tokens, dim2))
        # self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
        #                      padding=1, groups=dim)
        self.dwc = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)
        # pool_size = int(agent_num ** 0.5)
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x, x_t):
        """
        Args:
            x_t: text embeddings with shape of (B,N_t,D)
            x: input features with shape of (B, N, C)/(B, T*L, C)
        """
        b, n, c = x.shape
        
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c
        tokens_expanded = self.tokens.expand(b, -1, -1)
        x_t = torch.cat([x_t, tokens_expanded],dim=1)
        agent_tokens = self.agent(x_t)

        h = self.num_frame
        w = n // self.num_frame
        # agent_tokens = self.pool(q[:, 1:, :].reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, x_t.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)


        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) )
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v


        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) )
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        # print(x.shape,v.shape)
        v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
        # v_ = v[:, :, 1:, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # x[:, 1:, :] = x[:, 1:, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n - 1, c)
        x = x + self.dwc(v).permute(0, 2, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MMLayer(nn.Module):
    def __init__(self,mm_hidden_size, hidden_size, hidden_dropout_prob=0.1, num_frame=16, layer_norm_eps = 1e-12, is_cross = False):
        super().__init__()
        # compress: self-attention
        self.self_attention = MMSelfAttention(mm_hidden_size, qkv_bias=True)
        # compress: agent cross-attention
        self.is_cross = is_cross
        if self.is_cross:
            self.cross_attention = MMAgentCrossAttention(mm_hidden_size, hidden_size, num_frame=num_frame, qkv_bias = True)

        self.LayerNorm = nn.LayerNorm(mm_hidden_size, eps=layer_norm_eps)

        self.dropout = nn.Dropout(hidden_dropout_prob)
        #TODO: layernorm
        # compress: FFN
        self.ffn = nn.Sequential(
            nn.Linear(mm_hidden_size, mm_hidden_size),
            nn.GELU(),
            nn.Linear(mm_hidden_size, mm_hidden_size),
            nn.Dropout(hidden_dropout_prob),
        )
        
    def forward(self,x, x_t, tokens):
        batch_size = x.size(0)
        token_num = tokens.size(1)
        
        out = self.dropout(self.self_attention(x, tokens)[0])
        out = self.LayerNorm(out) + tokens
        if self.is_cross:
            o_out = torch.cat([x, out], dim=1)
            out = self.cross_attention(o_out, x_t)
            out = self.LayerNorm(self.dropout(out)) + o_out
            out = self.LayerNorm(self.ffn(out)) + out
            return out[:, :-token_num, :], out[:, -token_num:, :]
        out = self.LayerNorm(self.ffn(out)) + out
        return x, out


class MMProjector(nn.Module):
    # TODO: 目前是固定压缩长度，后续可以考虑多级压缩
    def __init__(self, mm_hidden_size, hidden_size, num_frame = 16, num_tokens = 356, num_hidden_layers = 6):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, mm_hidden_size))
        self.layer = nn.ModuleList([MMLayer(mm_hidden_size, hidden_size, num_frame= num_frame, is_cross= (i%2 == 1) ) for i in range(num_hidden_layers)])
        # linear
        self.proj = nn.Linear(mm_hidden_size, hidden_size)
    def forward(self, x, x_t):
        x = x.squeeze(1)
        
        b = x.size(0)
        tokens_expanded = self.tokens.unsqueeze(0).expand(b, -1, -1)
        hidden_states = x

        for i, layer_module in enumerate(self.layer):
            # print('start',i,hidden_states.shape,tokens_expanded.shape)
            hidden_states, tokens_expanded = layer_module(
                hidden_states,
                x_t,
                tokens_expanded
            )
            # print(i,hidden_states.shape,tokens_expanded.shape) 
        return self.proj(tokens_expanded)

    
class VisionConfig:
    def __init__(self):
        self.frame_size = 224
        self.patch_size = 14
        self.hidden_size = 1024
        self.use_vid_start_end = None
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None


class VideoChatGPTConfig(LlamaConfig):
    model_type = "VideoChatGPT"


class VideoChatGPTLlamaModel(LlamaModel):
    config_class = VideoChatGPTConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):  # TODO: Remove unused params
        super(VideoChatGPTLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_config = VisionConfig()

        if hasattr(config, "use_mm_proj"):
            # self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            self.mm_projector = MMProjector(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        vision_config = self.vision_config
        num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size

        if not hasattr(self, 'mm_projector'):
            # self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)
            self.mm_projector = MMProjector(self.config.mm_hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            video_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            question_ids: torch.LongTensor = None,
            video_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if question_ids is not None:
            question_embeds = self.embed_tokens(question_ids)

        if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None:
            # self.mm_projector=self.mm_projector.cuda()
            video_features = self.mm_projector(video_spatio_temporal_features,question_embeds)
            dummy_video_features = torch.zeros(video_features.shape[1], 1024, device=inputs_embeds.device,
                                               dtype=inputs_embeds.dtype)
            dummy_video_features = self.mm_projector.proj(dummy_video_features)
            
            # self.mm_projector=self.mm_projector.to('cpu')

            new_input_embeds = []
            cur_video_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0:
                    # Multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_video_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_video_idx += 1
                    continue
                if self.vision_config.use_vid_start_end:
                    if (cur_input_ids == self.vision_config.vid_start_token).sum() != (
                            cur_input_ids == self.vision_config.vid_end_token).sum():
                        raise ValueError("The number of video start tokens and video end tokens should be the same.")
                    video_start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                    for video_start_token_pos in video_start_tokens:
                        cur_video_features = video_features[cur_video_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_video_features.shape[0]
                        if cur_input_ids[video_start_token_pos + num_patches + 1] != self.vision_config.vid_end_token:
                            raise ValueError("The video end token should follow the video start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos].detach(),
                                                              cur_input_embeds[
                                                              video_start_token_pos:video_start_token_pos + 1],
                                                              cur_video_features, cur_input_embeds[
                                                                                  video_start_token_pos + num_patches
                                                                                  + 1:video_start_token_pos
                                                                                  + num_patches + 2],
                                                              cur_input_embeds[
                                                              video_start_token_pos + num_patches + 2:].detach()),
                                                             dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos + 1],
                                                              cur_video_features,
                                                              cur_input_embeds[video_start_token_pos
                                                                               + num_patches + 1:]), dim=0)
                        cur_video_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_video_features = video_features[cur_video_idx]
                    num_patches = cur_video_features.shape[0]
                    if (cur_input_ids == self.vision_config.vid_patch_token).sum() != num_patches:
                        raise ValueError(
                            "The number of video patch tokens should be the same as the number of video patches.")
                    masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches,
                                                       device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The video patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(),
                                                          cur_video_features,
                                                          cur_input_embeds[mask_index_start + num_patches:].detach()),
                                                         dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_video_features,
                                                          cur_input_embeds[mask_index_start + num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_video_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
        if video_mask is not None and not self.training:
            # print(attention_mask.shape,video_mask.shape)# TODO:video_mask貌似不会自动添加，可以用0填充长度
            padding_length = attention_mask.size(-1) - video_mask.size(-1)
            if padding_length>0:
                video_mask = F.pad(video_mask, (0, padding_length), "constant", 0)
            # print(attention_mask.shape,video_mask.shape)
            attention_mask = F.pad(attention_mask - video_mask, (0, 1), "constant", 1)
            attention_mask[:, -2] = 0
        return super(VideoChatGPTLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class VideoChatGPTLlamaForCausalLM(LlamaForCausalLM):
    config_class = VideoChatGPTConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VideoChatGPTLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            question_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            video_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not self.training and video_mask is not None and inputs_embeds is None: # repeat the last id
            last_id = input_ids[:,-1].unsqueeze(1)
            input_ids = torch.cat([input_ids,last_id],dim=-1)

        
        if not self.training and past_key_values is not None and video_mask is not None:
            # print(past_key_values[0][0].shape)
            new_past_key_values = tuple(
                (
                    t1[..., :-1, :],  # 去除第一个张量倒数第二个维度上的最后一个元素
                    t2[..., :-1, :]   # 去除第二个张量倒数第二个维度上的最后一个元素
                )
                for t1, t2 in past_key_values
            )
            past_key_values = new_past_key_values
            # print(past_key_values[0][0].shape)
            
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            question_ids=question_ids,
            video_mask=video_mask,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=video_spatio_temporal_features
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                'question_ids': kwargs.get("question_ids", None),
                "video_mask": kwargs.get("video_mask", None),
                "attention_mask": attention_mask,
                "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_vid_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_config
        vision_config.use_vid_start_end = mm_use_vid_start_end
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_vid_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]


AutoConfig.register("VideoChatGPT", VideoChatGPTConfig)
AutoModelForCausalLM.register(VideoChatGPTConfig, VideoChatGPTLlamaForCausalLM)