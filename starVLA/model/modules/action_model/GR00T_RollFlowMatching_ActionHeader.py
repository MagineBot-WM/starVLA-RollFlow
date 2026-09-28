"""Parallel Groot heads for padding-aware FM and pure rolling flow matching.

``RollFlowMatchingActionHead`` inherits the standard Groot module graph and
uses only scalar-time Groot forwards. ``PaddedGrootActionHead`` is a separate
padding-focused comparison baseline.
"""

from typing import Optional

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    MLP,
)
from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.rolling_flow_matching_head.rolling_flow import (
    RollingFlow,
    RollingFlowConfig,
)


class TokenTimeActionEncoder(nn.Module):
    """Groot action encoder accepting one flow time per action token.

    Parameter names/shapes match ``GR00T_ActionHeader.ActionEncoder`` so
    existing standard Groot action-head weights load without key conversion.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        source_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        if source_encoder is None:
            self.layer1 = nn.Linear(action_dim, hidden_size)
            self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
            self.layer3 = nn.Linear(hidden_size, hidden_size)
            self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)
        else:
            # Keep the original Groot layers and initialization intact when
            # the configured embedding width already matches the DiT.
            self.layer1 = source_encoder.layer1
            self.layer2 = source_encoder.layer2
            self.layer3 = source_encoder.layer3
            self.pos_encoding = source_encoder.pos_encoding

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = actions.shape
        if timesteps.ndim == 3 and timesteps.shape[-1] == 1:
            timesteps = timesteps.squeeze(-1)
        if timesteps.ndim == 1 and timesteps.shape[0] == batch:
            timesteps = timesteps[:, None].expand(-1, horizon)
        if timesteps.shape != (batch, horizon):
            raise ValueError(
                f"timesteps must have shape [B] or [B,H], got {tuple(timesteps.shape)}"
            )

        action_embedding = self.layer1(actions)
        time_embedding = self.pos_encoding(timesteps).to(dtype=action_embedding.dtype)
        hidden = swish(self.layer2(torch.cat((action_embedding, time_embedding), dim=-1)))
        return self.layer3(hidden)


class TokenTimeDiT(nn.Module):
    """DiT forward path with per-token AdaLN times and self-attention masks.

    Its modules are the original DiT modules, re-registered at the same names;
    it adds no parameters and preserves the standard checkpoint key layout.
    """

    def __init__(self, source: nn.Module):
        super().__init__()
        self.config = source.config
        self.inner_dim = source.inner_dim
        self.attention_head_dim = source.attention_head_dim
        self._interleave_self_attention = source._interleave_self_attention
        self.gradient_checkpointing = source.gradient_checkpointing
        self.timestep_encoder = source.timestep_encoder
        self.transformer_blocks = source.transformer_blocks
        self.norm_out = source.norm_out
        self.proj_out_1 = source.proj_out_1
        self.proj_out_2 = source.proj_out_2

    def _encode_time(self, timestep: torch.Tensor) -> torch.Tensor:
        shape = timestep.shape
        encoder = self.timestep_encoder
        projected = encoder.time_proj(timestep.reshape(-1)).to(
            dtype=next(encoder.parameters()).dtype
        )
        return encoder.timestep_embedder(projected).reshape(*shape, -1)

    @staticmethod
    def _ada_norm(norm: nn.Module, hidden: torch.Tensor, time_embedding: torch.Tensor):
        modulation = norm.linear(norm.silu(time_embedding))
        scale, shift = modulation.chunk(2, dim=-1)
        if time_embedding.ndim == 2:
            scale, shift = scale[:, None], shift[:, None]
        return norm.norm(hidden) * (1 + scale) + shift

    @classmethod
    def _run_block(
        cls,
        block: nn.Module,
        hidden: torch.Tensor,
        time_embedding: torch.Tensor,
        *,
        self_attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    ) -> torch.Tensor:
        if block.norm_type == "ada_norm":
            normalized = cls._ada_norm(block.norm1, hidden, time_embedding)
        else:
            normalized = block.norm1(hidden)
        if block.pos_embed is not None:
            normalized = block.pos_embed(normalized)

        is_self_attention = encoder_hidden_states is None
        attended = block.attn1(
            normalized,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=self_attention_mask if is_self_attention else encoder_attention_mask,
        )
        if block.final_dropout:
            attended = block.final_dropout(attended)
        hidden = attended + hidden
        if hidden.ndim == 4:
            hidden = hidden.squeeze(1)

        hidden = block.ff(block.norm3(hidden)) + hidden
        if hidden.ndim == 4:
            hidden = hidden.squeeze(1)
        return hidden

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states,
        timestep: torch.Tensor,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None,
        self_attention_mask=None,
        return_pre_output: bool = False,
    ):
        time_embedding = self._encode_time(timestep)
        hidden_states = hidden_states.contiguous()
        layerwise_context = isinstance(encoder_hidden_states, (list, tuple))
        if layerwise_context:
            encoder_hidden_states = [value.contiguous() for value in encoder_hidden_states]
        else:
            encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]
        for index, block in enumerate(self.transformer_blocks):
            self_attention = index % 2 == 1 and self._interleave_self_attention
            if self_attention:
                block_context = None
                block_context_mask = None
            else:
                block_context = encoder_hidden_states[index] if layerwise_context else encoder_hidden_states
                block_context_mask = encoder_attention_mask
            hidden_states = self._run_block(
                block,
                hidden_states,
                time_embedding,
                self_attention_mask=self_attention_mask,
                encoder_hidden_states=block_context,
                encoder_attention_mask=block_context_mask,
            )
            all_hidden_states.append(hidden_states)

        if return_pre_output:
            return (hidden_states, all_hidden_states) if return_all_hidden_states else hidden_states

        conditioning = torch.nn.functional.silu(time_embedding)
        shift, scale = self.proj_out_1(conditioning).chunk(2, dim=-1)
        if conditioning.ndim == 2:
            shift, scale = shift[:, None], scale[:, None]
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
        prediction = self.proj_out_2(hidden_states)
        return (prediction, all_hidden_states) if return_all_hidden_states else prediction


def _cfg(config, name: str, default):
    value = getattr(config, name, default)
    return default if value is None else value


def _repeat_time_group_batch(value, batch: int, groups: int):
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == batch:
            return value[:, None].expand(batch, groups, *value.shape[1:]).reshape(
                batch * groups, *value.shape[1:]
            )
        return value
    if isinstance(value, list):
        return [_repeat_time_group_batch(item, batch, groups) for item in value]
    if isinstance(value, tuple):
        return tuple(_repeat_time_group_batch(item, batch, groups) for item in value)
    return value


class PaddedGrootActionHead(FlowmatchingActionHead):
    """Standard Groot FM objective with padding-aware loss and DiT masking.

    This parallel baseline leaves the official framework/head untouched and
    lets three-way comparisons use the same padding semantics.
    """
    def __init__(self, full_config):
        super().__init__(full_config)
        config = full_config.framework.action_model
        source_dit = self.model
        if source_dit.inner_dim != self.input_embedding_dim:
            self.input_embedding_dim = source_dit.inner_dim
            self.state_encoder = (
                MLP(config.state_dim, self.hidden_size, self.input_embedding_dim)
                if config.state_dim else None
            )
            self.action_encoder = TokenTimeActionEncoder(
                self.action_dim, self.input_embedding_dim
            )
            self.action_decoder = MLP(
                source_dit.config.output_dim, self.hidden_size, self.action_dim
            )
            self.future_tokens = nn.Embedding(
                config.num_target_vision_tokens, self.input_embedding_dim
            )
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
            if config.add_pos_embed:
                self.position_embedding = nn.Embedding(
                    config.max_seq_len, self.input_embedding_dim
                )
                nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.action_encoder = TokenTimeActionEncoder(
                self.action_dim,
                self.input_embedding_dim,
                source_encoder=self.action_encoder,
            )
        self.model = TokenTimeDiT(source_dit)
        self.last_loss_stats = None

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask=None,
        action_padding_mask: Optional[torch.Tensor] = None,
        training_step: int = 0,
    ) -> torch.Tensor:
        del training_step
        batch, horizon, _ = actions.shape
        if horizon != self.action_horizon:
            raise ValueError(f"expected action horizon {self.action_horizon}, got {horizon}")
        actions = actions.to(device=self.device, dtype=torch.float32)
        if action_padding_mask is not None:
            action_padding_mask = action_padding_mask.to(device=actions.device, dtype=torch.bool)
            if action_padding_mask.shape != (batch, horizon):
                raise ValueError(
                    f"action_padding_mask must have shape {(batch, horizon)}, "
                    f"got {tuple(action_padding_mask.shape)}"
                )
            actions = actions.masked_fill(action_padding_mask.unsqueeze(-1), 0)

        noise = torch.randn_like(actions)
        if action_padding_mask is not None:
            noise = noise.masked_fill(action_padding_mask.unsqueeze(-1), 0)
        time = self.sample_time(batch, actions.device, actions.dtype)
        noisy_actions = (
            (1 - time[:, None, None]) * noise + time[:, None, None] * actions
        ).to(dtype=self.dtype)
        target_velocity = actions - noise
        discrete_time = (time * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_actions, discrete_time)

        state_features = (
            self.state_encoder(state.to(device=self.device, dtype=self.dtype))
            if state is not None else None
        )
        if self.config.add_pos_embed:
            positions = torch.arange(horizon, device=actions.device)
            action_features = action_features + self.position_embedding(positions)[None]
        future_tokens = self.future_tokens.weight[None].expand(batch, -1, -1)
        prefix = [future_tokens]
        if state_features is not None:
            prefix.insert(0, state_features)
        prefix_length = sum(value.shape[1] for value in prefix)
        hidden_states = torch.cat((*prefix, action_features), dim=1)

        self_attention_mask = None
        if action_padding_mask is not None:
            valid_keys = torch.cat(
                (
                    torch.ones(batch, prefix_length, device=actions.device, dtype=torch.bool),
                    ~action_padding_mask,
                ),
                dim=1,
            )
            self_attention_mask = torch.zeros(
                batch, 1, prefix_length + horizon,
                device=actions.device, dtype=hidden_states.dtype,
            )
            self_attention_mask.masked_fill_(
                ~valid_keys[:, None, :], torch.finfo(hidden_states.dtype).min
            )

        model_output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=self._cast_context(vl_embs),
            encoder_attention_mask=encoder_attention_mask,
            self_attention_mask=self_attention_mask,
            timestep=discrete_time,
        )
        prediction = self.action_decoder(model_output[:, -horizon:])
        squared_error = (prediction.float() - target_velocity.float()).square()
        if action_padding_mask is None:
            loss = squared_error.mean()
        else:
            valid = (~action_padding_mask).unsqueeze(-1)
            loss = torch.where(valid, squared_error, torch.zeros_like(squared_error)).sum()
            loss = loss / valid.sum().clamp_min(1) / actions.shape[-1]
        self.last_loss_stats = {"fm_loss": loss.detach()}
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask=None,
        refinement_steps: Optional[int] = None,
        velocity_mode: Optional[str] = None,
    ) -> torch.Tensor:
        del refinement_steps, velocity_mode
        return FlowmatchingActionHead.predict_action(
            self,
            self._cast_context(vl_embs),
            None if state is None else state.to(device=self.device, dtype=self.dtype),
            encoder_attention_mask=encoder_attention_mask,
        )

    def _cast_context(self, context):
        if isinstance(context, (list, tuple)):
            return type(context)(value.to(device=self.device, dtype=self.dtype) for value in context)
        return context.to(device=self.device, dtype=self.dtype)

    def reset(self) -> None:
        return None

    reset_cache = reset


class RollFlowMatchingActionHead(FlowmatchingActionHead):
    """Original Groot parameterization with staircase-time FM and rolling output."""

    def __init__(self, full_config):
        super().__init__(full_config)
        config = full_config.framework.action_model
        self.execution_horizon = int(_cfg(config, "execution_horizon", 1))
        self.rolling_flow = RollingFlow(
            RollingFlowConfig(
                horizon=self.action_horizon,
                action_dim=int(self.action_dim),
                chunk_size=self.execution_horizon,
                inference_steps=_cfg(
                    config, "inference_steps", _cfg(config, "num_inference_timesteps", None)
                ),
                p_k1=float(_cfg(config, "p_k1", 0.7)),
                use_ot=bool(_cfg(config, "use_ot", True)),
                clip_velocity=float(_cfg(config, "clip_velocity", 0.0)),
                iterative_cold_start=bool(_cfg(config, "iterative_cold_start", True)),
                reset_cache_each_step=bool(_cfg(config, "reset_cache_each_step", False)),
            )
        )
        self.last_loss_stats = None

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask=None,
        action_padding_mask: Optional[torch.Tensor] = None,
        training_step: int = 0,
    ) -> torch.Tensor:
        del training_step
        if action_padding_mask is not None:
            action_padding_mask = action_padding_mask.to(
                device=actions.device, dtype=torch.bool
            )
        state_features = (
            self.state_encoder(state.to(device=self.device, dtype=self.dtype))
            if state is not None
            else None
        )
        loss, self.last_loss_stats = self.rolling_flow.training_loss(
            self._predict_grouped,
            actions,
            sample_time=lambda batch, device, dtype: self.sample_time(batch, device, dtype),
            context=self._cast_context(vl_embs),
            pad=action_padding_mask,
            predict_kwargs={
                "state_features": state_features,
                "encoder_attention_mask": encoder_attention_mask,
                "action_padding_mask": action_padding_mask,
            },
        )
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask=None,
        refinement_steps: Optional[int] = None,
        velocity_mode: Optional[str] = None,
    ) -> torch.Tensor:
        if velocity_mode not in (None, "instant"):
            raise ValueError("this action head supports instantaneous-velocity inference only")
        state_features = (
            self.state_encoder(state.to(device=self.device, dtype=self.dtype))
            if state is not None
            else None
        )
        return self.rolling_flow.step(
            self._predict_grouped,
            batch=vl_embs.shape[0],
            context=self._cast_context(vl_embs),
            device=vl_embs.device,
            dtype=self.dtype,
            refinement_steps=refinement_steps,
            predict_kwargs={
                "state_features": state_features,
                "encoder_attention_mask": encoder_attention_mask,
            },
        )

    def _predict_grouped(
        self,
        actions: torch.Tensor,
        action_times: torch.Tensor,
        context,
        *,
        state_features: Optional[torch.Tensor] = None,
        encoder_attention_mask=None,
        action_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the unmodified Groot scalar-time path to each time plateau."""
        batch, horizon, _ = actions.shape
        if action_times.ndim == 3 and action_times.shape[-1] == 1:
            action_times = action_times.squeeze(-1)
        if action_times.ndim == 1 and action_times.shape[0] == batch:
            action_times = action_times[:, None].expand(-1, horizon)
        if action_times.shape != (batch, horizon):
            raise ValueError(
                f"action_times must be [B], [B,H] or [B,H,1], got {tuple(action_times.shape)}"
            )
        actions = actions.to(device=self.device, dtype=self.dtype)
        if action_padding_mask is not None:
            action_padding_mask = action_padding_mask.to(
                device=actions.device, dtype=torch.bool
            )
            if action_padding_mask.shape != (batch, horizon):
                raise ValueError(
                    f"action_padding_mask must be {(batch, horizon)}, "
                    f"got {tuple(action_padding_mask.shape)}"
                )
            actions = actions.masked_fill(action_padding_mask.unsqueeze(-1), 0)

        chunk_size = self.execution_horizon
        if horizon % chunk_size:
            raise ValueError("action sequence length must be divisible by execution_horizon")
        chunk_times = action_times.reshape(batch, horizon // chunk_size, chunk_size)[:, :, 0]
        runs = []
        run_start = 0
        for chunk in range(1, chunk_times.shape[1]):
            if not torch.equal(chunk_times[:, chunk], chunk_times[:, chunk - 1]):
                runs.append((run_start * chunk_size, chunk * chunk_size))
                run_start = chunk
        runs.append((run_start * chunk_size, horizon))

        time_groups = torch.stack([action_times[:, start] for start, _ in runs], dim=1)
        num_groups = len(runs)
        grouped_actions = actions[:, None].expand(
            batch, num_groups, horizon, self.action_dim
        ).reshape(batch * num_groups, horizon, self.action_dim)
        grouped_context = _repeat_time_group_batch(
            self._cast_context(context), batch, num_groups
        )
        grouped_state = _repeat_time_group_batch(state_features, batch, num_groups)
        grouped_attention_mask = _repeat_time_group_batch(
            encoder_attention_mask, batch, num_groups
        )
        grouped_prediction = self._predict_scalar_time(
            grouped_actions,
            time_groups.reshape(-1).to(device=actions.device),
            grouped_context,
            state_features=grouped_state,
            encoder_attention_mask=grouped_attention_mask,
            output_slice=(0, horizon),
        ).reshape(batch, num_groups, horizon, self.action_dim)

        prediction = torch.empty(
            batch, horizon, self.action_dim, device=actions.device, dtype=torch.float32
        )
        for group, (start, end) in enumerate(runs):
            prediction[:, start:end] = grouped_prediction[:, group, start:end]
        return prediction

    def _cast_context(self, context):
        if isinstance(context, (list, tuple)):
            return type(context)(
                value.to(device=self.device, dtype=self.dtype) for value in context
            )
        return context.to(device=self.device, dtype=self.dtype)

    def _predict_scalar_time(
        self,
        actions: torch.Tensor,
        time: torch.Tensor,
        context,
        *,
        state_features: Optional[torch.Tensor],
        encoder_attention_mask,
        output_slice: tuple[int, int],
    ) -> torch.Tensor:
        """Same encoder, scalar timestep, DiT, and decoder as standard Groot."""
        batch, horizon, _ = actions.shape
        discrete_time = (time * self.num_timestep_buckets).long()
        action_features = self.action_encoder(actions, discrete_time)
        if self.config.add_pos_embed:
            if horizon > self.position_embedding.num_embeddings:
                raise ValueError(f"action horizon {horizon} exceeds max_seq_len")
            positions = torch.arange(horizon, device=actions.device)
            action_features = action_features + self.position_embedding(positions)[None]

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
        hidden_states = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )
        model_output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
            timestep=discrete_time,
        )
        prediction = self.action_decoder(model_output[:, -horizon:]).float()
        return prediction[:, output_slice[0] : output_slice[1]]

    def reset(self) -> None:
        self.rolling_flow.reset()

    reset_cache = reset

    @property
    def cache_info(self):
        return self.rolling_flow.cache_info


def get_action_model(config=None):
    return RollFlowMatchingActionHead(full_config=config)
