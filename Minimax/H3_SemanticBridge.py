"""H3 语义桥（MiniMax H3）：选权重并把语义桥直接应用到条件张量。

移植自 ComfyUI-Yuan 独立包 semantic_bridge.py；上游为 T8mars/comfyui-minimax-h3-audio-T8
（GPL-3.0-or-later），数值实现与其一致。权重随包放在本节点的 Minimax/models/ 下，
缺失时由节点按 SHA256 校验下载到同一目录。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
import urllib.request
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import folder_paths
import torch
from torch import nn
from torch.nn import functional as F

# 权重契约常量
SCHEMA = "t8_semantic_bridge_v1"
RECEIPT_KEY = "t8_semantic_bridge"
SHAPES = {
    "fc1.weight": (512, 5120), "fc1.bias": (512,),
    "fc2.weight": (512, 512), "fc2.bias": (512,),
    "fc3.weight": (5120, 512), "fc3.bias": (5120,),
}
MAX_FILE_BYTES = 256 * 1024 * 1024
_CACHE = OrderedDict()  # 只缓存 CPU FP32 权重，最多保留两份。
_CACHE_LOCK = threading.RLock()


@dataclass(frozen=True)
class _TransWeights:
    state: dict
    metadata: dict


@dataclass(frozen=True)
class _TrainerMLPWeights:
    state: dict
    spec: object


# safetensors 读字节与元数据
def _weight_metadata(data: bytes) -> dict:
    header_size = int.from_bytes(data[:8], "little")
    if header_size <= 0 or header_size > min(len(data) - 8, 1 << 20):
        raise ValueError("Invalid WushuBridge safetensors metadata header")
    header = json.loads(data[8:8 + header_size])
    if not isinstance(header, dict):
        raise ValueError("Invalid Bridge safetensors header")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid WushuBridge safetensors metadata")
    return metadata


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _read_bridge_bytes(path, expected_sha=None):
    path = Path(path)
    if path.suffix.lower() != ".safetensors" or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Expected a small Semantic Bridge .safetensors file")
    # 解码的就是被哈希的那份字节，避免路径替换后身份混淆。
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Bridge file exceeds supported size")
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise ValueError("Bridge model content changed; re-execute the configuration node")
    return data, sha


# 可变宽度 MLP 权重校验
@dataclass(frozen=True)
class TrainerMLPSpec:
    hidden: int
    residual_skip: bool
    residual_scale: float
    application_contract: dict | None


def validate_trainer_mlp(state: dict, metadata: dict) -> TrainerMLPSpec:
    if metadata.get("arch") != "mlp" or int(metadata.get("dim", "0")) != 5120:
        raise ValueError("Expected a 5120-dimensional WushuBridge MLP model")
    hidden = int(metadata.get("hidden", "0"))
    if hidden < 1 or hidden > 4096:
        raise ValueError("Unsupported WushuBridge MLP width")
    shapes = {
        "net.fc1.weight": (hidden, 5120), "net.fc1.bias": (hidden,),
        "net.fc2.weight": (hidden, hidden), "net.fc2.bias": (hidden,),
        "net.fc3.weight": (5120, hidden), "net.fc3.bias": (5120,),
    }
    if set(state) != set(shapes):
        raise ValueError("WushuBridge MLP tensor keys do not match the declared architecture")
    for key, shape in shapes.items():
        value = state[key]
        if (tuple(value.shape) != shape
                or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or not torch.isfinite(value).all().item()):
            raise ValueError(f"Invalid WushuBridge MLP tensor: {key}")
    skip = metadata.get("residual_skip", "False").lower()
    scale = float(metadata.get("residual_scale", "0.1"))
    if skip not in ("true", "false") or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Invalid WushuBridge MLP residual metadata")
    extra = json.loads(metadata.get("extra_json", "{}"))
    if not isinstance(extra, dict):
        raise ValueError("Invalid WushuBridge MLP extra_json metadata")
    contract = extra.get("application_contract")
    if contract is not None and not isinstance(contract, dict):
        raise ValueError("Invalid WushuBridge MLP application contract")
    return TrainerMLPSpec(hidden, skip == "true", scale, contract)


# Transformer 桥：结构、位置编码、调用
@dataclass(frozen=True)
class TransSpec:
    hidden: int
    layers: int
    heads: int
    max_tokens: int
    residual_skip: bool
    residual_scale: float
    application_contract: dict | None


def spec_from_metadata(metadata: dict) -> TransSpec:
    if metadata.get("arch") != "trans" or int(metadata.get("dim", "0")) != 5120:
        raise ValueError("Expected a 5120-dimensional WushuBridge trans model")
    hidden = int(metadata.get("hidden", "0"))
    layers = int(metadata.get("layers", "0"))
    heads = int(metadata.get("heads", "0"))
    max_tokens = int(metadata.get("max_tokens", "4096"))
    scale = float(metadata.get("residual_scale", "0.1"))
    if (hidden < 64 or hidden > 1024 or layers < 1 or layers > 4 or heads < 1
            or heads > 16 or hidden % heads or max_tokens < 1 or max_tokens > 16384
            or not math.isfinite(scale) or not 0 < scale <= 1):
        raise ValueError("Unsupported WushuBridge trans architecture")
    skip = metadata.get("residual_skip", "False").lower()
    if skip not in ("true", "false"):
        raise ValueError("Invalid WushuBridge residual_skip metadata")
    extra = json.loads(metadata.get("extra_json", "{}"))
    if not isinstance(extra, dict):
        raise ValueError("Invalid WushuBridge extra_json metadata")
    contract = extra.get("application_contract")
    if contract is not None and not isinstance(contract, dict):
        raise ValueError("Invalid WushuBridge application contract")
    return TransSpec(hidden, layers, heads, max_tokens, skip == "true", scale, contract)


def contract_settings(contract: dict) -> dict:
    # 只认训练端固定的推理公式，不猜默认值。
    alpha = contract.get("alpha_min")
    if (type(contract.get("schema")) is not int or contract["schema"] != 1
            or type(alpha) not in (int, float) or not math.isfinite(alpha)
            or not 0 < alpha <= 1 or type(contract.get("alpha_max")) not in (int, float)
            or contract["alpha_max"] != alpha
            or contract.get("magnitude_match") != "per_token"
            or contract.get("token_span") != "all"
            or type(contract.get("tail_ratio")) not in (int, float) or contract["tail_ratio"] != 1
            or type(contract.get("chunk_tokens")) is not int or contract["chunk_tokens"] != 0
            or any(contract.get(key) is not False for key in ("auto_alpha", "guard", "allow_dim_mismatch"))):
        raise ValueError("Unsupported or incomplete Bridge fixed application contract")
    return {"alpha": float(alpha), "magnitude_match": "per_token",
            "token_scope": "all_tokens", "chunk_tokens": 0}


def validate_contract(spec: TransSpec | dict | None, *, alpha: float, magnitude_match: str,
                      token_scope: str, chunk_tokens: int) -> None:
    contract = spec.application_contract if isinstance(spec, TransSpec) else spec
    if contract is None:
        return
    contract_settings(contract)
    expected = {
        "schema": 1, "alpha_min": alpha, "alpha_max": alpha,
        "magnitude_match": magnitude_match,
        "token_span": "all" if token_scope == "all_tokens" else token_scope,
        "tail_ratio": 1.0, "chunk_tokens": chunk_tokens,
        "auto_alpha": False, "guard": False, "allow_dim_mismatch": False,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("WushuBridge application contract mismatch: required "
                         f"alpha={contract.get('alpha_min')}, "
                         f"magnitude_match={contract.get('magnitude_match')}, "
                         f"token_scope=all_tokens, chunk_tokens={contract.get('chunk_tokens')}. "
                         "Use H3 Semantic Bridge / 自动模型参数 (T8 EXP), or set these values explicitly.")


def _positions(length: int, hidden: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / hidden))
    values = torch.zeros(length, hidden, dtype=torch.float32)
    values[:, 0::2] = torch.sin(positions * divisor)
    values[:, 1::2] = torch.cos(positions * divisor)[:, :values[:, 1::2].shape[1]]
    return values.unsqueeze(0)


class TransBridge(nn.Module):
    def __init__(self, spec: TransSpec):
        super().__init__()
        self.spec = spec
        self.in_proj = nn.Linear(5120, spec.hidden)
        self.register_buffer("_pe", _positions(spec.max_tokens, spec.hidden), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=spec.hidden, nhead=spec.heads, dim_feedforward=spec.hidden * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=spec.layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(spec.hidden)
        self.out_proj = nn.Linear(spec.hidden, 5120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.in_proj(x)
        pe = self._pe
        if encoded.shape[1] > pe.shape[1]:
            pe = F.interpolate(pe.transpose(1, 2), size=encoded.shape[1],
                               mode="linear", align_corners=False).transpose(1, 2)
        encoded = encoded + pe[:, :encoded.shape[1]].to(device=x.device, dtype=x.dtype)
        correction = self.out_proj(self.norm(self.encoder(encoded)))
        return x + self.spec.residual_scale * correction if self.spec.residual_skip else correction


def build_trans_bridge(state: dict, metadata: dict) -> TransBridge:
    spec = spec_from_metadata(metadata)
    # 初始化参数随后被 state 覆盖，且不能消费调用方的 CPU 噪声随机数。
    with torch.random.fork_rng(devices=[]):
        model = TransBridge(spec)
    expected = {"net." + key for key in model.state_dict()}
    if set(state) != expected:
        raise ValueError("WushuBridge trans tensor keys do not match the declared architecture")
    for key, value in state.items():
        if (tuple(value.shape) != tuple(model.state_dict()[key[4:]].shape)
                or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or not torch.isfinite(value).all().item()):
            raise ValueError(f"Invalid WushuBridge trans tensor: {key}")
    model.load_state_dict({key[4:]: value for key, value in state.items()}, strict=True)
    return model.eval()


# 桥参数与权重读取
@dataclass(frozen=True)
class BridgeConfig:
    path: str
    sha256: str
    alpha: float = 0.10
    magnitude_match: str = "per_token"
    token_scope: str = "all_tokens"
    device: str = "auto"
    chunk_tokens: int = 256
    enabled: bool = True
    compute_profile: str = "fp32"

    def __post_init__(self):
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("Bridge alpha must be finite and between 0 and 1")
        if self.magnitude_match not in ("per_token", "global", "none"):
            raise ValueError("Unknown Bridge magnitude_match")
        if self.token_scope not in ("all_tokens", "text_only_preserve_reference"):
            raise ValueError("Unknown Bridge token_scope")
        if self.device not in ("auto", "cpu", "cuda") or self.compute_profile != "fp32":
            raise ValueError("Unknown Bridge device/compute_profile")
        if type(self.chunk_tokens) is not int or not 0 <= self.chunk_tokens <= 65536:
            raise ValueError("Bridge chunk_tokens must be zero (whole sequence) or positive")

    @property
    def active(self):
        return self.enabled and self.alpha != 0

    def identity(self):
        if not self.active:
            return None
        values = asdict(self)
        values.pop("path")  # 回执/缓存不带本机绝对路径。
        return {"schema": SCHEMA, **values}


def bridge_identity(config):
    if config is None:
        return None
    if not isinstance(config, BridgeConfig):
        raise TypeError("Expected T8 Semantic Bridge configuration")
    return config.identity()


def validate_weights(weights):
    if set(weights) != set(SHAPES):
        raise ValueError("Bridge requires exactly six fc1/fc2/fc3 weight/bias tensors")
    for key, shape in SHAPES.items():
        value = weights[key]
        if tuple(value.shape) != shape or value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Invalid Bridge tensor {key}: {tuple(value.shape)} {value.dtype}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Non-finite Bridge tensor: {key}")


def read_weights(path, expected_sha=None):
    from safetensors.torch import load

    data, sha = _read_bridge_bytes(path, expected_sha)
    return load(data), sha, _weight_metadata(data)


def _weights(config):
    # 每次都重新哈希，同大小同 mtime 的替换也要挡住；不在 CUDA 上留缓存。
    if file_sha(config.path) != config.sha256:
        raise ValueError("Bridge model content changed; re-execute the configuration node")
    with _CACHE_LOCK:
        cached = _CACHE.get(config.sha256)
        if cached is None:
            state, _, metadata = read_weights(config.path, config.sha256)
            if "net.in_proj.weight" in state:
                cached = _TransWeights({name: tensor.float() for name, tensor in state.items()}, metadata)
            elif "net.fc1.weight" in state:
                cached = _TrainerMLPWeights(
                    {name[4:]: tensor.float() for name, tensor in state.items()},
                    validate_trainer_mlp(state, metadata))
            else:
                validate_weights(state)
                cached = {name: tensor.float() for name, tensor in state.items()}
            _CACHE[config.sha256] = cached
            while len(_CACHE) > 2:
                _CACHE.popitem(last=False)
        _CACHE.move_to_end(config.sha256)
        return cached


def _tensor_sha(tensor):
    data = tensor.detach().contiguous().cpu()
    sha = hashlib.sha256(str((tuple(data.shape), data.dtype)).encode())
    sha.update(data.view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def _project(h, weights):
    with torch.autocast(device_type=h.device.type, enabled=False):
        normalized = h / (h.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        first = F.silu(F.linear(normalized, weights["fc1.weight"], weights["fc1.bias"]))
        second = F.silu(F.linear(first, weights["fc2.weight"], weights["fc2.bias"]))
        return F.linear(second, weights["fc3.weight"], weights["fc3.bias"])


def _predict_mlp(h, weights, trainer_spec=None):
    projected = _project(h, weights)
    if trainer_spec is not None and trainer_spec.residual_skip:
        normalized = h / (h.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        return normalized + trainer_spec.residual_scale * projected
    return projected


def _check_cancel():
    # 只在实际执行时导入，脱离 ComfyUI 也能导入本模块。
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def _receipt(identity, encoding_source, native, output, metadata, token_scope):
    receipt = {
        **identity, "encoding_source": encoding_source,
        "input_sha256": _tensor_sha(native), "output_sha256": _tensor_sha(output),
        "shape": list(native.shape), "dtype": str(native.dtype),
        "applied_tokens_per_batch": (int((metadata["minimax_token_tags"] == 1).sum().item())
                                     if token_scope != "all_tokens" else native.shape[1]),
        "reference_payload_present": bool(metadata.get("minimax_refs")),
    }
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def _apply_trans_bridge(conditioning, config, model, identity, cancel, encoding_source):
    outputs, receipts = [], []
    for native, metadata in conditioning:
        device = native.device if config.device == "auto" else torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Bridge CUDA explicitly selected but CUDA is unavailable")
        model = model.to(device=device, dtype=torch.float32)
        h = native.to(device=device, dtype=torch.float32)
        pred = torch.empty_like(h)
        stride = config.chunk_tokens or h.shape[1]
        with torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, h.shape[1], stride):
                cancel()
                segment = h[:, start:start + stride]
                normalized = segment / (segment.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()
                pred[:, start:start + stride] = model(normalized)
        if config.magnitude_match == "per_token":
            pred = pred * ((h.square().mean(-1, keepdim=True) + 1e-8).sqrt() /
                           (pred.square().mean(-1, keepdim=True) + 1e-8).sqrt())
        elif config.magnitude_match == "global":
            pred = pred * ((h.square().mean() + 1e-8).sqrt() /
                           (pred.square().mean() + 1e-8).sqrt())
        result = h + config.alpha * (pred - h)
        if config.token_scope == "text_only_preserve_reference":
            tags = metadata["minimax_token_tags"].to(device)
            result = torch.where((tags == 1)[None, :, None], result, h)
        output = result.to(device=native.device, dtype=native.dtype)
        if not torch.isfinite(output).all().item():
            raise ValueError("Non-finite Bridge output")
        receipt = _receipt(identity, encoding_source, native, output, metadata, config.token_scope)
        outputs.append([output, {**metadata, RECEIPT_KEY: receipt}])
        receipts.append(receipt)
    cancel()
    return outputs, {"enabled": True, "applied": True, "identity": identity, "items": receipts,
                     "quality": "experimental_not_human_qualified",
                     "warning": "Reference audio/singing can degrade; video quality requires actual A/B review."}


@torch.inference_mode()
def apply_bridge(conditioning, config, *, encoding_source="external_unknown", cancel=None):
    identity = bridge_identity(config)
    if identity is None:
        return conditioning, {"enabled": False, "applied": False}
    cancel = cancel or _check_cancel
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError("Bridge requires non-empty CONDITIONING")
    # 先校验全部条目，再读权重、出结果。
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
            raise ValueError("Invalid CONDITIONING item")
        native, metadata = item
        if not isinstance(native, torch.Tensor) or native.ndim != 3 or native.shape[-1] != 5120:
            raise ValueError("Bridge expects raw H3 conditioning [B,T,5120], not projected embeds")
        if native.numel() == 0 or not native.is_floating_point() or not torch.isfinite(native).all().item():
            raise ValueError("Bridge input must contain non-empty finite floating point embeddings")
        if RECEIPT_KEY in metadata or metadata.get("sensenova_h3_distilled"):
            raise ValueError("Semantic Bridge already applied; use a fresh native encoding, do not stack bridges")
        if "minimax_prompt_relay_binding" in metadata or "t8_prompt_relay_binding_hash" in metadata:
            raise ValueError("Apply Bridge through the Prompt Relay optional input, before Relay binding")
        if config.token_scope == "text_only_preserve_reference":
            tags = metadata.get("minimax_token_tags")
            if not isinstance(tags, torch.Tensor) or tags.ndim != 1 or tags.numel() != native.shape[1]:
                raise ValueError("Text-only Bridge requires exact native minimax_token_tags")
            if not torch.all((tags == 0) | (tags == 1)).item():
                raise ValueError("Unknown native H3 token tags")
    cancel()
    state = _weights(config)
    if isinstance(state, _TransWeights):
        model = build_trans_bridge(state.state, state.metadata)
        validate_contract(model.spec, alpha=config.alpha,
                          magnitude_match=config.magnitude_match,
                          token_scope=config.token_scope, chunk_tokens=config.chunk_tokens)
        return _apply_trans_bridge(conditioning, config, model, identity, cancel, encoding_source)
    trainer_spec = None
    if isinstance(state, _TrainerMLPWeights):
        trainer_spec = state.spec
        validate_contract(trainer_spec.application_contract, alpha=config.alpha,
                          magnitude_match=config.magnitude_match,
                          token_scope=config.token_scope, chunk_tokens=config.chunk_tokens)
        state = state.state
    outputs, receipts = [], []
    for native, metadata in conditioning:
        device = native.device if config.device == "auto" else torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Bridge CUDA explicitly selected but CUDA is unavailable")
        # 权重只在本条目的作用域内上设备，用完即弃。
        weights = {name: tensor.to(device=device) for name, tensor in state.items()}
        rows = native.reshape(-1, 5120)
        output = torch.empty_like(rows)
        global_scale = None
        if config.magnitude_match == "global":
            # 整条（所有 batch/token）统计，不是逐块匹配。
            source_sum = torch.zeros((), dtype=torch.float64, device=device)
            target_sum = torch.zeros_like(source_sum)
            chunk_tokens = config.chunk_tokens or rows.shape[0]
            for start in range(0, rows.shape[0], chunk_tokens):
                cancel()
                h = rows[start:start + chunk_tokens].to(device=device, dtype=torch.float32)
                projected = _predict_mlp(h, weights, trainer_spec)
                source_sum += projected.double().square().sum()
                target_sum += h.double().square().sum()
            global_scale = ((target_sum / native.numel() + 1e-8) /
                            (source_sum / native.numel() + 1e-8)).sqrt().float()
        chunk_tokens = config.chunk_tokens or rows.shape[0]
        for start in range(0, rows.shape[0], chunk_tokens):
            cancel()
            h = rows[start:start + chunk_tokens].to(device=device, dtype=torch.float32)
            projected = _predict_mlp(h, weights, trainer_spec)
            if config.magnitude_match == "per_token":
                projected = projected * ((h.square().mean(-1, keepdim=True) + 1e-8).sqrt() /
                                         (projected.square().mean(-1, keepdim=True) + 1e-8).sqrt())
            elif global_scale is not None:
                projected = projected * global_scale
            result = h + config.alpha * (projected - h)
            if not torch.isfinite(result).all().item():
                raise ValueError("Non-finite Bridge output")
            output[start:start + chunk_tokens] = result.to(device=native.device, dtype=native.dtype)
        output = output.reshape_as(native)
        if config.token_scope == "text_only_preserve_reference":
            tags = metadata["minimax_token_tags"].to(native.device)
            output = torch.where((tags == 1)[None, :, None], output, native)
        if not torch.isfinite(output).all().item():
            raise ValueError("Bridge output overflows the original conditioning dtype")
        receipt = _receipt(identity, encoding_source, native, output, metadata, config.token_scope)
        outputs.append([output, {**metadata, RECEIPT_KEY: receipt}])
        receipts.append(receipt)
        del weights
    cancel()
    return outputs, {"enabled": True, "applied": True, "identity": identity, "items": receipts,
                     "quality": "experimental_not_human_qualified",
                     "warning": "Reference audio/singing can degrade; unchanged audio inputs do not prove generated audio quality."}


# 权重内容识别与推荐参数
WUSHU_V1_SHA = "4b7a459aac43e066b9cd8d3f84f52b7d6c2777e7de05f35a3cb042abab889ebd"
COMIC_SHA = "d4303d77e1ff96b678484d891308eeeb261b919d9e6498610a2bb89f7aeaa9a9"
LEGACY_SETTINGS = {"alpha": .10, "magnitude_match": "per_token",
                   "token_scope": "all_tokens", "chunk_tokens": 256}


def inspect_bridge_profile(path, expected_sha=None):
    data, sha = _read_bridge_bytes(path, expected_sha)
    metadata = _weight_metadata(data)
    header_size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + header_size])
    keys = set(header) - {"__metadata__"}
    settings = None
    contract = None
    name = "Unidentified trainer Bridge"
    source = "manual_required"
    if keys == set(SHAPES):
        for key, shape in SHAPES.items():
            if header[key].get("shape") != list(shape) or header[key].get("dtype") not in ("F16", "BF16", "F32"):
                raise ValueError("Invalid six-tensor Bridge header")
        settings, name, source = dict(LEGACY_SETTINGS), "Original/BUNNY six-tensor MLP", "legacy_defaults"
        arch = "legacy_mlp"
    elif "net.in_proj.weight" in keys:
        spec = spec_from_metadata(metadata)
        contract, arch = spec.application_contract, "trans"
    elif "net.fc1.weight" in keys and metadata.get("arch") == "mlp":
        extra = json.loads(metadata.get("extra_json", "{}"))
        if not isinstance(extra, dict):
            raise ValueError("Invalid trainer Bridge extra_json")
        contract, arch = extra.get("application_contract"), "trainer_mlp"
    else:
        raise ValueError("Unsupported Semantic Bridge weight format; do not select a LoRA, JEV or H3 checkpoint")
    if contract is not None:
        if not isinstance(contract, dict):
            raise ValueError("Invalid Bridge application contract")
        settings = contract_settings(contract)
        source, name = "fixed_model_metadata", "Trainer Bridge (fixed application contract)"
    if sha == COMIC_SHA:
        name = "T8 comic-combat"
        if contract is None:
            raise ValueError("Known comic-combat content is missing its fixed contract")
    elif sha == WUSHU_V1_SHA:
        name, source = "Wushu v1", "sha_pinned_author_recommendation"
        settings = {"alpha": .12, "magnitude_match": "per_token",
                    "token_scope": "all_tokens", "chunk_tokens": 0}
    return {"name": name, "architecture": arch, "sha256": sha, "settings": settings,
            "settings_source": source, "fixed_contract": contract}


def validate_profile_settings(profile, *, alpha, magnitude_match, token_scope, chunk_tokens):
    validate_contract(profile["fixed_contract"], alpha=alpha, magnitude_match=magnitude_match,
                      token_scope=token_scope, chunk_tokens=chunk_tokens)
    actual = {"alpha": alpha, "magnitude_match": magnitude_match,
              "token_scope": token_scope, "chunk_tokens": chunk_tokens}
    settings = profile["settings"]
    return (["Manual settings differ from this weight's recommended preset; use the automatic config "
             "for the preset. Transformer chunking changes cross-token context."]
            if settings is not None and settings != actual else [])


# 权重目录与下载（权重随包放在本节点的 Minimax/models/，也读 ComfyUI/models/semantic_bridge/）
NODE_DIR = Path(__file__).resolve().parent
NODE_MODELS_DIR = NODE_DIR / "models"
LEGACY_DIRS = ("t8_compat",)  # 旧版的 t8_compat 子目录仍会找。
DEFAULT_ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")
USER_AGENT = "comfyui-yuan-semantic-bridge/2.0"
MAX_BYTES = 512 * 1024 * 1024
CHUNK = 1 << 20
_DOWNLOAD_LOCK = threading.RLock()


@dataclass(frozen=True)
class ModelEntry:
    key: str
    label: str
    repo: str
    repo_path: str
    filename: str
    sha256: str
    size: int
    family: str

    def target(self) -> Path:
        return NODE_MODELS_DIR / self.filename


# 两个 *_T8_Compat 的哈希取自发布仓库 SHA256SUMS.txt；动漫战斗与武术 v1 沿用上游按 SHA 固定身份。
CATALOG: tuple[ModelEntry, ...] = (
    ModelEntry(
        key="original_v1",
        label="通用语义",
        repo="t8star/Semantic-Bridge-Comfy",
        repo_path="models/semantic_bridge/t8_compat/MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors",
        filename="MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors",
        sha256="896829a178e89e23f35799d403971b19b10bbacb2da00454c576770b2f58a9b1",
        size=11_024_432,
        family="legacy_mlp",
    ),
    ModelEntry(
        key="bunny_v1",
        label="动作逻辑",
        repo="t8star/Semantic-Bridge-Comfy",
        repo_path="models/semantic_bridge/t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors",
        filename="BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors",
        sha256="601769d0c6311628840a7bc5896b2a1035661f0ae2bddc602b782b955c95d3d8",
        size=22_046_928,
        family="legacy_mlp",
    ),
    ModelEntry(
        key="comic_combat",
        label="T8 动漫战斗",
        repo="t8star/semantic_bridge_T8-comic-combat",
        repo_path="semantic_bridge_T8_comic_combat.safetensors",
        filename="semantic_bridge_T8_comic_combat.safetensors",
        sha256="d4303d77e1ff96b678484d891308eeeb261b919d9e6498610a2bb89f7aeaa9a9",
        size=16_831_336,
        family="trans",
    ),
    ModelEntry(
        key="wushu_v1",
        label="武打逻辑",
        repo="t8star/semantic_bridge_T8-comic-combat",
        repo_path="wushu_bridge_wushu_v1.safetensors",
        filename="wushu_bridge_wushu_v1.safetensors",
        sha256="4b7a459aac43e066b9cd8d3f84f52b7d6c2777e7de05f35a3cb042abab889ebd",
        size=16_832_808,
        family="trans",
    ),
)


def endpoints(override: str | None = None) -> tuple[str, ...]:
    # 镜像列表：节点参数 > 环境变量 > 默认（先镜像后官方）。
    for candidate in (override, os.environ.get("H3_SEMANTIC_BRIDGE_ENDPOINTS")):
        if candidate:
            items = [item.strip().rstrip("/") for item in re.split(r"[,;\s]+", candidate) if item.strip()]
            if items:
                return tuple(items)
    return DEFAULT_ENDPOINTS


def _weight_roots() -> list[Path]:
    # 本节点的 Minimax/models/ 优先，其次 ComfyUI 自己的 semantic_bridge 目录。
    roots = [NODE_MODELS_DIR]
    for path in folder_paths.folder_names_and_paths.get("semantic_bridge", ([], set()))[0]:
        candidate = Path(path)
        if candidate not in roots:
            roots.append(candidate)
    default = Path(folder_paths.models_dir) / "semantic_bridge"
    if default not in roots:
        roots.append(default)
    return roots


def find_installed(entry: ModelEntry) -> Path | None:
    # 已有副本：按 _weight_roots() 顺序找，含旧版 t8_compat 子目录。
    for root in _weight_roots():
        candidates = [root / entry.filename]
        candidates += [root / sub / entry.filename for sub in LEGACY_DIRS]
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def _url(base: str, entry: ModelEntry) -> str:
    return f"{base}/{entry.repo}/resolve/main/{entry.repo_path}?download=true"


def _fetch_once(entry: ModelEntry, target: Path, base: str, progress=None) -> None:
    # 先下到 *.part，哈希核对通过才改名。
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    free = shutil.disk_usage(target.parent).free
    if free < entry.size * 1.1:
        raise OSError(f"磁盘空间不足：需要 {entry.size}，可用 {free}")
    request = urllib.request.Request(_url(base, entry), headers={
        "User-Agent": USER_AGENT, "Accept": "application/octet-stream", "Accept-Encoding": "identity"})
    digest_state = hashlib.sha256()
    received = 0
    started = time.monotonic()
    last_report = 0.0
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as handle:
        declared = response.headers.get("Content-Length")
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            received += len(chunk)
            if received > MAX_BYTES:
                handle.close()
                part.unlink(missing_ok=True)
                raise ValueError(f"{entry.filename} 超出支持的大小")
            digest_state.update(chunk)
            handle.write(chunk)
            now = time.monotonic()
            if progress and now - last_report > 1.0:
                last_report = now
                progress(entry, received, entry.size, now - started)
    if declared and declared.isdigit() and int(declared) != received:
        part.unlink(missing_ok=True)
        raise ValueError(f"{entry.filename} 下载不完整（{received}/{declared} 字节）")
    actual = digest_state.hexdigest()
    if actual != entry.sha256:
        part.unlink(missing_ok=True)
        raise ValueError(f"{entry.filename} 哈希不符（得到 {actual}）")
    os.replace(part, target)


def _verify_structure(path: Path, entry: ModelEntry) -> None:
    # 哈希对了还要能读：必须是本模块认识的桥权重。
    try:
        profile = inspect_bridge_profile(path)
    except Exception as error:
        path.unlink(missing_ok=True)
        raise ValueError(f"{entry.filename} 不是可用的语义桥权重：{error}") from error
    if entry.family != profile["architecture"]:
        path.unlink(missing_ok=True)
        raise ValueError(f"{entry.filename} 架构不符：期望 {entry.family}，实际 {profile['architecture']}")


def download(entry: ModelEntry, *, endpoint: str | None = None, attempts: int = 2,
             progress=None) -> Path:
    # 下载一份权重到插件目录；已存在直接返回。
    with _DOWNLOAD_LOCK:
        existing = find_installed(entry)
        if existing is not None:
            return existing
        target = entry.target()
        errors = []
        for base in endpoints(endpoint):
            for attempt in range(attempts):
                try:
                    _fetch_once(entry, target, base, progress)
                    _verify_structure(target, entry)
                    return target
                except Exception as error:
                    errors.append(f"{base} #{attempt + 1}: {type(error).__name__}: {error}")
        raise RuntimeError(f"{entry.filename} 下载失败：\n  " + "\n  ".join(errors))


def console_progress(entry: ModelEntry, received: int, total: int, elapsed: float) -> None:
    total = total or entry.size
    pct = min(100.0, 100.0 * received / total) if total else 0.0
    rate = received / elapsed / (1024 * 1024) if elapsed > 0 else 0.0
    end = "\n" if received >= total else "\r"
    print(f"[语义桥] {entry.filename}: {received / 1048576:6.1f}/{total / 1048576:.1f} MB "
          f"({pct:5.1f}%) {rate:5.1f} MB/s", end=end, flush=True)


# 下拉框与权重解析
_CATEGORY = "Yuan Tool/MiniMax"
MODE_MANUAL = "手动模式"
MODE_AUTO = "自动模式"
MODES = (MODE_MANUAL, MODE_AUTO)
# 节点 id 带命名空间，可与上游 T8 插件共存；想顶替上游就把这里换成上游 id。
NODE_IDS = {"bridge": "Yuan_H3SemanticBridge"}
# 旧版在模型目录为空时写进工作流的假值，仅用于给出可读提示。
LEGACY_PLACEHOLDER = "No Semantic Bridge models installed"
# 旧版下载行的前缀，兼容旧工作流。
DOWNLOAD_PREFIX = "⬇ 下载: "


def _model_roots():
    # 本节点自带的 Minimax/models/ 优先，其次 ComfyUI 自己的 semantic_bridge 目录。
    return _weight_roots()


def model_paths():
    # 列出可用权重；同名不同路径时带上绝对路径区分。
    grouped = {}
    for root in _model_roots():
        if root.is_dir():
            for path in sorted(root.rglob("*.safetensors")):
                if ".cache" not in path.relative_to(root).parts:
                    grouped.setdefault(path.relative_to(root).as_posix(), set()).add(str(path.resolve()))
    result = {}
    for name, paths in sorted(grouped.items()):
        for path in sorted(paths):
            label = name if len(paths) == 1 else f"{name} [{path}]"
            result[label] = path
    return result


def model_options():
    # 手动模式的下拉：权重文件名（官方 4 份 + 本机额外的）。
    known = [entry.filename for entry in CATALOG]
    options = list(known)
    options.extend(name for name in model_paths() if name.rsplit("/", 1)[-1] not in known)
    return options


def preset_options():
    return [entry.label for entry in CATALOG]

def _model_entry(name):
    # 把下拉值映射到官方权重：认名称、认文件名，也认旧的 t8_compat/ 写法。
    if not isinstance(name, str):
        return None
    name = name[len(DOWNLOAD_PREFIX):] if name.startswith(DOWNLOAD_PREFIX) else name
    bare = name.rsplit("/", 1)[-1]
    for entry in CATALOG:
        if name == entry.label or bare == entry.filename:
            return entry
    return None


def resolve_model(name):
    if name == LEGACY_PLACEHOLDER:
        raise FileNotFoundError(
            "模型仍是旧版空列表占位值 " + repr(LEGACY_PLACEHOLDER) + "："
            "那是模型目录为空时被存进工作流的假值，不是文件。请在下拉框重新选择，"
            "或删掉该节点重新添加。")
    entry = _model_entry(name)
    if entry is not None:
        path = find_installed(entry)
        if path is None:
            raise FileNotFoundError(
                f"「{entry.label}」对应的 {entry.filename} 还没有下载到本机（{entry.repo}）。"
                "保持「自动下载」开启直接运行即可。")
        return str(path)
    path = model_paths().get(name)
    if path is None:
        raise FileNotFoundError(
            "找不到语义桥权重：" + str(name) + "。下拉框里选一个，"
            "或把 .safetensors 放到本节点的 Minimax/models 目录。")
    return path


def ensure_model(name, auto_download=True):
    # 返回 (权重路径, 本次下载到的路径或 None)。
    entry = _model_entry(name)
    if entry is None:
        return resolve_model(name), None
    existing = find_installed(entry)
    if existing is not None:
        return str(existing), None
    if not auto_download:
        raise ValueError(
            f"「{entry.label}」（{entry.filename}）未安装，且「自动下载」已关闭。"
            "打开自动下载，或手动把权重放进本节点的 Minimax/models 目录。")
    print(f"[语义桥] 下载缺失权重：{entry.label}（{entry.filename}）← {entry.repo}")
    fetched = str(download(entry, progress=console_progress))
    return fetched, fetched


# 节点：一个节点完成参数选择与条件应用（V1 schema，参数与独立包一致）
class Yuan_H3SemanticBridge:
    """MiniMax H3 语义桥：选权重与模式，直接把语义桥应用到输入的条件上。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING", {
                "display_name": "条件",
                "tooltip": "原生 H3 条件 [B,T,5120]；投影后的 embeds 会被拒绝。"}),
            "mode": (list(MODES), {
                "default": MODE_MANUAL, "display_name": "模式",
                "tooltip": "手动模式：自己填强度/幅度匹配/作用范围/分块；自动模式：按权重内容自动取推荐参数。"}),
            "model_name": (model_options(), {
                "display_name": "模型（手动）",
                "tooltip": "手动模式用：按文件名选权重，没下载的会在运行时自动下载。也可把其它 .safetensors 放进本节点的 Minimax/models 目录。"}),
            "model_preset": (preset_options(), {
                "default": CATALOG[0].label, "display_name": "模型（自动）",
                "tooltip": "自动模式用：通用语义 / 动作逻辑 / T8 动漫战斗 / 武打逻辑，按权重内容自动取推荐参数。手动模式请用上面的「模型（手动）」。"}),
            "enabled": ("BOOLEAN", {
                "default": True, "display_name": "启用",
                "tooltip": "关闭则完全不读权重，条件原样返回，可用于对照。"}),
            "alpha": ("FLOAT", {
                "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "display_name": "强度（alpha）",
                "tooltip": "0 = 完全不读权重直接旁路。T8 动漫战斗固定 1.0，武打逻辑推荐 0.12。"}),
            "magnitude_match": (["per_token", "global", "none"], {
                "default": "per_token", "display_name": "幅度匹配",
                "tooltip": "per_token 保持每个 token 的幅度（推荐）；global 整条统一缩放；none 不缩放。"}),
            "token_scope": (["all_tokens", "text_only_preserve_reference"], {
                "default": "all_tokens", "display_name": "作用范围",
                "tooltip": "all_tokens 复现作者行为；text_only 只改原生 tag=1 的文本行，需要条件里带 minimax_token_tags。"}),
            "chunk_tokens": ("INT", {
                "default": 256, "min": 0, "max": 65536, "display_name": "分块 token 数",
                "tooltip": "0 = 整条序列（Transformer 类必须 0）；六张量逐 token，分块在数学上等价。"}),
            "device": (["auto", "cpu", "cuda"], {
                "default": "auto", "display_name": "计算设备",
                "tooltip": "权重计算设备。auto 跟随条件张量；显存紧就用 cpu。"}),
            "auto_download": ("BOOLEAN", {
                "default": True, "display_name": "自动下载",
                "tooltip": "选中权重不在本机时自动下载（边下边算 SHA256）。关闭则不联网。"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("条件",)
    OUTPUT_TOOLTIPS = ("已应用语义桥的条件，接采样器。",)
    FUNCTION = "apply"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("H3 语义桥：选权重与模式，直接把语义桥应用到输入的条件上。手动模式自己填强度等参数；"
                   "自动模式按权重内容取推荐参数。权重缺失时会自动下载到本节点的 Minimax/models 目录。"
                   "实验特性：不保证画质或声音质量，请同 seed A/B。")

    @classmethod
    def VALIDATE_INPUTS(cls, conditioning=None, mode=MODE_MANUAL, model_name=None, enabled=True,
                        alpha=0.10, magnitude_match="per_token", token_scope="all_tokens",
                        device="auto", chunk_tokens=256, auto_download=True, model_preset=None,
                        **kwargs):
        # 这些字段被具名后由本方法负责校验，故便宜检查全部重做。
        if mode is not None and mode not in MODES:
            return f"未知模式：{mode}"
        if device is not None and device not in ("auto", "cpu", "cuda"):
            return "未知的计算设备"
        if mode in (None, MODE_MANUAL):
            if alpha is not None:
                try:
                    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
                        return "强度必须是 0 到 1 之间的有限数"
                except TypeError:
                    return "强度必须是数字"
            for value, choices, label in ((magnitude_match, ("per_token", "global", "none"), "幅度匹配"),
                                          (token_scope, ("all_tokens", "text_only_preserve_reference"), "作用范围")):
                if value is not None and value not in choices:
                    return f"未知的{label}"
            if chunk_tokens is not None and (type(chunk_tokens) is not int or not 0 <= chunk_tokens <= 65536):
                return "分块 token 数必须是 0 到 65536 之间的整数"
        selected = model_preset if mode == MODE_AUTO else model_name
        if selected is None:
            return True
        if selected == LEGACY_PLACEHOLDER:
            return "模型是旧版空列表占位值，不是文件：请在下拉框重新选择。"
        entry = _model_entry(selected)
        if entry is not None:
            path = find_installed(entry)
            if path is None:
                # 还没下载：允许自动下载就先放行，其余错误留到执行时报告。
                if auto_download:
                    return True
                return f"「{entry.label}」（{entry.filename}）未安装，且「自动下载」已关闭。"
            installed = str(path)
        else:
            installed = model_paths().get(selected)
            if installed is None:
                try:
                    resolve_model(selected)
                except (ValueError, TypeError, OSError) as error:
                    return str(error)
                return True
        try:
            profile = inspect_bridge_profile(installed)
        except (ValueError, TypeError, OSError) as error:
            return str(error)
        if mode == MODE_AUTO and profile["settings"] is None:
            return "这个权重没有可靠的自动预设；请改用「手动模式」并参考它的模型卡。"
        if mode == MODE_MANUAL and alpha not in (0, None):
            try:
                validate_profile_settings(profile, alpha=alpha, magnitude_match=magnitude_match,
                                          token_scope=token_scope, chunk_tokens=chunk_tokens)
            except (ValueError, TypeError) as error:
                return str(error)
        return True

    @classmethod
    def IS_CHANGED(cls, mode=MODE_MANUAL, model_name=None, enabled=True, alpha=0.10, **kwargs):
        if mode == MODE_MANUAL and (not enabled or alpha == 0):
            return "disabled"
        selected = kwargs.get("model_preset") if mode == MODE_AUTO else model_name
        entry = _model_entry(selected)
        if entry is not None:
            path = find_installed(entry)
            return file_sha(path) if path else "pending:" + entry.sha256
        path = model_paths().get(selected) if isinstance(selected, str) else None
        return file_sha(path) if path else "pending:unknown"

    def apply(self, conditioning, mode=MODE_MANUAL, model_name="", enabled=True, alpha=0.10,
              magnitude_match="per_token", token_scope="all_tokens", chunk_tokens=256,
              device="auto", auto_download=True, model_preset=None):
        mode = mode if mode in MODES else MODE_MANUAL
        if mode == MODE_MANUAL and (not enabled or alpha == 0):
            print("[语义桥] 强度为 0 或已停用：条件原样返回（未读权重）")
            return (conditioning,)

        selected = model_preset if mode == MODE_AUTO else model_name
        path, downloaded = ensure_model(selected, auto_download)
        profile = inspect_bridge_profile(path)
        weight = Path(path).name

        if mode == MODE_MANUAL:
            config = BridgeConfig(path, profile["sha256"], alpha, magnitude_match,
                                  token_scope, device, chunk_tokens, enabled)
            warnings = validate_profile_settings(profile, alpha=alpha, magnitude_match=magnitude_match,
                                                  token_scope=token_scope, chunk_tokens=chunk_tokens)
            detail = f"alpha={alpha} 匹配={magnitude_match} 范围={token_scope} 分块={chunk_tokens}"
            source = ""
        else:
            settings = profile["settings"]
            if settings is None:
                raise ValueError("这个权重没有可靠的自动预设；请改用「手动模式」并参考它的模型卡。")
            config = BridgeConfig(path, profile["sha256"], device=device, **settings)
            validate_profile_settings(profile, **settings)
            warnings = []
            detail = (f"alpha={settings['alpha']} 匹配={settings['magnitude_match']} "
                      f"范围={settings['token_scope']} 分块={settings['chunk_tokens']}")
            source = f"（{selected}）"

        result, _ = apply_bridge(conditioning, config)
        print(f"[语义桥] 模式={mode} 权重={weight}{source} {detail}"
              + (f" 已下载={downloaded}" if downloaded else "")
              + (" 提示=" + "；".join(warnings) if warnings else ""))
        return (result,)


NODE_CLASS_MAPPINGS = {NODE_IDS["bridge"]: Yuan_H3SemanticBridge}

NODE_DISPLAY_NAME_MAPPINGS = {NODE_IDS["bridge"]: "H3 语义桥"}
