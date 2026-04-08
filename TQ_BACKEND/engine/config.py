from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

KVQuantScheme = Literal["none", "mlx", "turboquant"]
SamplingProfile = Literal[
    "controlled",
    "best_effort_general",
    "best_effort_reasoning",
]

ALLOWED_MLX_KV_BITS = (2.0, 3.0, 4.0)
ALLOWED_TURBOQUANT_KV_BITS = (2.0, 2.5, 3.0, 3.5, 4.0)
SUPPORTED_SAMPLING_PROFILES = (
    "controlled",
    "best_effort_general",
    "best_effort_reasoning",
)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


class SettingsError(ValueError):
    """Raised when the backend configuration is invalid."""


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float | None
    repetition_penalty: float | None


@dataclass(frozen=True)
class ServerSettings:
    env_files: tuple[str, ...]
    host: str
    port: int
    api_key: str | None
    max_concurrent_requests: int
    model_id: str
    adapter_path: str | None
    revision: str | None
    trust_remote_code: bool
    preload_model: bool
    log_level: str
    sampling_profile: SamplingProfile
    controlled_sampling: SamplingSettings
    best_effort_general_sampling: SamplingSettings
    best_effort_reasoning_sampling: SamplingSettings
    default_max_output_tokens: int
    default_repetition_context_size: int
    default_presence_context_size: int
    default_prefill_step_size: int
    force_disable_thinking: bool
    kv_quant_scheme: KVQuantScheme
    kv_bits: float | None
    kv_group_size: int
    quantized_kv_start: int
    turboquant_seed: int
    turboquant_from_first_token: bool
    paged_ssd_cache_dir: str | None

    @property
    def active_sampling(self) -> SamplingSettings:
        if self.sampling_profile == "best_effort_general":
            return self.best_effort_general_sampling
        if self.sampling_profile == "best_effort_reasoning":
            return self.best_effort_reasoning_sampling
        return self.controlled_sampling

    @property
    def active_kv_quant_scheme(self) -> str | None:
        if self.kv_quant_scheme == "none":
            return None
        if self.kv_quant_scheme == "mlx":
            return "uniform"
        return self.kv_quant_scheme

    @property
    def use_turboquant_prompt_cache(self) -> bool:
        return (
            self.kv_quant_scheme == "turboquant"
            and self.kv_bits is not None
            and self.turboquant_from_first_token
        )

    @property
    def cache_namespace(self) -> str:
        revision = self.revision or "main"
        adapter = self.adapter_path or "-"
        bits = "none" if self.kv_bits is None else f"{self.kv_bits:g}"
        return "|".join(
            (
                f"model={self.model_id}",
                f"revision={revision}",
                f"adapter={adapter}",
                f"scheme={self.kv_quant_scheme}",
                f"bits={bits}",
                f"group={self.kv_group_size}",
                f"start={self.quantized_kv_start}",
                f"tq_seed={self.turboquant_seed}",
                f"tq_first={int(self.turboquant_from_first_token)}",
            )
        )


def _load_env_files() -> tuple[str, ...]:
    project_dir = Path(__file__).resolve().parents[1]
    explicit = os.environ.get("TQ_ENV_FILE")

    candidates: list[Path] = []
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = (Path.cwd() / explicit_path).resolve()
        candidates.append(explicit_path)

    candidates.extend(
        [
            (Path.cwd() / ".env").resolve(),
            (project_dir / ".env").resolve(),
        ]
    )

    loaded: list[str] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(str(path))

    return tuple(loaded)


def _read_raw(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _read_first_raw(*names: str) -> str | None:
    for name in names:
        value = _read_raw(name)
        if value is not None:
            return value
    return None


def _read_str(name: str, default: str) -> str:
    value = _read_raw(name)
    return default if value is None else value


def _read_optional_str(name: str) -> str | None:
    return _read_raw(name)


def _read_bool(name: str, default: bool) -> bool:
    value = _read_raw(name)
    if value is None:
        return default

    normalized = value.lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise SettingsError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}."
    )


def _parse_int(name: str, raw: str, *, minimum: int | None = None) -> int:
    try:
        result = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer.") from exc

    if minimum is not None and result < minimum:
        raise SettingsError(f"{name} must be >= {minimum}.")
    return result


def _read_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = _read_raw(name)
    if value is None:
        return default
    return _parse_int(name, value, minimum=minimum)


def _parse_float(
    name: str,
    raw: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a float.") from exc

    if minimum is not None and result < minimum:
        raise SettingsError(f"{name} must be >= {minimum}.")
    if maximum is not None and result > maximum:
        raise SettingsError(f"{name} must be <= {maximum}.")
    return result


def _read_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = _read_raw(name)
    if value is None:
        return default
    return _parse_float(name, value, minimum=minimum, maximum=maximum)


def _read_optional_float(
    name: str,
    default: float | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    value = _read_raw(name)
    if value is None:
        return default
    return _parse_float(name, value, minimum=minimum, maximum=maximum)


def _read_profile_float(
    primary: str,
    fallback: str | None,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = _read_first_raw(primary, *( [fallback] if fallback else [] ))
    if raw is None:
        return default
    return _parse_float(primary, raw, minimum=minimum, maximum=maximum)


def _read_profile_optional_float(
    primary: str,
    fallback: str | None,
    default: float | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    raw = _read_first_raw(primary, *( [fallback] if fallback else [] ))
    if raw is None:
        return default
    return _parse_float(primary, raw, minimum=minimum, maximum=maximum)


def _read_profile_int(
    primary: str,
    fallback: str | None,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    raw = _read_first_raw(primary, *( [fallback] if fallback else [] ))
    if raw is None:
        return default
    return _parse_int(primary, raw, minimum=minimum)


def canonicalize_kv_quant_scheme(value: str | None) -> KVQuantScheme:
    if value is None:
        return "turboquant"

    normalized = value.strip().lower()
    aliases = {
        "none": "none",
        "off": "none",
        "mlx": "mlx",
        "default": "mlx",
        "uniform": "mlx",
        "turboquant": "turboquant",
    }
    if normalized not in aliases:
        raise SettingsError(
            "KV quantization scheme must be one of: none, mlx, turboquant."
        )
    return aliases[normalized]


def default_kv_bits_for_scheme(scheme: KVQuantScheme) -> float | None:
    if scheme == "none":
        return None
    if scheme == "mlx":
        return 4.0
    return 3.5


def validate_kv_bits(
    scheme: KVQuantScheme,
    bits: float | None,
    *,
    source: str = "kv_bits",
) -> float | None:
    if scheme == "none":
        return None
    if bits is None:
        raise SettingsError(f"{source} is required when KV quantization is enabled.")

    bits = float(bits)
    allowed = (
        ALLOWED_MLX_KV_BITS if scheme == "mlx" else ALLOWED_TURBOQUANT_KV_BITS
    )
    if bits not in allowed:
        allowed_text = ", ".join(
            str(int(v)) if float(v).is_integer() else str(v) for v in allowed
        )
        raise SettingsError(
            f"{source}={bits} is not allowed for scheme={scheme}. "
            f"Allowed values: {allowed_text}."
        )
    return bits


def validate_sampling_profile(value: str) -> SamplingProfile:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_SAMPLING_PROFILES:
        allowed = ", ".join(SUPPORTED_SAMPLING_PROFILES)
        raise SettingsError(f"TQ_SAMPLING_PROFILE must be one of: {allowed}.")
    return normalized  # type: ignore[return-value]


def _build_sampling_settings(
    prefix: str,
    *,
    default_temperature: float,
    default_top_p: float,
    default_top_k: int,
    default_min_p: float,
    default_presence_penalty: float | None,
    default_repetition_penalty: float | None,
    compatibility_prefix: str | None = None,
) -> SamplingSettings:
    return SamplingSettings(
        temperature=_read_profile_float(
            f"{prefix}_TEMPERATURE",
            f"{compatibility_prefix}_TEMPERATURE" if compatibility_prefix else None,
            default_temperature,
            minimum=0.0,
        ),
        top_p=_read_profile_float(
            f"{prefix}_TOP_P",
            f"{compatibility_prefix}_TOP_P" if compatibility_prefix else None,
            default_top_p,
            minimum=0.0,
            maximum=1.0,
        ),
        top_k=_read_profile_int(
            f"{prefix}_TOP_K",
            f"{compatibility_prefix}_TOP_K" if compatibility_prefix else None,
            default_top_k,
            minimum=0,
        ),
        min_p=_read_profile_float(
            f"{prefix}_MIN_P",
            f"{compatibility_prefix}_MIN_P" if compatibility_prefix else None,
            default_min_p,
            minimum=0.0,
            maximum=1.0,
        ),
        presence_penalty=_read_profile_optional_float(
            f"{prefix}_PRESENCE_PENALTY",
            None,
            default_presence_penalty,
            minimum=0.0,
        ),
        repetition_penalty=_read_profile_optional_float(
            f"{prefix}_REPETITION_PENALTY",
            f"{compatibility_prefix}_REPETITION_PENALTY"
            if compatibility_prefix
            else None,
            default_repetition_penalty,
            minimum=0.0,
        ),
    )


def _validate_sampling_settings(name: str, settings: SamplingSettings) -> None:
    if settings.temperature > 0.0 and settings.top_p == 0.0:
        raise SettingsError(f"{name}: top_p must be > 0 when temperature > 0.")


def load_settings() -> ServerSettings:
    env_files = _load_env_files()

    sampling_profile = validate_sampling_profile(
        _read_str("TQ_SAMPLING_PROFILE", "controlled")
    )

    controlled_sampling = _build_sampling_settings(
        "TQ_CONTROLLED",
        default_temperature=0.0,
        default_top_p=1.0,
        default_top_k=0,
        default_min_p=0.0,
        default_presence_penalty=None,
        default_repetition_penalty=None,
        compatibility_prefix="TQ_DEFAULT",
    )
    best_effort_general_sampling = _build_sampling_settings(
        "TQ_BEST_EFFORT_GENERAL",
        default_temperature=0.7,
        default_top_p=0.8,
        default_top_k=20,
        default_min_p=0.0,
        default_presence_penalty=1.5,
        default_repetition_penalty=None,
    )
    best_effort_reasoning_sampling = _build_sampling_settings(
        "TQ_BEST_EFFORT_REASONING",
        default_temperature=1.0,
        default_top_p=0.95,
        default_top_k=20,
        default_min_p=0.0,
        default_presence_penalty=1.5,
        default_repetition_penalty=None,
    )

    _validate_sampling_settings("controlled", controlled_sampling)
    _validate_sampling_settings("best_effort_general", best_effort_general_sampling)
    _validate_sampling_settings("best_effort_reasoning", best_effort_reasoning_sampling)

    kv_quant_scheme = canonicalize_kv_quant_scheme(
        _read_str("TQ_KV_QUANT_SCHEME", "turboquant")
    )
    kv_bits = _read_optional_float(
        "TQ_KV_BITS",
        default_kv_bits_for_scheme(kv_quant_scheme),
        minimum=1.0,
    )
    kv_bits = validate_kv_bits(kv_quant_scheme, kv_bits, source="TQ_KV_BITS")

    force_disable_thinking = _read_bool("TQ_FORCE_DISABLE_THINKING", True)
    if not force_disable_thinking:
        raise SettingsError(
            "This backend is benchmark-oriented and currently requires "
            "TQ_FORCE_DISABLE_THINKING=true."
        )

    return ServerSettings(
        env_files=env_files,
        host=_read_str("TQ_HOST", "0.0.0.0"),
        port=_read_int("TQ_PORT", 8000, minimum=1),
        api_key=_read_optional_str("TQ_API_KEY"),
        max_concurrent_requests=_read_int(
            "TQ_MAX_CONCURRENT_REQUESTS", 1, minimum=1
        ),
        model_id=_read_str("TQ_MODEL", "mlx-community/Qwen3.5-0.8B-MLX-bf16"),
        adapter_path=_read_optional_str("TQ_ADAPTER_PATH"),
        revision=_read_optional_str("TQ_REVISION"),
        trust_remote_code=_read_bool("TQ_TRUST_REMOTE_CODE", False),
        preload_model=_read_bool("TQ_PRELOAD_MODEL", True),
        log_level=_read_str("TQ_LOG_LEVEL", "INFO").upper(),
        sampling_profile=sampling_profile,
        controlled_sampling=controlled_sampling,
        best_effort_general_sampling=best_effort_general_sampling,
        best_effort_reasoning_sampling=best_effort_reasoning_sampling,
        default_max_output_tokens=_read_int(
            "TQ_DEFAULT_MAX_OUTPUT_TOKENS", 512, minimum=1
        ),
        default_repetition_context_size=_read_int(
            "TQ_DEFAULT_REPETITION_CONTEXT_SIZE", 20, minimum=1
        ),
        default_presence_context_size=_read_int(
            "TQ_DEFAULT_PRESENCE_CONTEXT_SIZE", 20, minimum=1
        ),
        default_prefill_step_size=_read_int("TQ_PREFILL_STEP_SIZE", 2048, minimum=1),
        force_disable_thinking=force_disable_thinking,
        kv_quant_scheme=kv_quant_scheme,
        kv_bits=kv_bits,
        kv_group_size=_read_int("TQ_KV_GROUP_SIZE", 64, minimum=1),
        quantized_kv_start=_read_int("TQ_QUANTIZED_KV_START", 0, minimum=0),
        turboquant_seed=_read_int("TQ_TURBOQUANT_SEED", 0),
        turboquant_from_first_token=_read_bool(
            "TQ_TURBOQUANT_FROM_FIRST_TOKEN", True
        ),
        paged_ssd_cache_dir=_resolve_paged_ssd_cache_dir(),
    )


def _resolve_paged_ssd_cache_dir() -> str | None:
    """Resolve the paged SSD cache directory from environment.

    omlx's :class:`SchedulerConfig.paged_ssd_cache_dir` enables the
    paged KV-cache *and* the SSD-backed ``VisionFeatureSSDCache`` (it
    creates ``<dir>/vision_features``). The latter is what makes vision
    features survive runtime reloads, which is critical for
    sweep-style benchmark runs that reload the model 5-7 times per
    sample image.

    - If ``TQ_OMLX_CACHE_DIR`` is unset → defaults to
      ``~/.omlx_bench/cache``
    - If set to an empty string → disables SSD cache (back to in-memory only)
    """
    raw = os.environ.get("TQ_OMLX_CACHE_DIR")
    if raw is None:
        return str((Path.home() / ".omlx_bench" / "cache").resolve())
    raw = raw.strip()
    if not raw:
        return None
    return str(Path(raw).expanduser().resolve())


def replace_settings(
    base: ServerSettings,
    *,
    model_id: str | None = None,
    adapter_path: str | None = None,
    revision: str | None = None,
    kv_quant_scheme: str | None = None,
    kv_bits: float | None = None,
    sampling_profile: str | None = None,
) -> ServerSettings:
    next_scheme = (
        canonicalize_kv_quant_scheme(kv_quant_scheme)
        if kv_quant_scheme is not None
        else base.kv_quant_scheme
    )
    bit_candidate = kv_bits if kv_bits is not None else base.kv_bits
    try:
        next_bits = validate_kv_bits(next_scheme, bit_candidate, source="kv_bits")
    except SettingsError:
        if kv_bits is None:
            next_bits = default_kv_bits_for_scheme(next_scheme)
            next_bits = validate_kv_bits(next_scheme, next_bits, source="kv_bits")
        else:
            raise

    next_profile = (
        validate_sampling_profile(sampling_profile)
        if sampling_profile is not None
        else base.sampling_profile
    )

    return replace(
        base,
        model_id=model_id if model_id is not None else base.model_id,
        adapter_path=adapter_path if adapter_path is not None else base.adapter_path,
        revision=revision if revision is not None else base.revision,
        kv_quant_scheme=next_scheme,
        kv_bits=next_bits,
        sampling_profile=next_profile,
    )
