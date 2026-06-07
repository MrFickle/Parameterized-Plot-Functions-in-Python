"""Natural-language PlotSpec generation with deterministic caching."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .render import render_plot
from .specs import RenderResult, get_all_plot_schemas, get_plot_schema, list_plot_types, validate_plot_spec


SCHEMA_VERSION = "plotspec-pydantic-v1"


@dataclass
class LLMPlotConfig:
    """
    Function purpose:
        Store configuration for LLM-backed PlotSpec generation.

    Args:
        provider: LLM provider name. Currently only "openai" is supported.
        model: Provider model name.
        api_key: Provider API key.
        cache_dir: Directory used for deterministic generated-spec cache files.

    Outputs:
        Dataclass containing LLM plotting configuration.
    """

    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    cache_dir: str = ".ppf_cache"

    @classmethod
    def from_env(cls) -> LLMPlotConfig:
        """
        Function purpose:
            Build LLM plotting configuration from environment variables.

        Args:
            None.

        Outputs:
            LLMPlotConfig populated from environment values and defaults.
        """
        return cls(
            provider=os.getenv("PPF_LLM_PROVIDER", "openai"),
            model=os.getenv("PPF_LLM_MODEL", "gpt-4.1-mini"),
            api_key=os.getenv("PPF_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            cache_dir=os.getenv("PPF_LLM_CACHE_DIR", ".ppf_cache"),
        )


def load_llm_config(path: str | Path) -> LLMPlotConfig:
    """
    Function purpose:
        Load LLM plotting configuration from a JSON or YAML file.

    Args:
        path: Path to a JSON, YAML, or YML config file.

    Outputs:
        LLMPlotConfig populated from the config file.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"LLM config file does not exist: {config_path}")
    if config_path.suffix.lower() == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("LLM config file must end with .json, .yaml, or .yml.")
    return LLMPlotConfig(**data)


def plot_from_instructions(
    data: Any,
    instructions: str,
    plot_type: str | None = None,
    config: LLMPlotConfig | None = None,
    config_path: str | Path | None = None,
    rerun_every_time: bool = False,
    output: dict[str, Any] | None = None,
) -> RenderResult:
    """
    Function purpose:
        Generate or reuse a cached PlotSpec from natural-language instructions and render it.

    Args:
        data: DataFrame-like object, list of dictionaries, or dictionary used by the generated PlotSpec.
        instructions: Plain-English plotting request.
        plot_type: Optional plot type constraint. When omitted, the LLM chooses the plot type.
        config: Optional in-memory LLM configuration.
        config_path: Optional path to a JSON/YAML LLM configuration file.
        rerun_every_time: Whether to ignore the cache and ask the LLM again.
        output: Optional output config merged into the generated PlotSpec before rendering.

    Outputs:
        RenderResult from rendering the generated or cached PlotSpec.
    """
    llm_config = _resolve_config(config=config, config_path=config_path)
    data_profile = profile_data_for_llm(data)
    cache_path = _cache_path(llm_config.cache_dir, instructions, plot_type, data_profile, llm_config.model)

    if cache_path.exists() and not rerun_every_time:
        spec = json.loads(cache_path.read_text(encoding="utf-8"))["generated_spec"]
    else:
        spec = generate_plot_spec_from_instructions(
            data_profile=data_profile,
            instructions=instructions,
            plot_type=plot_type,
            config=llm_config,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "instructions": instructions,
            "plot_type": plot_type,
            "provider": llm_config.provider,
            "model": llm_config.model,
            "schema_version": SCHEMA_VERSION,
            "data_profile": data_profile,
            "generated_spec": spec,
        }
        cache_path.write_text(json.dumps(cache_payload, indent=2, sort_keys=True), encoding="utf-8")

    if output is not None:
        spec["output"] = {**spec.get("output", {}), **output}

    normalized_spec = validate_plot_spec(spec)
    return render_plot(normalized_spec, dataframes={"data": data})


def generate_plot_spec_from_instructions(
    data_profile: dict[str, Any],
    instructions: str,
    plot_type: str | None,
    config: LLMPlotConfig,
) -> dict[str, Any]:
    """
    Function purpose:
        Ask the configured LLM to generate a PlotSpec from instructions and a data profile.

    Args:
        data_profile: JSON-serializable profile of the user's data.
        instructions: Plain-English plotting request.
        plot_type: Optional plot type constraint.
        config: LLM plotting configuration.

    Outputs:
        Validated PlotSpec dictionary.
    """
    if config.provider != "openai":
        raise ValueError("Only the 'openai' LLM provider is currently supported.")
    schema_payload = get_plot_schema(plot_type) if plot_type is not None else get_all_plot_schemas()
    prompt = _build_spec_generation_prompt(data_profile, instructions, plot_type, schema_payload)
    spec = _call_openai_for_spec(prompt=prompt, config=config)
    spec.setdefault("data", {"dataframe": {"name": "data", "mappings": {}}})
    return validate_plot_spec(spec)


def profile_data_for_llm(data: Any, sample_rows: int = 5) -> dict[str, Any]:
    """
    Function purpose:
        Build a compact, JSON-safe profile of data for LLM PlotSpec generation.

    Args:
        data: DataFrame-like object, list of dictionaries, or dictionary.
        sample_rows: Maximum number of sample rows to include.

    Outputs:
        Dictionary describing columns, dtypes, shape, and sample values when available.
    """
    if hasattr(data, "to_dict") and hasattr(data, "shape"):
        records = data.head(sample_rows).to_dict(orient="records") if hasattr(data, "head") else data.to_dict(orient="records")[:sample_rows]
        columns = list(getattr(data, "columns", []))
        dtypes = {str(column): str(dtype) for column, dtype in getattr(data, "dtypes", {}).items()}
        return {"kind": "dataframe", "shape": list(data.shape), "columns": columns, "dtypes": dtypes, "sample_rows": _json_safe(records)}
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        columns = sorted({key for row in data for key in row})
        return {"kind": "records", "shape": [len(data), len(columns)], "columns": columns, "sample_rows": _json_safe(data[:sample_rows])}
    if isinstance(data, dict):
        return {"kind": "mapping", "keys": list(data.keys()), "sample": _json_safe({key: data[key] for key in list(data.keys())[:sample_rows]})}
    return {"kind": type(data).__name__, "repr": repr(data)[:500]}


def _resolve_config(config: LLMPlotConfig | None, config_path: str | Path | None) -> LLMPlotConfig:
    """Resolve explicit, file-based, or environment-based LLM config."""
    if config is not None:
        return config
    if config_path is not None:
        return load_llm_config(config_path)
    return LLMPlotConfig.from_env()


def _cache_path(cache_dir: str, instructions: str, plot_type: str | None, data_profile: dict[str, Any], model: str) -> Path:
    """Build the deterministic cache path for generated PlotSpecs."""
    key_payload = {
        "instructions": instructions,
        "plot_type": plot_type,
        "data_profile": data_profile,
        "model": model,
        "schema_version": SCHEMA_VERSION,
    }
    digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def _build_spec_generation_prompt(data_profile: dict[str, Any], instructions: str, plot_type: str | None, schema_payload: dict[str, Any]) -> str:
    """Build the JSON-only prompt sent to the LLM."""
    allowed_plot_types = [plot_type] if plot_type is not None else list_plot_types()
    return json.dumps(
        {
            "task": "Generate one valid PlotSpec JSON object. Return JSON only.",
            "instructions": instructions,
            "allowed_plot_types": allowed_plot_types,
            "data_profile": data_profile,
            "data_contract": "Use dataframe mode with name 'data' and explicit mappings for columns when tabular data is provided.",
            "schema": schema_payload,
        },
        indent=2,
        sort_keys=True,
    )


def _call_openai_for_spec(prompt: str, config: LLMPlotConfig) -> dict[str, Any]:
    """Call OpenAI and parse the returned PlotSpec JSON."""
    if not config.api_key:
        raise ValueError("OpenAI API key is required. Set PPF_LLM_API_KEY, OPENAI_API_KEY, config.api_key, or config_path.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI support requires the optional dependency: pip install 'parameterized-plot-functions-in-python[llm]'") from exc
    client = OpenAI(api_key=config.api_key)
    response = client.responses.create(
        model=config.model,
        input=prompt,
        text={"format": {"type": "json_object"}},
    )
    return json.loads(response.output_text)


def _json_safe(value: Any) -> Any:
    """Convert common non-JSON scalar values into JSON-safe equivalents."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return str(value)
