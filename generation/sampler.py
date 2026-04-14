"""
Multi-backend LLM sampler for PDDL candidate generation.

Supports AWS Bedrock, OpenRouter, OpenAI, and local HuggingFace models.
Credentials are loaded from .env via python-dotenv. Backends that cannot
initialize (missing keys, unavailable SDK) are skipped gracefully.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from generation.prompts import (
    SYSTEM_PROMPT,
    make_generation_prompt,
    extract_pddl_from_response,
)

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class SamplerConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 1024
    retry_attempts: int = 3
    retry_delay_sec: float = 2.0


@dataclass
class SampleResult:
    """Result from a single LLM call."""
    raw_response: str
    extracted_pddl: str
    backend: str
    model: str
    latency_sec: float = 0.0
    error: Optional[str] = None


class BaseSampler(ABC):
    """Abstract base for all LLM backends."""

    backend_name: str = "base"

    def __init__(self, model: str, config: Optional[SamplerConfig] = None):
        self.model = model
        self.config = config or SamplerConfig()

    @abstractmethod
    def _call_llm(self, prompt: str, system: str) -> str:
        """Send a single prompt to the LLM and return the raw response text."""

    def sample(
        self,
        natural_language: str,
        domain: str = "",
        K: int = 4,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> list[SampleResult]:
        """
        Generate K PDDL candidates for a natural-language description.
        Returns a list of SampleResult (may be shorter than K on errors).
        """
        prompt = make_generation_prompt(natural_language, domain=domain)
        temp_backup = self.config.temperature
        topp_backup = self.config.top_p
        if temperature is not None:
            self.config.temperature = temperature
        if top_p is not None:
            self.config.top_p = top_p

        results = []
        for i in range(K):
            result = self._sample_one(prompt, attempt_idx=i)
            results.append(result)

        self.config.temperature = temp_backup
        self.config.top_p = topp_backup
        return results

    # Errors that should not be retried (deterministic failures)
    NON_RETRYABLE_PATTERNS = [
        "model identifier is invalid",
        "AccessDeniedException",
        "not authorized",
        "api key",
        "authentication",
        "invalid_api_key",
    ]

    def _is_retryable(self, error_msg: str) -> bool:
        error_lower = error_msg.lower()
        return not any(p.lower() in error_lower for p in self.NON_RETRYABLE_PATTERNS)

    def _sample_one(self, prompt: str, attempt_idx: int = 0) -> SampleResult:
        """Generate a single sample with retry logic."""
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                t0 = time.time()
                raw = self._call_llm(prompt, system=SYSTEM_PROMPT)
                latency = time.time() - t0
                pddl = extract_pddl_from_response(raw)
                return SampleResult(
                    raw_response=raw,
                    extracted_pddl=pddl,
                    backend=self.backend_name,
                    model=self.model,
                    latency_sec=latency,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"{self.backend_name} attempt {attempt+1}/{self.config.retry_attempts} "
                    f"failed: {last_error}"
                )
                if not self._is_retryable(last_error):
                    logger.warning(f"{self.backend_name}: non-retryable error, giving up")
                    break
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay_sec * (2 ** attempt)
                    time.sleep(delay)

        return SampleResult(
            raw_response="",
            extracted_pddl="",
            backend=self.backend_name,
            model=self.model,
            error=last_error,
        )


class BedrockSampler(BaseSampler):
    """AWS Bedrock backend using the Converse API."""

    backend_name = "bedrock"

    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[SamplerConfig] = None,
    ):
        resolved_model = model or os.environ.get("BEDROCK_MODEL_ID", "")
        if not resolved_model:
            raise ValueError(
                "No Bedrock model ID. Set BEDROCK_MODEL_ID in .env or pass model=."
            )
        super().__init__(resolved_model, config)

        import boto3
        from botocore.config import Config

        read_timeout = int(os.environ.get("BEDROCK_READ_TIMEOUT", "600"))
        boto_config = Config(
            read_timeout=read_timeout,
            connect_timeout=60,
            retries={"max_attempts": 5, "mode": "adaptive"},
        )
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            config=boto_config,
        )
        logger.info(f"BedrockSampler initialized: model={self.model}")

    def _bedrock_inference_config(self) -> dict:
        """
        Build inferenceConfig for Converse. Some Anthropic models on Bedrock reject
        specifying both temperature and topP; send only one (temperature by default).
        Set BEDROCK_USE_TOP_P=1 to use top_p instead of temperature.
        """
        cfg: dict = {"maxTokens": self.config.max_new_tokens}
        if os.environ.get("BEDROCK_USE_TOP_P", "").lower() in ("1", "true", "yes"):
            cfg["topP"] = self.config.top_p
        else:
            cfg["temperature"] = self.config.temperature
        return cfg

    def _call_llm(self, prompt: str, system: str) -> str:
        try:
            return self._call_converse(prompt, system)
        except Exception as e:
            if "model identifier is invalid" in str(e).lower():
                logger.info("Converse API rejected model ID, falling back to invoke_model")
                return self._call_invoke_model(prompt, system)
            raise

    def _call_converse(self, prompt: str, system: str) -> str:
        response = self._client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=[{"text": system}],
            inferenceConfig=self._bedrock_inference_config(),
        )
        return response["output"]["message"]["content"][0]["text"]

    def _call_invoke_model(self, prompt: str, system: str) -> str:
        """Fallback for cross-region inference profiles or older model IDs."""
        payload: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_new_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if os.environ.get("BEDROCK_USE_TOP_P", "").lower() in ("1", "true", "yes"):
            payload["top_p"] = self.config.top_p
        else:
            payload["temperature"] = self.config.temperature
        body = json.dumps(payload)
        response = self._client.invoke_model(
            modelId=self.model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]


class OpenRouterSampler(BaseSampler):
    """OpenRouter backend (OpenAI-compatible API)."""

    backend_name = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "meta-llama/llama-3-8b-instruct",
        config: Optional[SamplerConfig] = None,
    ):
        super().__init__(model, config)
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("No OPENROUTER_API_KEY found in .env")

        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.BASE_URL,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/vcsr-research",
                "X-Title": "VCSR-NegGen",
            },
        )
        logger.info(f"OpenRouterSampler initialized: model={self.model}")

    def _call_llm(self, prompt: str, system: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        return response.choices[0].message.content


class OpenAISampler(BaseSampler):
    """Direct OpenAI API backend."""

    backend_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        config: Optional[SamplerConfig] = None,
    ):
        super().__init__(model, config)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("No OPENAI_API_KEY found in .env")

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        logger.info(f"OpenAISampler initialized: model={self.model}")

    def _call_llm(self, prompt: str, system: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        return response.choices[0].message.content


class HuggingFaceSampler(BaseSampler):
    """Local HuggingFace model via transformers pipeline."""

    backend_name = "huggingface"

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        config: Optional[SamplerConfig] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model, config)

        import torch
        from transformers import pipeline

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            device=resolved_device,
            torch_dtype=torch.float16 if resolved_device == "cuda" else torch.float32,
        )
        logger.info(
            f"HuggingFaceSampler initialized: model={self.model}, device={resolved_device}"
        )

    def _call_llm(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        outputs = self._pipe(
            messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
        )
        return outputs[0]["generated_text"][-1]["content"]


BACKEND_REGISTRY: dict[str, type[BaseSampler]] = {
    "bedrock": BedrockSampler,
    "openrouter": OpenRouterSampler,
    "openai": OpenAISampler,
    "huggingface": HuggingFaceSampler,
}


@dataclass
class BackendSpec:
    """Parsed backend specification from config."""
    type: str
    model: Optional[str] = None
    K: int = 2
    extra_kwargs: dict = field(default_factory=dict)


class MultiSampler:
    """
    Wraps multiple backends and distributes K samples across them.
    Backends that fail to initialize are skipped with a warning.
    """

    def __init__(
        self,
        backend_specs: list[dict],
        config: Optional[SamplerConfig] = None,
    ):
        self.config = config or SamplerConfig()
        self.backends: list[tuple[BaseSampler, int]] = []
        self._init_backends(backend_specs)

    def _init_backends(self, specs: list[dict]):
        for spec_dict in specs:
            spec = BackendSpec(
                type=spec_dict["type"],
                model=spec_dict.get("model"),
                K=spec_dict.get("K", 2),
                extra_kwargs={
                    k: v for k, v in spec_dict.items()
                    if k not in ("type", "model", "K")
                },
            )

            cls = BACKEND_REGISTRY.get(spec.type)
            if cls is None:
                logger.warning(f"Unknown backend type '{spec.type}', skipping")
                continue

            kwargs = {"config": self.config}
            if spec.model:
                kwargs["model"] = spec.model
            kwargs.update(spec.extra_kwargs)

            try:
                backend = cls(**kwargs)
                self.backends.append((backend, spec.K))
                logger.info(
                    f"Backend '{spec.type}' ready (model={backend.model}, K={spec.K})"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize backend '{spec.type}': {e} -- skipping"
                )

        if not self.backends:
            raise RuntimeError(
                "No backends initialized successfully. "
                "Check .env credentials and backend configs."
            )

        total_k = sum(k for _, k in self.backends)
        logger.info(
            f"MultiSampler ready: {len(self.backends)} backend(s), total K={total_k}"
        )

    @property
    def total_k(self) -> int:
        return sum(k for _, k in self.backends)

    def sample(
        self,
        natural_language: str,
        domain: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> list[SampleResult]:
        """
        Generate candidates from all backends according to their K allocation.
        """
        all_results = []
        for backend, k in self.backends:
            results = backend.sample(
                natural_language,
                domain=domain,
                K=k,
                temperature=temperature,
                top_p=top_p,
            )
            all_results.extend(results)
        return all_results

    def sample_single_backend(
        self,
        backend_idx: int,
        natural_language: str,
        domain: str = "",
        K: Optional[int] = None,
    ) -> list[SampleResult]:
        """Generate from a specific backend by index."""
        backend, default_k = self.backends[backend_idx]
        return backend.sample(natural_language, domain=domain, K=K or default_k)
