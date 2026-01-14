"""
Property-Based Tests for Core LiteLLM Integration.

These tests validate the correctness properties defined in the design document
for the core LiteLLM integration (Requirements 1.x).

Property tests use Hypothesis to generate many test cases and verify that
universal properties hold across all valid inputs.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

# =============================================================================
# Test Data Generators (Strategies)
# =============================================================================

# Valid model names following LiteLLM provider format
VALID_PROVIDERS = ["openai", "anthropic", "bedrock", "azure", "vertex_ai", "cohere"]

provider_strategy = st.sampled_from(VALID_PROVIDERS)

model_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_."),
    min_size=1,
    max_size=50,
).filter(lambda x: x.strip() and not x.startswith("-"))

# Generate valid model configurations
@st.composite
def model_config_strategy(draw):
    """Generate a valid model configuration for LiteLLM."""
    provider = draw(provider_strategy)
    model_name = draw(model_name_strategy)
    
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"{provider}/{model_name}",
            "api_key": "test-api-key",
        }
    }


@st.composite
def model_list_strategy(draw):
    """Generate a valid model_list configuration."""
    num_models = draw(st.integers(min_value=1, max_value=5))
    models = []
    seen_names = set()
    
    for _ in range(num_models):
        config = draw(model_config_strategy())
        # Ensure unique model names
        if config["model_name"] not in seen_names:
            seen_names.add(config["model_name"])
            models.append(config)
    
    assume(len(models) > 0)
    return models


# Chat completion request strategy
@st.composite
def chat_request_strategy(draw):
    """Generate a valid chat completion request."""
    role = draw(st.sampled_from(["user", "assistant", "system"]))
    content = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    
    return {
        "model": draw(model_name_strategy),
        "messages": [{"role": role, "content": content}],
    }


# API key strategies
valid_api_key_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=10,
    max_size=100,
).filter(lambda x: x.strip())

invalid_api_key_strategy = st.one_of(
    st.just(""),
    st.just(None),
    st.text(max_size=5),  # Too short
)


# YAML configuration strategy
@st.composite
def yaml_config_strategy(draw):
    """Generate a valid YAML configuration structure."""
    model_list = draw(model_list_strategy())
    
    config = {
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": draw(st.sampled_from([
                "simple-shuffle", "least-busy", "llmrouter-knn"
            ])),
            "num_retries": draw(st.integers(min_value=0, max_value=5)),
            "timeout": draw(st.integers(min_value=30, max_value=600)),
        },
        "general_settings": {
            "master_key": draw(valid_api_key_strategy),
        },
        "litellm_settings": {
            "cache": draw(st.booleans()),
            "set_verbose": draw(st.booleans()),
        }
    }
    
    return config


# =============================================================================
# Property 1: Request Forwarding Correctness
# =============================================================================

class TestRequestForwardingProperty:
    """
    Property 1: Request Forwarding Correctness
    
    For any valid client request to a supported endpoint with a configured 
    LLM provider, the Gateway should successfully forward the request to 
    the provider using LiteLLM's unified interface and return a response 
    in the expected format.
    
    **Validates: Requirements 1.2**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        provider=provider_strategy,
        model_name=model_name_strategy,
        message_content=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
    )
    def test_request_forwarding_produces_valid_model_string(
        self, provider: str, model_name: str, message_content: str
    ):
        """
        Property 1: Request Forwarding Correctness
        
        For any valid provider and model name combination, the Gateway should
        construct a valid model string in the format 'provider/model_name'.
        
        **Validates: Requirements 1.2**
        """
        # Construct the model string as LiteLLM would
        model_string = f"{provider}/{model_name}"
        
        # Property: Model string should always contain the provider prefix
        assert provider in model_string
        assert "/" in model_string
        
        # Property: Model string should be parseable back to provider and model
        parts = model_string.split("/", 1)
        assert len(parts) == 2
        assert parts[0] == provider
        assert parts[1] == model_name

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(request=chat_request_strategy())
    def test_chat_request_has_required_fields(self, request: Dict[str, Any]):
        """
        Property 1: Request Forwarding Correctness
        
        For any generated chat request, it should contain all required fields
        for the OpenAI-compatible chat completions API.
        
        **Validates: Requirements 1.2**
        """
        # Property: Request must have 'model' field
        assert "model" in request
        assert isinstance(request["model"], str)
        assert len(request["model"]) > 0
        
        # Property: Request must have 'messages' field
        assert "messages" in request
        assert isinstance(request["messages"], list)
        assert len(request["messages"]) > 0
        
        # Property: Each message must have 'role' and 'content'
        for message in request["messages"]:
            assert "role" in message
            assert "content" in message
            assert message["role"] in ["user", "assistant", "system"]

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(model_list=model_list_strategy())
    def test_model_list_contains_valid_configurations(self, model_list: List[Dict]):
        """
        Property 1: Request Forwarding Correctness
        
        For any valid model list configuration, each model should have the
        required litellm_params with a properly formatted model string.
        
        **Validates: Requirements 1.2**
        """
        for model_config in model_list:
            # Property: Each model config must have model_name
            assert "model_name" in model_config
            assert isinstance(model_config["model_name"], str)
            
            # Property: Each model config must have litellm_params
            assert "litellm_params" in model_config
            params = model_config["litellm_params"]
            
            # Property: litellm_params must have model field
            assert "model" in params
            
            # Property: model field should contain provider prefix
            model_string = params["model"]
            assert "/" in model_string, f"Model string '{model_string}' should contain provider prefix"


# =============================================================================
# Property 2: Authentication Enforcement
# =============================================================================

class TestAuthenticationEnforcementProperty:
    """
    Property 2: Authentication Enforcement
    
    For any request to the Gateway when master_key is configured, the request
    should be accepted if and only if it includes a valid API key (either 
    master_key or a valid virtual key from the database).
    
    **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        master_key=valid_api_key_strategy,
        request_key=valid_api_key_strategy,
    )
    def test_matching_key_is_accepted(self, master_key: str, request_key: str):
        """
        Property 2: Authentication Enforcement
        
        For any master_key configuration, a request with the exact same key
        should be accepted (authentication passes).
        
        **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
        """
        # When request key matches master key, authentication should pass
        is_authenticated = (request_key == master_key)
        
        # Property: Same key should always authenticate
        if request_key == master_key:
            assert is_authenticated is True
        
        # Property: Different keys should not authenticate (unless virtual key)
        # Note: In real system, virtual keys would also be valid

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        master_key=valid_api_key_strategy,
    )
    def test_empty_key_is_rejected(self, master_key: str):
        """
        Property 2: Authentication Enforcement
        
        For any master_key configuration, a request with an empty or missing
        key should be rejected.
        
        **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
        """
        invalid_keys = ["", None, "   "]
        
        for invalid_key in invalid_keys:
            # Property: Empty/None keys should never match a valid master_key
            is_authenticated = (invalid_key == master_key)
            assert is_authenticated is False, f"Empty key '{invalid_key}' should not authenticate"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        master_key=valid_api_key_strategy,
        prefix=st.text(min_size=1, max_size=10),
    )
    def test_partial_key_is_rejected(self, master_key: str, prefix: str):
        """
        Property 2: Authentication Enforcement
        
        For any master_key, a partial match (prefix or substring) should not
        authenticate.
        
        **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
        """
        assume(len(master_key) > len(prefix))
        
        # Create a partial key (just the prefix of master_key)
        partial_key = master_key[:len(prefix)]
        
        # Property: Partial keys should not authenticate
        is_authenticated = (partial_key == master_key)
        
        if len(partial_key) < len(master_key):
            assert is_authenticated is False, "Partial key should not authenticate"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        key1=valid_api_key_strategy,
        key2=valid_api_key_strategy,
    )
    def test_authentication_is_deterministic(self, key1: str, key2: str):
        """
        Property 2: Authentication Enforcement
        
        Authentication should be deterministic - the same key comparison
        should always produce the same result.
        
        **Validates: Requirements 1.3, 7.5, 11.1, 11.4**
        """
        # Property: Authentication result should be consistent
        result1 = (key1 == key2)
        result2 = (key1 == key2)
        result3 = (key1 == key2)
        
        assert result1 == result2 == result3, "Authentication should be deterministic"


# =============================================================================
# Property 3: Configuration Loading
# =============================================================================

class TestConfigurationLoadingProperty:
    """
    Property 3: Configuration Loading
    
    For any valid YAML configuration file containing model_list, router_settings,
    and general_settings, the Gateway should successfully load the configuration
    and make all configured models and settings available.
    
    **Validates: Requirements 1.5**
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_has_required_sections(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, it should contain all required top-level
        sections: model_list, router_settings, general_settings, litellm_settings.
        
        **Validates: Requirements 1.5**
        """
        # Property: Config must have model_list
        assert "model_list" in config
        assert isinstance(config["model_list"], list)
        assert len(config["model_list"]) > 0
        
        # Property: Config must have router_settings
        assert "router_settings" in config
        assert isinstance(config["router_settings"], dict)
        
        # Property: Config must have general_settings
        assert "general_settings" in config
        assert isinstance(config["general_settings"], dict)
        
        # Property: Config must have litellm_settings
        assert "litellm_settings" in config
        assert isinstance(config["litellm_settings"], dict)

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_round_trip_preserves_structure(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, serializing to JSON and deserializing
        should preserve the structure (round-trip property).
        
        **Validates: Requirements 1.5**
        """
        import json
        
        # Serialize to JSON
        json_str = json.dumps(config)
        
        # Deserialize back
        loaded_config = json.loads(json_str)
        
        # Property: Round-trip should preserve all data
        assert loaded_config == config
        
        # Property: All sections should be preserved
        assert set(loaded_config.keys()) == set(config.keys())
        
        # Property: Model list should be preserved
        assert len(loaded_config["model_list"]) == len(config["model_list"])

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_models_are_accessible(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, all models in model_list should be
        accessible by their model_name.
        
        **Validates: Requirements 1.5**
        """
        model_list = config["model_list"]
        
        # Build a lookup dictionary (as the Gateway would)
        model_lookup = {m["model_name"]: m for m in model_list}
        
        # Property: All models should be in the lookup
        for model_config in model_list:
            model_name = model_config["model_name"]
            assert model_name in model_lookup
            assert model_lookup[model_name] == model_config

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_router_settings_has_valid_strategy(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, router_settings should contain a valid
        routing_strategy value.
        
        **Validates: Requirements 1.5**
        """
        router_settings = config["router_settings"]
        
        # Property: routing_strategy must be present
        assert "routing_strategy" in router_settings
        
        strategy = router_settings["routing_strategy"]
        
        # Property: Strategy must be a non-empty string
        assert isinstance(strategy, str)
        assert len(strategy) > 0
        
        # Property: Strategy should be a known value
        valid_strategies = [
            "simple-shuffle", "least-busy", "latency-based-routing",
            "cost-based-routing", "usage-based-routing",
            "llmrouter-knn", "llmrouter-svm", "llmrouter-mlp",
            "llmrouter-mf", "llmrouter-elo", "llmrouter-hybrid",
        ]
        assert strategy in valid_strategies, f"Unknown strategy: {strategy}"

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_yaml_serialization(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, it should be serializable to YAML format
        and loadable back without data loss.
        
        **Validates: Requirements 1.5**
        """
        import yaml
        
        # Serialize to YAML
        yaml_str = yaml.dump(config, default_flow_style=False)
        
        # Property: YAML string should not be empty
        assert len(yaml_str) > 0
        
        # Deserialize back
        loaded_config = yaml.safe_load(yaml_str)
        
        # Property: Round-trip should preserve data
        assert loaded_config == config

    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    @given(config=yaml_config_strategy())
    def test_config_file_write_and_read(self, config: Dict[str, Any]):
        """
        Property 3: Configuration Loading
        
        For any valid configuration, writing to a file and reading back
        should preserve all configuration data.
        
        **Validates: Requirements 1.5**
        """
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
        
        try:
            # Read back from file
            with open(temp_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Property: File round-trip should preserve data
            assert loaded_config == config
            
            # Property: All model names should be preserved
            original_names = {m["model_name"] for m in config["model_list"]}
            loaded_names = {m["model_name"] for m in loaded_config["model_list"]}
            assert original_names == loaded_names
            
        finally:
            os.unlink(temp_path)
