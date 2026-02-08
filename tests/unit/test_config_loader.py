"""
Unit tests for config_loader.py - S3/GCS Configuration Download Utilities
==========================================================================

Tests cover:
- download_config_from_s3: success, failure, directory creation
- download_config_from_gcs: success, failure, None contents
- download_model_from_s3: single file, directory prefix, errors
- download_custom_router_from_s3: delegation to download_model_from_s3
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.config_loader import (
    download_config_from_s3,
    download_config_from_gcs,
    download_model_from_s3,
    download_custom_router_from_s3,
)


def _mock_boto3():
    """Create a mock boto3 module and inject it into sys.modules."""
    mock_module = MagicMock()
    return mock_module


# =============================================================================
# S3 Config Download Tests
# =============================================================================


class TestDownloadConfigFromS3:
    """Tests for download_config_from_s3."""

    def test_successful_download(self, tmp_path):
        mock_boto3 = _mock_boto3()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        local_path = str(tmp_path / "config" / "config.yaml")
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = download_config_from_s3(
                "my-bucket", "path/to/config.yaml", local_path
            )

        assert result is True
        mock_boto3.client.assert_called_once_with("s3")
        mock_client.download_file.assert_called_once_with(
            "my-bucket", "path/to/config.yaml", local_path
        )

    def test_creates_parent_directory(self, tmp_path):
        mock_boto3 = _mock_boto3()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        local_path = str(tmp_path / "deep" / "nested" / "config.yaml")
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = download_config_from_s3("bucket", "key", local_path)

        assert result is True
        assert Path(local_path).parent.exists()

    def test_s3_error_returns_false(self):
        mock_boto3 = _mock_boto3()
        mock_client = MagicMock()
        mock_client.download_file.side_effect = Exception("NoSuchBucket")
        mock_boto3.client.return_value = mock_client

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = download_config_from_s3(
                "nonexistent", "key", "/tmp/test_config.yaml"
            )
        assert result is False


# =============================================================================
# GCS Config Download Tests
# =============================================================================


class TestDownloadConfigFromGCS:
    """Tests for download_config_from_gcs."""

    @staticmethod
    def _make_gcs_module(mock_gcs_instance):
        """Create a mock GCS module with GCSBucketLogger returning the instance."""
        mock_mod = MagicMock()
        mock_mod.GCSBucketLogger.return_value = mock_gcs_instance
        return mock_mod

    @pytest.mark.asyncio
    async def test_successful_download(self, tmp_path):
        local_path = str(tmp_path / "config.yaml")
        mock_gcs = MagicMock()
        mock_gcs.download_gcs_object = AsyncMock(return_value=b"model_list:\n  - gpt-4")

        mock_mod = self._make_gcs_module(mock_gcs)
        with patch.dict(
            sys.modules,
            {"litellm.integrations.gcs_bucket.gcs_bucket": mock_mod},
        ):
            result = await download_config_from_gcs(
                "my-bucket", "config.yaml", local_path
            )

        assert result is True
        assert Path(local_path).read_bytes() == b"model_list:\n  - gpt-4"

    @pytest.mark.asyncio
    async def test_none_contents_returns_false(self, tmp_path):
        local_path = str(tmp_path / "config.yaml")
        mock_gcs = MagicMock()
        mock_gcs.download_gcs_object = AsyncMock(return_value=None)

        mock_mod = self._make_gcs_module(mock_gcs)
        with patch.dict(
            sys.modules,
            {"litellm.integrations.gcs_bucket.gcs_bucket": mock_mod},
        ):
            result = await download_config_from_gcs(
                "bucket", "missing.yaml", local_path
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_gcs_error_returns_false(self, tmp_path):
        local_path = str(tmp_path / "config.yaml")
        mock_gcs = MagicMock()
        mock_gcs.download_gcs_object = AsyncMock(side_effect=Exception("GCS error"))

        mock_mod = self._make_gcs_module(mock_gcs)
        with patch.dict(
            sys.modules,
            {"litellm.integrations.gcs_bucket.gcs_bucket": mock_mod},
        ):
            result = await download_config_from_gcs("bucket", "key", local_path)

        assert result is False

    @pytest.mark.asyncio
    async def test_creates_parent_directory(self, tmp_path):
        local_path = str(tmp_path / "sub" / "dir" / "config.yaml")
        mock_gcs = MagicMock()
        mock_gcs.download_gcs_object = AsyncMock(return_value=b"data")

        mock_mod = self._make_gcs_module(mock_gcs)
        with patch.dict(
            sys.modules,
            {"litellm.integrations.gcs_bucket.gcs_bucket": mock_mod},
        ):
            result = await download_config_from_gcs("bucket", "key", local_path)

        assert result is True
        assert Path(local_path).parent.exists()


# =============================================================================
# S3 Model Download Tests
# =============================================================================


class TestDownloadModelFromS3:
    """Tests for download_model_from_s3."""

    def test_single_file_download(self, tmp_path):
        mock_boto3 = _mock_boto3()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        local_dir = str(tmp_path / "models")
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = download_model_from_s3(
                "bucket", "models/knn_router.pkl", local_dir
            )

        assert result is True
        mock_client.download_file.assert_called_once_with(
            "bucket",
            "models/knn_router.pkl",
            str(Path(local_dir) / "knn_router.pkl"),
        )

    def test_directory_prefix_download(self, tmp_path):
        mock_boto3 = _mock_boto3()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Simulate paginated listing
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/knn/model.pkl"},
                    {"Key": "models/knn/config.json"},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator

        local_dir = str(tmp_path / "models")
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = download_model_from_s3("bucket", "models/knn/", local_dir)

        assert result is True
        assert mock_client.download_file.call_count == 2

    def test_creates_local_directory(self, tmp_path):
        mock_boto3 = _mock_boto3()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        local_dir = str(tmp_path / "new" / "dir")
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = download_model_from_s3("bucket", "model.pkl", local_dir)

        assert result is True
        assert Path(local_dir).exists()

    def test_s3_error_returns_false(self):
        mock_boto3 = _mock_boto3()
        mock_client = MagicMock()
        mock_client.download_file.side_effect = Exception("AccessDenied")
        mock_boto3.client.return_value = mock_client

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = download_model_from_s3("bucket", "model.pkl", "/tmp/models_test")
        assert result is False


# =============================================================================
# Custom Router Download Tests
# =============================================================================


class TestDownloadCustomRouterFromS3:
    """Tests for download_custom_router_from_s3."""

    def test_delegates_to_download_model(self):
        mock_boto3 = _mock_boto3()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = download_custom_router_from_s3(
                "bucket", "routers/custom.py", "/tmp/routers_test"
            )
        assert result is True
