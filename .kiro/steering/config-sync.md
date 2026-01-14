---
inclusion: fileMatch
fileMatchPattern: "**/config_sync.py"
---

# Config Sync Reference

## Overview

The `ConfigSyncManager` provides background synchronization of configuration from S3/GCS with ETag-based change detection to minimize bandwidth.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_S3_BUCKET` | - | S3 bucket for config sync |
| `CONFIG_S3_KEY` | - | S3 key path to config file |
| `CONFIG_GCS_BUCKET` | - | GCS bucket for config sync |
| `CONFIG_GCS_KEY` | - | GCS key path to config file |
| `CONFIG_HOT_RELOAD` | `false` | Enable hot reload on config change |
| `CONFIG_SYNC_ENABLED` | `true` | Enable background sync |
| `CONFIG_SYNC_INTERVAL` | `60` | Sync interval in seconds |

## ETag-Based Change Detection

The sync manager uses ETags to avoid unnecessary downloads:

```python
def _download_from_s3_if_changed(self) -> bool:
    current_etag = self._get_s3_etag()
    
    # Skip download if ETag hasn't changed
    if current_etag == self._last_s3_etag:
        return False
    
    # Download and update cached ETag
    s3_client.download_file(bucket, key, local_path)
    self._last_s3_etag = current_etag
    return True
```

## Hot Reload Trigger

When config changes are detected:
1. File is downloaded from S3/GCS
2. MD5 hash is compared to detect actual content changes
3. If changed and `hot_reload_enabled`, triggers reload via:
   - Custom callback (`on_config_changed`)
   - Or SIGHUP signal to LiteLLM

```python
def _trigger_reload(self):
    if self.on_config_changed:
        self.on_config_changed()
    else:
        os.kill(os.getpid(), signal.SIGHUP)
```

## Singleton Pattern

Use the global singleton for consistent state:

```python
from litellm_llmrouter.config_sync import get_sync_manager

manager = get_sync_manager()
manager.start()  # Start background sync

# Force immediate sync
manager.force_sync()

# Get status
status = manager.get_status()
```

## Status Response

```python
{
    "enabled": True,
    "hot_reload_enabled": True,
    "sync_interval_seconds": 60,
    "s3": {
        "enabled": True,
        "bucket": "my-bucket",
        "key": "config/config.yaml",
        "last_etag": "abc123..."
    },
    "local_config_path": "/app/config/config.yaml",
    "local_config_hash": "def456...",
    "reload_count": 5,
    "last_sync_time": 1705142400.0,
    "running": True
}
```

## Thread Safety

The sync loop runs in a daemon thread with proper shutdown:

```python
def start(self):
    self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
    self._sync_thread.start()

def stop(self):
    self._stop_event.set()
    if self._sync_thread and self._sync_thread.is_alive():
        self._sync_thread.join(timeout=5)
```

## AWS Credentials

For S3 sync, ensure AWS credentials are available via:
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- IAM role (ECS task role, EC2 instance profile)
- AWS credentials file

## Adding GCS Support

GCS sync follows the same pattern but uses `google-cloud-storage`:

```python
from google.cloud import storage

def _get_gcs_etag(self) -> str | None:
    client = storage.Client()
    bucket = client.bucket(self.gcs_bucket)
    blob = bucket.blob(self.gcs_key)
    blob.reload()
    return blob.etag
```
