import os
import logging
from typing import Optional
from google.cloud import storage
from google.api_core import exceptions

logger = logging.getLogger(__name__)

def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a GCS URI into bucket name and blob name.

    Args:
        gcs_uri: GCS URI in format gs://bucket-name/path/to/blob

    Returns:
        Tuple of (bucket_name, blob_name)

    Raises:
        ValueError: If URI format is invalid
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}")

    # Remove gs:// prefix and split
    path = gcs_uri[5:]  # Remove 'gs://'
    parts = path.split("/", 1)

    if len(parts) < 2:
        raise ValueError(f"Invalid GCS URI format, missing blob name: {gcs_uri}")

    bucket_name = parts[0]
    blob_name = parts[1]

    return bucket_name, blob_name

def download_from_gcs(gcs_uri: str, local_path: str) -> None:
    """Download file from GCS to local path.

    Args:
        gcs_uri: GCS URI (gs://bucket/path/to/file)
        local_path: Local file path to save to

    Raises:
        FileNotFoundError: If the GCS object doesn't exist
        PermissionError: If no access to the bucket/object
        Exception: For other GCS-related errors
    """
    try:
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)

        logger.info(f"Downloading from GCS: {gcs_uri} -> {local_path}")

        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if blob exists
        if not blob.exists():
            raise FileNotFoundError(f"GCS object not found: {gcs_uri}")

        # Download the file
        blob.download_to_filename(local_path)

        # Verify download
        if not os.path.exists(local_path):
            raise Exception(f"Download failed: {local_path} not created")

        file_size = os.path.getsize(local_path)
        logger.info(f"Successfully downloaded {file_size} bytes to {local_path}")

    except exceptions.NotFound:
        raise FileNotFoundError(f"GCS object not found: {gcs_uri}")
    except exceptions.Forbidden:
        raise PermissionError(f"Access denied to GCS object: {gcs_uri}")
    except exceptions.Unauthorized:
        raise PermissionError(f"Authentication failed for GCS access")
    except Exception as e:
        logger.error(f"Failed to download from GCS {gcs_uri}: {str(e)}")
        raise

def upload_to_gcs(local_path: str, gcs_uri: str, content_type: Optional[str] = None) -> None:
    """Upload file from local path to GCS.

    Args:
        local_path: Local file path to upload
        gcs_uri: Target GCS URI (gs://bucket/path/to/file)
        content_type: Optional content type to set

    Raises:
        FileNotFoundError: If local file doesn't exist
        PermissionError: If no access to write to bucket
        Exception: For other GCS-related errors
    """
    try:
        # Check if local file exists
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        bucket_name, blob_name = parse_gcs_uri(gcs_uri)

        logger.info(f"Uploading to GCS: {local_path} -> {gcs_uri}")

        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Set content type if provided
        if content_type:
            blob.content_type = content_type

        # Upload the file
        blob.upload_from_filename(local_path)

        file_size = os.path.getsize(local_path)
        logger.info(f"Successfully uploaded {file_size} bytes to {gcs_uri}")

    except exceptions.Forbidden:
        raise PermissionError(f"Access denied to GCS bucket: {bucket_name}")
    except exceptions.Unauthorized:
        raise PermissionError(f"Authentication failed for GCS access")
    except Exception as e:
        logger.error(f"Failed to upload to GCS {gcs_uri}: {str(e)}")
        raise

def list_gcs_objects(bucket_name: str, prefix: Optional[str] = None, max_results: int = 1000) -> list[str]:
    """List objects in a GCS bucket with optional prefix filter.

    Args:
        bucket_name: GCS bucket name
        prefix: Optional prefix to filter objects
        max_results: Maximum number of results to return

    Returns:
        List of GCS URIs for matching objects

    Raises:
        PermissionError: If no access to the bucket
        Exception: For other GCS-related errors
    """
    try:
        logger.info(f"Listing objects in bucket: {bucket_name}")

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # List blobs with optional prefix
        blobs = client.list_blobs(
            bucket_name,
            prefix=prefix,
            max_results=max_results
        )

        # Convert to GCS URIs
        uris = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]

        logger.info(f"Found {len(uris)} objects")
        return uris

    except exceptions.NotFound:
        raise FileNotFoundError(f"Bucket not found: {bucket_name}")
    except exceptions.Forbidden:
        raise PermissionError(f"Access denied to bucket: {bucket_name}")
    except exceptions.Unauthorized:
        raise PermissionError(f"Authentication failed for GCS access")
    except Exception as e:
        logger.error(f"Failed to list objects in bucket {bucket_name}: {str(e)}")
        raise

def check_gcs_credentials() -> bool:
    """Check if GCS credentials are properly configured.

    Returns:
        True if credentials are valid, False otherwise
    """
    try:
        client = storage.Client()
        # Try to list buckets to verify credentials
        list(client.list_buckets(max_results=1))
        logger.info("GCS credentials verified successfully")
        return True
    except Exception as e:
        logger.warning(f"GCS credentials check failed: {str(e)}")
        return False