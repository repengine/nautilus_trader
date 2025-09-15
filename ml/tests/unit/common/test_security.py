#!/usr/bin/env python3

"""
Tests for ML model artifact security and integrity verification.

These tests ensure that the SHA-256 integrity verification system correctly
detects tampered model artifacts and prevents loading of compromised models.
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ml.common.security import ArtifactIntegrityError
from ml.common.security import calculate_file_sha256
from ml.common.security import secure_onnx_load
from ml.common.security import verify_artifact_integrity


class TestCalculateFileSha256:
    """Test SHA-256 calculation for files."""

    def test_calculate_file_sha256_valid_file(self, tmp_path: Path) -> None:
        """Test SHA-256 calculation for a valid file."""
        # Create a test file with known content
        test_file = tmp_path / "test.bin"
        test_content = b"This is test content for SHA-256 calculation"
        test_file.write_bytes(test_content)

        # Calculate expected digest
        expected_digest = hashlib.sha256(test_content).hexdigest()

        # Test function
        actual_digest = calculate_file_sha256(test_file)

        assert actual_digest == expected_digest
        assert len(actual_digest) == 64  # SHA-256 hex string length

    def test_calculate_file_sha256_large_file(self, tmp_path: Path) -> None:
        """Test SHA-256 calculation for a large file (tests chunked reading)."""
        # Create a large test file (>8KB to test chunked reading)
        test_file = tmp_path / "large_test.bin"
        test_content = b"A" * 10000  # 10KB of 'A' characters
        test_file.write_bytes(test_content)

        # Calculate expected digest
        expected_digest = hashlib.sha256(test_content).hexdigest()

        # Test function
        actual_digest = calculate_file_sha256(test_file)

        assert actual_digest == expected_digest

    def test_calculate_file_sha256_empty_file(self, tmp_path: Path) -> None:
        """Test SHA-256 calculation for an empty file."""
        # Create an empty test file
        test_file = tmp_path / "empty.bin"
        test_file.touch()

        # Calculate expected digest for empty content
        expected_digest = hashlib.sha256(b"").hexdigest()

        # Test function
        actual_digest = calculate_file_sha256(test_file)

        assert actual_digest == expected_digest

    def test_calculate_file_sha256_nonexistent_file(self, tmp_path: Path) -> None:
        """Test SHA-256 calculation raises error for nonexistent file."""
        nonexistent_file = tmp_path / "nonexistent.bin"

        with pytest.raises(FileNotFoundError, match="File not found"):
            calculate_file_sha256(nonexistent_file)

    def test_calculate_file_sha256_permission_error(self, tmp_path: Path) -> None:
        """Test SHA-256 calculation handles permission errors."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test content")

        # Mock open to raise permission error
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(IOError, match="Failed to read file"):
                calculate_file_sha256(test_file)


class TestVerifyArtifactIntegrity:
    """Test artifact integrity verification."""

    def test_verify_artifact_integrity_valid(self, tmp_path: Path) -> None:
        """Test integrity verification passes for valid artifact."""
        # Create test file and calculate digest
        test_file = tmp_path / "model.onnx"
        test_content = b"Valid ONNX model content"
        test_file.write_bytes(test_content)
        expected_digest = hashlib.sha256(test_content).hexdigest()

        # Test verification passes
        result = verify_artifact_integrity(test_file, expected_digest, strict=True)
        assert result is True

        # Test verification passes in non-strict mode too
        result = verify_artifact_integrity(test_file, expected_digest, strict=False)
        assert result is True

    def test_verify_artifact_integrity_tampered_strict(self, tmp_path: Path) -> None:
        """Test integrity verification fails for tampered artifact in strict mode."""
        # Create test file
        test_file = tmp_path / "model.onnx"
        original_content = b"Original ONNX model content"
        test_file.write_bytes(original_content)
        expected_digest = hashlib.sha256(original_content).hexdigest()

        # Tamper with the file
        tampered_content = b"Tampered ONNX model content"
        test_file.write_bytes(tampered_content)

        # Test verification fails in strict mode
        with pytest.raises(ArtifactIntegrityError) as exc_info:
            verify_artifact_integrity(test_file, expected_digest, strict=True)

        error = exc_info.value
        assert "integrity verification failed" in str(error).lower()
        assert "tampered" in str(error).lower()
        assert error.expected_digest == expected_digest
        assert error.actual_digest == hashlib.sha256(tampered_content).hexdigest()

    def test_verify_artifact_integrity_tampered_non_strict(self, tmp_path: Path) -> None:
        """Test integrity verification fails gracefully for tampered artifact in non-strict mode."""
        # Create test file
        test_file = tmp_path / "model.onnx"
        original_content = b"Original ONNX model content"
        test_file.write_bytes(original_content)
        expected_digest = hashlib.sha256(original_content).hexdigest()

        # Tamper with the file
        tampered_content = b"Tampered ONNX model content"
        test_file.write_bytes(tampered_content)

        # Test verification returns False in non-strict mode
        result = verify_artifact_integrity(test_file, expected_digest, strict=False)
        assert result is False

    def test_verify_artifact_integrity_no_digest_strict(self, tmp_path: Path) -> None:
        """Test integrity verification with no digest in strict mode."""
        test_file = tmp_path / "model.onnx"
        test_file.write_bytes(b"Some content")

        # Test with None digest - should pass with warning
        result = verify_artifact_integrity(test_file, None, strict=True)
        assert result is True

        # Test with empty digest - should pass with warning
        result = verify_artifact_integrity(test_file, "", strict=True)
        assert result is True

    def test_verify_artifact_integrity_no_digest_non_strict(self, tmp_path: Path) -> None:
        """Test integrity verification with no digest in non-strict mode."""
        test_file = tmp_path / "model.onnx"
        test_file.write_bytes(b"Some content")

        # Test with None digest - should pass
        result = verify_artifact_integrity(test_file, None, strict=False)
        assert result is True

        # Test with empty digest - should pass
        result = verify_artifact_integrity(test_file, "", strict=False)
        assert result is True

    def test_verify_artifact_integrity_file_not_found(self, tmp_path: Path) -> None:
        """Test integrity verification with nonexistent file."""
        nonexistent_file = tmp_path / "nonexistent.onnx"
        digest = "abc123"

        with pytest.raises(ValueError, match="Cannot verify artifact integrity"):
            verify_artifact_integrity(nonexistent_file, digest, strict=True)

        with pytest.raises(ValueError, match="Cannot verify artifact integrity"):
            verify_artifact_integrity(nonexistent_file, digest, strict=False)


class TestSecureOnnxLoad:
    """Test secure ONNX model loading with integrity verification."""

    def test_secure_onnx_load_valid_model(self, tmp_path: Path) -> None:
        """Test secure loading of valid ONNX model."""
        # Create a minimal ONNX file (not a real model, just for testing the security layer)
        test_file = tmp_path / "model.onnx"
        test_content = b"fake onnx model content"
        test_file.write_bytes(test_content)
        expected_digest = hashlib.sha256(test_content).hexdigest()

        # Mock ONNX imports and InferenceSession
        with patch("ml.common.security.HAS_ONNX", True), \
             patch("ml.common.security.ort") as mock_ort:

            mock_session = mock_ort.InferenceSession.return_value

            # Test secure loading
            result = secure_onnx_load(
                file_path=test_file,
                expected_digest=expected_digest,
                strict_integrity=True
            )

            assert result == mock_session
            mock_ort.InferenceSession.assert_called_once_with(str(test_file))

    def test_secure_onnx_load_tampered_model_strict(self, tmp_path: Path) -> None:
        """Test secure loading rejects tampered ONNX model in strict mode."""
        # Create test file
        test_file = tmp_path / "model.onnx"
        original_content = b"original onnx model content"
        test_file.write_bytes(original_content)
        expected_digest = hashlib.sha256(original_content).hexdigest()

        # Tamper with the file
        tampered_content = b"tampered onnx model content"
        test_file.write_bytes(tampered_content)

        # Test secure loading rejects tampered model
        with pytest.raises(ArtifactIntegrityError):
            secure_onnx_load(
                file_path=test_file,
                expected_digest=expected_digest,
                strict_integrity=True
            )

    def test_secure_onnx_load_tampered_model_non_strict(self, tmp_path: Path) -> None:
        """Test secure loading handles tampered ONNX model in non-strict mode."""
        # Create test file
        test_file = tmp_path / "model.onnx"
        original_content = b"original onnx model content"
        test_file.write_bytes(original_content)
        expected_digest = hashlib.sha256(original_content).hexdigest()

        # Tamper with the file
        tampered_content = b"tampered onnx model content"
        test_file.write_bytes(tampered_content)

        # Mock ONNX imports
        with patch("ml.common.security.HAS_ONNX", True), \
             patch("ml.common.security.ort") as mock_ort:

            mock_session = mock_ort.InferenceSession.return_value

            # Test secure loading proceeds with warning in non-strict mode
            result = secure_onnx_load(
                file_path=test_file,
                expected_digest=expected_digest,
                strict_integrity=False
            )

            # Should still load the model despite integrity failure in non-strict mode
            assert result == mock_session

    def test_secure_onnx_load_no_digest(self, tmp_path: Path) -> None:
        """Test secure loading with no digest provided."""
        test_file = tmp_path / "model.onnx"
        test_file.write_bytes(b"onnx model content")

        # Mock ONNX imports
        with patch("ml.common.security.HAS_ONNX", True), \
             patch("ml.common.security.ort") as mock_ort:

            mock_session = mock_ort.InferenceSession.return_value

            # Test loading without digest
            result = secure_onnx_load(
                file_path=test_file,
                expected_digest=None,
                strict_integrity=True
            )

            assert result == mock_session

    def test_secure_onnx_load_with_session_options(self, tmp_path: Path) -> None:
        """Test secure loading with custom session options."""
        test_file = tmp_path / "model.onnx"
        test_content = b"onnx model content"
        test_file.write_bytes(test_content)
        expected_digest = hashlib.sha256(test_content).hexdigest()

        # Mock ONNX imports and session options
        with patch("ml.common.security.HAS_ONNX", True), \
             patch("ml.common.security.ort") as mock_ort:

            mock_session = mock_ort.InferenceSession.return_value
            mock_session_options = "mock_session_options"
            mock_providers = ["CPUExecutionProvider"]

            # Test loading with custom options
            result = secure_onnx_load(
                file_path=test_file,
                expected_digest=expected_digest,
                session_options=mock_session_options,
                providers=mock_providers,
                strict_integrity=True
            )

            assert result == mock_session
            mock_ort.InferenceSession.assert_called_once_with(
                str(test_file),
                sess_options=mock_session_options,
                providers=mock_providers
            )

    def test_secure_onnx_load_onnx_not_available(self, tmp_path: Path) -> None:
        """Test secure loading when ONNX is not available."""
        test_file = tmp_path / "model.onnx"
        test_file.write_bytes(b"onnx model content")

        # Mock ONNX not available
        with patch("ml.common.security.HAS_ONNX", False), \
             patch("ml.common.security.check_ml_dependencies") as mock_check:

            mock_check.side_effect = ImportError("ONNX not available")

            with pytest.raises(ImportError, match="ONNX Runtime not available"):
                secure_onnx_load(
                    file_path=test_file,
                    expected_digest=None,
                    strict_integrity=False
                )

    def test_secure_onnx_load_model_loading_error(self, tmp_path: Path) -> None:
        """Test secure loading when ONNX model loading fails."""
        test_file = tmp_path / "model.onnx"
        test_content = b"invalid onnx content"
        test_file.write_bytes(test_content)
        expected_digest = hashlib.sha256(test_content).hexdigest()

        # Mock ONNX imports but make InferenceSession fail
        with patch("ml.common.security.HAS_ONNX", True), \
             patch("ml.common.security.ort") as mock_ort:

            mock_ort.InferenceSession.side_effect = RuntimeError("Invalid ONNX model")

            with pytest.raises(ValueError, match="Failed to load ONNX model"):
                secure_onnx_load(
                    file_path=test_file,
                    expected_digest=expected_digest,
                    strict_integrity=True
                )


class TestArtifactIntegrityError:
    """Test custom exception for artifact integrity failures."""

    def test_artifact_integrity_error_basic(self) -> None:
        """Test basic ArtifactIntegrityError creation."""
        error = ArtifactIntegrityError("Test error message")
        assert str(error) == "Test error message"
        assert error.expected_digest is None
        assert error.actual_digest is None

    def test_artifact_integrity_error_with_digests(self) -> None:
        """Test ArtifactIntegrityError with digest information."""
        expected = "abc123"
        actual = "def456"
        error = ArtifactIntegrityError(
            "Integrity check failed",
            expected_digest=expected,
            actual_digest=actual
        )

        assert str(error) == "Integrity check failed"
        assert error.expected_digest == expected
        assert error.actual_digest == actual


class TestIntegrationScenarios:
    """Integration tests for realistic security scenarios."""

    def test_model_tampering_detection_scenario(self, tmp_path: Path) -> None:
        """Test end-to-end scenario of detecting model tampering."""
        # Step 1: Create a "legitimate" model file
        model_file = tmp_path / "production_model.onnx"
        legitimate_content = b"legitimate model weights and structure"
        model_file.write_bytes(legitimate_content)

        # Step 2: Calculate and store the legitimate digest
        legitimate_digest = calculate_file_sha256(model_file)

        # Step 3: Simulate an attacker tampering with the model
        malicious_content = b"malicious model with backdoor inserted"
        model_file.write_bytes(malicious_content)

        # Step 4: Verify that integrity check detects the tampering
        with pytest.raises(ArtifactIntegrityError) as exc_info:
            verify_artifact_integrity(model_file, legitimate_digest, strict=True)

        # Step 5: Verify the error contains security information
        error = exc_info.value
        assert error.expected_digest == legitimate_digest
        assert error.actual_digest == calculate_file_sha256(model_file)
        assert error.expected_digest != error.actual_digest
        assert "tampered" in str(error).lower()

    def test_legitimate_model_loading_scenario(self, tmp_path: Path) -> None:
        """Test end-to-end scenario of loading a legitimate model."""
        # Step 1: Create a legitimate model file
        model_file = tmp_path / "production_model.onnx"
        legitimate_content = b"legitimate model weights and structure"
        model_file.write_bytes(legitimate_content)

        # Step 2: Calculate the digest
        legitimate_digest = calculate_file_sha256(model_file)

        # Step 3: Verify that integrity check passes
        result = verify_artifact_integrity(model_file, legitimate_digest, strict=True)
        assert result is True

        # Step 4: Verify that secure loading would work
        with patch("ml.common.security.HAS_ONNX", True), \
             patch("ml.common.security.ort") as mock_ort:

            mock_session = mock_ort.InferenceSession.return_value

            loaded_model = secure_onnx_load(
                file_path=model_file,
                expected_digest=legitimate_digest,
                strict_integrity=True
            )

            assert loaded_model == mock_session

    def test_multiple_file_tampering_detection(self, tmp_path: Path) -> None:
        """Test detection of tampering across multiple model files."""
        # Create multiple model files
        models = {}
        digests = {}

        for i in range(3):
            model_file = tmp_path / f"model_{i}.onnx"
            content = f"model {i} content".encode()
            model_file.write_bytes(content)
            models[f"model_{i}"] = model_file
            digests[f"model_{i}"] = calculate_file_sha256(model_file)

        # Tamper with one model
        tampered_content = b"tampered model 1 content"
        models["model_1"].write_bytes(tampered_content)

        # Verify that only the tampered model fails verification
        assert verify_artifact_integrity(models["model_0"], digests["model_0"], strict=True)

        with pytest.raises(ArtifactIntegrityError):
            verify_artifact_integrity(models["model_1"], digests["model_1"], strict=True)

        assert verify_artifact_integrity(models["model_2"], digests["model_2"], strict=True)
