"""Custom exceptions for the application."""


class AppError(Exception):
    """Base application exception."""
    pass


class ModelLoadError(AppError):
    """Raised when model loading fails."""
    pass


class PredictionError(AppError):
    """Raised when prediction fails."""
    pass


class ImageProcessingError(AppError):
    """Raised when image preprocessing fails."""
    pass


class InvalidImageError(AppError):
    """Raised when uploaded image is invalid."""
    pass


class DataLoadError(AppError):
    """Raised when data loading fails."""
    pass


class ConfigurationError(AppError):
    """Raised when configuration is invalid."""
    pass