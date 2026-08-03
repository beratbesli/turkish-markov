"""Project-specific exceptions that can be shown directly by the CLI."""


class MarkovError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(MarkovError):
    """Raised for an invalid build or generation configuration."""


class CorpusError(MarkovError):
    """Raised when corpus discovery or decoding fails."""


class DatabaseError(MarkovError):
    """Raised when a database is missing, incompatible, or incomplete."""


class GenerationError(MarkovError):
    """Raised when generation cannot continue from a supplied prompt."""
