# Security policy

## Supported version

Security fixes are applied to the latest release and the `main` branch.

## Reporting

Please use GitHub's private vulnerability reporting for this repository. Do not open a
public issue containing secrets, sensitive corpus excerpts, or an exploitable proof of
concept. If private reporting is unavailable, contact the maintainer through the profile
contact method without attaching sensitive data.

## Data and local-service boundaries

The CLI reads local text and writes SQLite databases. Review corpus rights and sensitivity
before processing it, do not expose the optional local web interface to an untrusted
network, and do not open untrusted database files with elevated privileges.
