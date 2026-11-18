# Security policy

## Supported versions

Only the latest commit on `main` is maintained.

## Reporting a vulnerability

Please report security problems privately through GitHub: open the repository's Security tab and choose "Report a vulnerability". Do not open a public issue for a security problem.

Include what you found, the steps to reproduce it and the commit you tested. We aim to reply within seven days.

## Scope and known limits

PathSnap is a course project built for a local two-laptop demo. Keep these limits in mind before running it anywhere else:

- The API has no authentication and no HTTPS. Run it only on a trusted local network, such as a phone hotspot.
- `POST /predict` accepts only trip ids that appear in `GET /trips`, so it cannot be pointed at arbitrary files.
- Model files are loaded with joblib, which can run code. Load only the models in this repository or models you trained yourself.
- The repository holds no passwords, API keys or other secrets. Never commit any.
