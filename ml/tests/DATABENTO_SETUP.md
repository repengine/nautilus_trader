# Databento API Configuration

## Security Setup

The Databento API key is configured securely with the following measures:

### 1. Environment Variable Configuration

- API key is stored in `.env.local` (NOT tracked by git)
- Automatically loaded when activating virtual environment
- Never hardcoded in any Python files

### 2. Git Security
The following files are in `.gitignore` to prevent accidental exposure:

- `*.env`
- `.env.local`
- `.env.*.local`
- `*.key`

### 3. Automatic Loading
When you activate the virtual environment:

```bash
source .venv/bin/activate
```

The `.env.local` file is automatically sourced, setting:

- `DATABENTO_API_KEY` environment variable

### 4. Testing the Setup
Run the test script to verify configuration:

```bash
python ml/tests/test_databento_setup.py
```

### 5. Using in Code

```python
import os
import databento as db

# API key is automatically available from environment
client = db.Historical(os.environ["DATABENTO_API_KEY"])

# Or use Nautilus adapter
from nautilus_trader.adapters.databento import DatabentoDataClientConfig

config = DatabentoDataClientConfig(
    api_key=os.environ["DATABENTO_API_KEY"],  # Loaded from environment
    # other config...
)
```

## Important Security Notes

1. **NEVER** commit `.env.local` to git
2. **NEVER** hardcode the API key in source files
3. **ALWAYS** use environment variables for sensitive data
4. **ALWAYS** mask/truncate keys when displaying them
5. **ROTATE** API keys periodically for security

## Troubleshooting

If the API key is not loading:

1. Ensure you've activated the virtual environment
2. Check that `.env.local` exists in the project root
3. Verify the file has the correct export statement
4. Run `echo $DATABENTO_API_KEY` to check if it's set

## CI/CD Configuration

For GitHub Actions or other CI/CD:

1. Add `DATABENTO_API_KEY` as a repository secret
2. Never echo or log the full key
3. Use masked outputs in CI logs
