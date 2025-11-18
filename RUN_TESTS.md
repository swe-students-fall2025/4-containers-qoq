# How to Run Tests

## Web App Tests

### Option 1: Local Python Environment

1. **Install dependencies:**
   ```bash
   cd web-app
   pip install -r requirements.txt
   ```

2. **Run tests:**
   ```bash
   pytest
   ```

3. **Run with coverage:**
   ```bash
   pytest --cov=. --cov-report=term-missing
   ```

### Option 2: Using Docker

```bash
cd web-app
docker build -t web-app-test .
docker run --rm web-app-test pytest --cov=. --cov-report=term
```

### Option 3: Using Docker Compose

```bash
docker compose run --rm web-app pytest --cov=. --cov-report=term
```

## Quick Test Commands

```bash
# Run all tests
cd web-app && pytest

# Run with verbose output
cd web-app && pytest -v

# Run specific test
cd web-app && pytest test_app.py::test_index

# Run with coverage (must be >= 80%)
cd web-app && pytest --cov=. --cov-report=term --cov-fail-under=80
```

## Test Output

When tests pass, you'll see:
```
============================= test session starts ==============================
collected 25 items

test_app.py ......................... [100%]

============================= 25 passed in X.XXs ==============================
```

## Troubleshooting

If you get import errors:
- Make sure you're in the `web-app` directory
- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.10+)

If tests fail:
- Check that MongoDB mocking is working (tests don't need real MongoDB)
- Verify MediaPipe model file exists (tests mock this too)
