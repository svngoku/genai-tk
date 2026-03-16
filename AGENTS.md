# Development Guidelines for GenAI Toolkit

## Build/Lint/Test Commands

**Package Management:** Uses `uv` for dependency management
- `make install-dev` - Install with development dependencies
- `make fmt` - Format code with ruff (includes import sorting)
- `make lint` - Lint code with ruff
- `make test` - Run all tests (unit + integration)
- `make test-unit` - Run unit tests only
- `make test-integration` - Run integration tests only
- `make check` - Run format, lint, and test sequentially

## Code and tools execution ##
- Use 'uv' to run Python code, execute tests, install packages etc.

## Code Style Guidelines

**Formatting & Linting:**
- Line length: 120 characters
- Use ruff for formatting and linting
- Import sorting: isort rules (handled by ruff)

## Documentation**
- Rules for doctrings:
    - Use Google style 
    - Don't mention types 
    - Don't mention raised exceptions 
    - Use  ``` to format code in documentation (Fenced Code Blocks)
   -  Don't repeat code examples in module description if they are already in function docstrings. 
- Don't add comments for classes fields and arguments. Write only one line to describe the clas

**Type System:**
- Python 3.12+ required
- Use type hints extensively (pydantic models preferred)
- Return types encouraged but not strictly required (ANN201/ANN202 ignored). 
- Use 'None" type annotation when functions return nothing
- Avoid 'Any' type except if its necessary.
- Use Python 12 capabilities - Typically use '|' instead of 'Union', and '| None' instead of 'Optional'
- Use Pydantic to define class whenener possible. Avoid __init__ methods, use model_post_init() instead.

**Naming Conventions:**
- Classes: PascalCase (e.g., `LlmFactory`, `OmegaConfig`)
- Functions/variables: snake_case (e.g., `get_llm`, `llm_config`)
- Constants: UPPER_SNAKE_CASE (e.g., `APPLICATION_CONFIG_FILE`)
- Private members: underscore prefix (e.g., `_internal_method`)

**Import Organization:**
- Prefer explicit imports over star imports
- Alwary use absolute  imports for project modules.   Don't use relative imports

**Error Handling:**
- Use structured logging with loguru
- Raise specific exceptions with descriptive messages
- Use pydantic for data validation and error handling
- Implement proper error boundaries in agent workflows

**Testing:**
- Use pytest with asyncio support
- Test files: `test_*.py` or `*_test.py`
- Place unit tests in `tests/unit_tests/`
- Place integration tests in `tests/integration_tests/`
- Use faker for test data generation

**Configuration:**
- YAML-based configuration with OmegaConf
- Environment variable substitution supported
- Singleton pattern for global config access
- Separate configs for different environments (dev, prod, etc.)

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
