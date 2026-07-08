# Project Rules & Customizations

## Docker Build Guidelines

- **IS_LOCAL Variable Rule**: 
  - Locally, the `.env` file should have `IS_LOCAL=true` so that local runs bypass Neon DB logging.
  - **CRITICAL**: Whenever building the Docker container image (`docker buildx build`), you must temporarily change `IS_LOCAL=false` in the `.env` file, compile the image, and then restore `IS_LOCAL=true` in the `.env` file afterward. This ensures database logging runs correctly inside the production grading container environment.
