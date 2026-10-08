# Architecture Rationale

The repository is organized by engineering responsibility rather than by development day or experiment date.

- `vehicles`: physical plant models
- `guidance`: mission-level references
- `control`: feedback laws
- `estimation`: sensors and state estimation
- `simulation`: numerical propagation and loop orchestration
- `evaluation`: metrics
- `disturbances`: environmental inputs

This keeps vehicle-specific physics isolated from reusable GNC logic and makes the repository extensible to additional vehicle classes.
