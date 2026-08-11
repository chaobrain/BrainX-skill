- [Installation](https://brainx.chaobrain.com/brainunit/getting_started/installation.html) — Installs BrainUnit and configures its NumPy, JAX, and optional array backends.
- [Quickstart Guide](https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html) — Introduces quantities, arithmetic, conversions, unit-aware functions, JAX transformations, and unit validation.

- [Quantity](https://brainx.chaobrain.com/brainunit/physical_units/quantity.html) — Explains numerical values carrying physical units and the core behavior of `Quantity`.
- [Math Operations](https://brainx.chaobrain.com/brainunit/physical_units/math_operations.html) — Explains arithmetic rules, dimensional compatibility, and unit propagation.
- [Standard Units](https://brainx.chaobrain.com/brainunit/physical_units/standard_units.html) — Lists predefined SI and commonly used physical units.
- [Constants](https://brainx.chaobrain.com/brainunit/physical_units/constants.html) — Provides predefined physical constants as unit-aware quantities.
- [Unit Conversion](https://brainx.chaobrain.com/brainunit/physical_units/conversion.html) — Converts quantities between compatible physical units.
- [Temperature Conversions](https://brainx.chaobrain.com/brainunit/physical_units/temperature.html) — Converts temperatures between scales with offsets and scaling.

- [Array Creation](https://brainx.chaobrain.com/brainunit/unit_operations/array_creation.html) — Creates unit-aware arrays with NumPy- and JAX-style constructors.
- [NumPy Functions](https://brainx.chaobrain.com/brainunit/unit_operations/numpy_functions.html) — Applies common array and mathematical operations while preserving units.
- [Einstein Operations](https://brainx.chaobrain.com/brainunit/unit_operations/einstein_operations.html) — Performs unit-aware Einstein summation, rearrangement, reduction, and repetition.
- [Linear Algebra Functions](https://brainx.chaobrain.com/brainunit/unit_operations/linalg_functions.html) — Applies matrix decompositions, solvers, norms, and related operations with units.
- [Fourier Transform Functions](https://brainx.chaobrain.com/brainunit/unit_operations/fft_functions.html) — Applies Fourier transforms while tracking resulting units.
- [JAX LAX Functions](https://brainx.chaobrain.com/brainunit/unit_operations/lax_functions.html) — Wraps lower-level JAX `lax` primitives with unit propagation and checking.
- [Activation Functions](https://brainx.chaobrain.com/brainunit/unit_operations/activation_functions.html) — Applies neural-network activation functions with appropriate dimensional constraints.
- [Checking Function Units](https://brainx.chaobrain.com/brainunit/unit_operations/check_units.html) — Validates the expected input and output units of functions.
- [Assigning Function Units](https://brainx.chaobrain.com/brainunit/unit_operations/assign_units.html) — Declares unit behavior for functions whose units cannot be inferred automatically.
- [JAX Transforms: JIT and vmap with Units](https://brainx.chaobrain.com/brainunit/unit_operations/jax_transforms.html) — Uses JAX compilation and vectorization with unit-aware values.
- [Automatic Differentiation with Units](https://brainx.chaobrain.com/brainunit/unit_operations/autograd.html) — Computes gradients, Jacobians, and Hessians with derived physical units.
- [Sparse Matrices with Units](https://brainx.chaobrain.com/brainunit/unit_operations/sparse_matrices.html) — Represents and operates on sparse matrices carrying physical units.
- [Matplotlib Integration](https://brainx.chaobrain.com/brainunit/unit_operations/matplotlib_integration.html) — Plots quantities with unit conversion and unit-aware labeling.
- [Unit-Aware Type Annotations](https://brainx.chaobrain.com/brainunit/unit_operations/type_annotations.html) — Expresses expected physical dimensions through Python type annotations.

- [Combining and Defining Unit](https://brainx.chaobrain.com/brainunit/advanced_tutorials/combining_and_defining.html) — Combines existing units and defines custom derived units.
- [Mechanism of Quantity](https://brainx.chaobrain.com/brainunit/advanced_tutorials/mechanism.html) — Explains the internal relationship among quantities, units, dimensions, and mantissas.
- [Unit-Aware Computation with CustomArray](https://brainx.chaobrain.com/brainunit/advanced_tutorials/custom_array.html) — Integrates custom array containers with BrainUnit operations.
- [FAQs](https://brainx.chaobrain.com/brainunit/FAQs.html) — Answers common questions about units, conversions, compatibility, and JAX behavior.

- [Release Notes](https://brainx.chaobrain.com/brainunit/apis/changelog.html) — Records BrainUnit releases, additions, fixes, and compatibility changes.
- [brainunit Module](https://brainx.chaobrain.com/brainunit/apis/brainunit.html) — Documents the main namespace containing core units, quantities, dimensions, and validation utilities.
- [Quantity API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.Quantity.html) — Documents the core class representing numerical values associated with units.
- [Unit API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.Unit.html) — Documents the class representing a physical unit, scale, dimension, and optional offset.
- [Dimension API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.Dimension.html) — Documents the class representing physical-dimensional exponents.
- [UnitMismatchError API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.UnitMismatchError.html) — Documents the exception raised for incompatible physical units.
- [DimensionMismatchError API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.DimensionMismatchError.html) — Documents the exception raised for incompatible physical dimensions.
- [BackendError API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.BackendError.html) — Documents the exception raised for invalid or unsupported backend behavior.
- [CustomArray API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.CustomArray.html) — Documents the base abstraction for custom arrays integrated with BrainUnit.

- [brainunit.typing Module](https://brainx.chaobrain.com/brainunit/apis/brainunit.typing.html) — Documents unit-aware typing definitions and validation helpers.
- [PhysicalType API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.typing.PhysicalType.html) — Documents the class representing physical dimensions in type annotations.

- [brainunit.autograd Module](https://brainx.chaobrain.com/brainunit/apis/brainunit.autograd.html) — Documents unit-aware gradient, Jacobian, and Hessian transformations.
- [brainunit.math Module](https://brainx.chaobrain.com/brainunit/apis/brainunit.math.html) — Documents unit-aware mathematical, statistical, array, and activation operations.
- [brainunit.linalg Module](https://brainx.chaobrain.com/brainunit/apis/brainunit.linalg.html) — Documents unit-aware linear algebra operations.
- [brainunit.lax Module](https://brainx.chaobrain.com/brainunit/apis/brainunit.lax.html) — Documents unit-aware wrappers around JAX `lax` primitives.
- [brainunit.fft Module](https://brainx.chaobrain.com/brainunit/apis/brainunit.fft.html) — Documents unit-aware Fourier transform functions.

- [brainunit.sparse Module](https://brainx.chaobrain.com/brainunit/apis/brainunit.sparse.html) — Documents unit-aware sparse matrix classes and conversion operations.
- [SparseMatrix API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.sparse.SparseMatrix.html) — Documents the common base class for BrainUnit sparse matrices.
- [CSR API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.sparse.CSR.html) — Documents compressed sparse row matrices carrying units.
- [CSC API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.sparse.CSC.html) — Documents compressed sparse column matrices carrying units.
- [COO API](https://brainx.chaobrain.com/brainunit/apis/generated/brainunit.sparse.COO.html) — Documents coordinate-format sparse matrices carrying units.

- [brainunit.constants Module](https://brainx.chaobrain.com/brainunit/apis/brainunit.constants.html) — Documents predefined physical constants represented as quantities.