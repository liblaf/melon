# Changelog

## [0.5.0](https://github.com/liblaf/melon/compare/v0.4.2...v0.5.0) (2025-07-22)


### ⚠ BREAKING CHANGES

* **melon:** rename triangle to tri and reorganize modules
* Removed pixi configuration and bumped liblaf-grapes to major version 1.0. Projects using pixi will need to migrate to uv.

### 🎫 Chores

* modernize tooling and dependencies ([79e12fe](https://github.com/liblaf/melon/commit/79e12feed02d60b4031f77f71814b26bcc683cdf))


### ♻ Code Refactoring

* **melon:** rename triangle to tri and reorganize modules ([817726f](https://github.com/liblaf/melon/commit/817726ff754f561123a74d608485466505bdd6b5))

## [0.4.2](https://github.com/liblaf/melon/compare/v0.4.1..v0.4.2) - 2025-07-15

### ⬆️ Dependencies

- **deps:** update dependencies and config - ([71b9cd1](https://github.com/liblaf/melon/commit/71b9cd1a65e429e1a98271a61137b5aa0a6d386a))
- **deps:** update dependency liblaf-grapes to >=0.4.1,<0.5 (#49) - ([dbafa14](https://github.com/liblaf/melon/commit/dbafa141a025783b80f6987015265b1870f07eb5))

### ❤️ New Contributors

- [@liblaf](https://github.com/liblaf) made their first contribution
- [@renovate[bot]](https://github.com/apps/renovate) made their first contribution in [#49](https://github.com/liblaf/melon/pull/49)
- [@liblaf-bot[bot]](https://github.com/apps/liblaf-bot) made their first contribution

## [0.4.1](https://github.com/liblaf/melon/compare/v0.4.0..v0.4.1) - 2025-06-23

### ⬆️ Dependencies

- **deps:** loosen dependency version constraints - ([5a9fe99](https://github.com/liblaf/melon/commit/5a9fe9959d185c4e4c07e29aeb54072af6143ff6))

## [0.4.0](https://github.com/liblaf/melon/compare/v0.3.0..v0.4.0) - 2025-06-23

### 💥 BREAKING CHANGES

- **deps:** update dependencies and documentation tooling - ([137ecff](https://github.com/liblaf/melon/commit/137ecffd8d7ab7b8909d3520d73e9e7ec6f5bcfd))

## [0.3.0](https://github.com/liblaf/melon/compare/v0.2.8..v0.3.0) - 2025-06-11

### 💥 BREAKING CHANGES

- **cli:** add annotate-landmarks command and refactor entry points - ([f441374](https://github.com/liblaf/melon/commit/f44137416769e56de24535d5f2d08ff5ebde1db9))

### ⬆️ Dependencies

- **deps:** update dependency liblaf-grapes to >=0.2.1,<0.3 (#45) - ([d717ae6](https://github.com/liblaf/melon/commit/d717ae63b1bf957fdd2d332660a7c8b290e537e0))

## [0.2.8](https://github.com/liblaf/melon/compare/v0.2.7..v0.2.8) - 2025-05-28

### ✨ Features

- **plugin:** add joblib caching to tetwild function - ([8ec8241](https://github.com/liblaf/melon/commit/8ec8241f2d17be1f98797e81bdadb7efd1d2cfb4))

## [0.2.7](https://github.com/liblaf/melon/compare/v0.2.6..v0.2.7) - 2025-05-27

### ✨ Features

- **io:** enhance Paraview series writer and pyvista converters - ([72488cc](https://github.com/liblaf/melon/commit/72488cc7e565faf935f69ad0f87dc9d660cac9fc))

## [0.2.6](https://github.com/liblaf/melon/compare/v0.2.5..v0.2.6) - 2025-05-25

### ✨ Features

- **cli:** add new CLI framework with cyclopts integration - ([99d7983](https://github.com/liblaf/melon/commit/99d79831e2d76d6286abe99dce2c854ae2e1407d))

## [0.2.5](https://github.com/liblaf/melon/compare/v0.2.4..v0.2.5) - 2025-05-23

### 👷 Build System

- update dependencies in pyproject.toml - ([b872515](https://github.com/liblaf/melon/commit/b87251569f2a8090ea51f0f16420a2eecbd7efa6))
- migrate from Just to Mise for task management - ([354d93a](https://github.com/liblaf/melon/commit/354d93aa02ee8d60405b5970e0bf867f455e4901))

## [0.2.4](https://github.com/liblaf/melon/compare/v0.2.3..v0.2.4) - 2025-05-13

### 🐛 Bug Fixes

- **io/paraview:** remove debug logging in SeriesWriter - ([bb1a52f](https://github.com/liblaf/melon/commit/bb1a52f436fbdfdcdef8fd5e21b83efe41dea6fb))
- trigger release-please - ([80df7b5](https://github.com/liblaf/melon/commit/80df7b5516b0be5f668e3cc328b6fea1b07be126))

## [0.2.3](https://github.com/liblaf/melon/compare/v0.2.2..v0.2.3) - 2025-05-13

### ✨ Features

- **io:** add ParaView PVD and Series writers - ([3cd1ed8](https://github.com/liblaf/melon/commit/3cd1ed87abae3f99b05cc4d051413fd7fe25c6fe))

## [0.2.2](https://github.com/liblaf/melon/compare/v0.2.1..v0.2.2) - 2025-05-13

### ⬆️ Dependencies

- **deps:** update dependency python to >=3.13,<3.14 (#36) - ([73c0c94](https://github.com/liblaf/melon/commit/73c0c94d3c550501017e68935b3dffd9e532e979))
- **deps:** update dependency python to >=3.13,<3.14 (#21) - ([1f27494](https://github.com/liblaf/melon/commit/1f27494bf17c402b5c8aebd11fea37a8bb237d91))

## [0.2.1](https://github.com/liblaf/melon/compare/v0.2.0..v0.2.1) - 2025-05-11

### 🐛 Bug Fixes

- update tinyobjloader dependency to v2.0.0rc13 - ([7a93acb](https://github.com/liblaf/melon/commit/7a93acb77b45f9e2c7775bc132b7d775f7eaa5ab))
- adjust tetgen parameters and scipy version constraint - ([3ec2c11](https://github.com/liblaf/melon/commit/3ec2c11239540f42d7701fde18515299f872a97b))

## [0.2.0](https://github.com/liblaf/melon/compare/v0.1.17..v0.2.0) - 2025-05-11

### 💥 BREAKING CHANGES

- next (#32) - ([ba54e21](https://github.com/liblaf/melon/commit/ba54e212802e04d56d79904316bce0015ab818f1))

## [0.1.17](https://github.com/liblaf/melon/compare/v0.1.16..v0.1.17) - 2025-05-08

### ⬆️ Dependencies

- **deps:** update dependency liblaf-cherries to >=0.0.12,<0.0.13 (#29) - ([9c4d022](https://github.com/liblaf/melon/commit/9c4d022e24438aaef25a3f88583e0a06f606fd2d))

## [0.1.16](https://github.com/liblaf/melon/compare/v0.1.15..v0.1.16) - 2025-05-06

### ✨ Features

- add tetgen experiment pipeline and array to PointSet converter - ([ac3a353](https://github.com/liblaf/melon/commit/ac3a3534c8691c899f75fe6e8a163092d33f1b2a))

## [0.1.15](https://github.com/liblaf/melon/compare/v0.1.14..v0.1.15) - 2025-04-30

### ✨ Features

- improve skull mesh generation and add cranium classification - ([a82c73f](https://github.com/liblaf/melon/commit/a82c73fa80d1df2819b63cd3fe48a1c8a44ad98d))

## [0.1.14](https://github.com/liblaf/melon/compare/v0.1.13..v0.1.14) - 2025-04-24

### ✨ Features

- add complete human head anatomy processing pipeline - ([894d5ca](https://github.com/liblaf/melon/commit/894d5cadc442c1140fb5d6fc4d7f73a1bcfc31fb))

## [0.1.13](https://github.com/liblaf/melon/compare/v0.1.12..v0.1.13) - 2025-04-21

### ⬆️ Dependencies

- **deps:** update dependency pyvista to >=0.45,<0.46 (#24) - ([8c7f39d](https://github.com/liblaf/melon/commit/8c7f39d3141e47b393e87c9487f5205701afc4e8))

## [0.1.12](https://github.com/liblaf/melon/compare/v0.1.11..v0.1.12) - 2025-04-13

### ♻ Code Refactoring

- improve tetgen-clean script organization and flexibility - ([e5d5e95](https://github.com/liblaf/melon/commit/e5d5e954e6f1434dfd1975dda66db31118d437f4))

## [0.1.11](https://github.com/liblaf/melon/compare/v0.1.10..v0.1.11) - 2025-04-09

### ✨ Features

- **paraview:** make visualization optional in mandible pivot script - ([ca24c33](https://github.com/liblaf/melon/commit/ca24c33fa329ae71ad0129f0d231520263aae61a))

## [0.1.10](https://github.com/liblaf/melon/compare/v0.1.9..v0.1.10) - 2025-04-09

### ✨ Features

- **triangle:** add triangle mesh processing utilities - ([e389514](https://github.com/liblaf/melon/commit/e389514510fa49416401b6dcd61932c22426c168))

### ♻ Code Refactoring

- **io:** reorganize dispatcher module into abc and improve type hints - ([ab5dcbf](https://github.com/liblaf/melon/commit/ab5dcbf59a3a8365f1def093a1558bb3b6f14d24))
- **io/pyvista:** replace custom OBJ parser with tinyobjloader - ([945f48a](https://github.com/liblaf/melon/commit/945f48ad8a78f6f9076abb66d5b854dcc4151173))

## [0.1.9](https://github.com/liblaf/melon/compare/v0.1.8..v0.1.9) - 2025-04-01

### 🐛 Bug Fixes

- **io:** replace StrPath with PathLike type alias - ([c157934](https://github.com/liblaf/melon/commit/c157934db1e9a32a9eeb13f31635e85b3b06a3d9))

## [0.1.8](https://github.com/liblaf/melon/compare/v0.1.7..v0.1.8) - 2025-04-01

### ✨ Features

- **io:** add ParaView PVD writer support - ([ce4e628](https://github.com/liblaf/melon/commit/ce4e62809e57758841382f01f94f15499d1332b1))

## [0.1.7](https://github.com/liblaf/melon/compare/v0.1.6..v0.1.7) - 2025-03-31

### ⬆️ Dependencies

- **deps:** update dependency rich to v14 (#15) - ([a30c487](https://github.com/liblaf/melon/commit/a30c487f6711c7f3b01e9ee99db8fbc3b97c1b2f))

### 👷 Build System

- move optional dependencies to extras - ([f090b3d](https://github.com/liblaf/melon/commit/f090b3d9d4aa9975a7ea27591d0ef47216c5f20e))
- update dependency versions and add tetra module - ([43d50d7](https://github.com/liblaf/melon/commit/43d50d7a111614b178a55ca6e0ed42274e2dfe8d))

## [0.1.6](https://github.com/liblaf/melon/compare/v0.1.5..v0.1.6) - 2025-03-29

### ✨ Features

- **plugin:** add tetwild plugin for tetrahedral mesh generation - ([918f47e](https://github.com/liblaf/melon/commit/918f47ec2ad46cdb6e408413e5a94fd3a1059358))
- **tetra:** add tetrahedral mesh winding correction - ([db23780](https://github.com/liblaf/melon/commit/db237807966fe512ecf4b548a40cc3622bb6d3ee))

### 👷 Build System

- reorganize config files and update workflows - ([df320f5](https://github.com/liblaf/melon/commit/df320f5c467641e7a797410704ceeb56837726b5))

### ❤️ New Contributors

- [@liblaf-bot[bot]](https://github.com/apps/liblaf-bot) made their first contribution in [#14](https://github.com/liblaf/melon/pull/14)

## [0.1.5](https://github.com/liblaf/melon/compare/v0.1.4..v0.1.5) - 2025-03-23

### ✨ Features

- enhance mesh processing and OBJ file loading - ([4dd9c2a](https://github.com/liblaf/melon/commit/4dd9c2a8f9189a63ef55b74f3062547e38143d34))

### ⬆️ Dependencies

- **deps:** update dependency liblaf-cherries to v0.0.6 (#11) - ([aada5b5](https://github.com/liblaf/melon/commit/aada5b55c4463f9b2afcaa3cc6df608a55d25704))
- **deps:** update dependency jaxtyping to >=0.3,<0.4 (#12) - ([0b6529e](https://github.com/liblaf/melon/commit/0b6529ecaf694256bdc568b38e9bf709b46827e6))

### ♻ Code Refactoring

- update muscle generation logic and point data - ([4027476](https://github.com/liblaf/melon/commit/40274768a815f8183c923dae16bb196de980c461))

## [0.1.4](https://github.com/liblaf/melon/compare/v0.1.3..v0.1.4) - 2025-03-12

### ✨ Features

- **io:** add support for unstructured grid file formats - ([60929fc](https://github.com/liblaf/melon/commit/60929fc50f189e3060e3da3faafc4618d6c369b9))
- refactor scripts to use liblaf-cherries for configuration management - ([53e13b6](https://github.com/liblaf/melon/commit/53e13b6bf643ded9f5b2c6e8bac14cefaaaba64f))
- refactor proximity algorithms and add component transfer functionality - ([d09433a](https://github.com/liblaf/melon/commit/d09433a6afb2af178b6c1f1a263a8e9d5c3db4df))
- enhance file handling and add rigid alignment for face models - ([acdeab3](https://github.com/liblaf/melon/commit/acdeab341ee450f4b234edb516db645139bc61a5))

## [0.1.3](https://github.com/liblaf/melon/compare/v0.1.2..v0.1.3) - 2025-02-16

### ⬆️ Dependencies

- **deps:** update liblaf-grapes dependency to v0.1.0 - ([d8579ac](https://github.com/liblaf/melon/commit/d8579ac3fe26a44358c5a234c3053c02f26cefaf))

## [0.1.2](https://github.com/liblaf/melon/compare/v0.1.1..v0.1.2) - 2025-02-16

### ✨ Features

- add support for unstructured grid operations and documentation link - ([aa118c7](https://github.com/liblaf/melon/commit/aa118c7857cf783b75377b6b20f1e2e94ae3f787))

## [0.1.1](https://github.com/liblaf/melon/compare/v0.1.0..v0.1.1) - 2025-02-14

### ✨ Features

- **io:** add unstructured grid support for PyVista - ([0bb7bbe](https://github.com/liblaf/melon/commit/0bb7bbee0e5da312aea3930e83139df0dccb8a21))
- add fast wrapping and rigid alignment pipelines - ([b3b7434](https://github.com/liblaf/melon/commit/b3b7434f70da9800440eda23186bf676f8646106))

### ⬆️ Dependencies

- **deps:** update dependency liblaf-grapes to >=0.0.4,<0.0.5 (#3) - ([5c8c9ff](https://github.com/liblaf/melon/commit/5c8c9ffe547c5098ecd447dd66666119b1a4dc9d))

### 👷 Build System

- update build dependencies and configuration - ([65f966e](https://github.com/liblaf/melon/commit/65f966e2f46e90937c505d7ca078d72551a267e6))

### ❤️ New Contributors

- [@renovate[bot]](https://github.com/apps/renovate) made their first contribution in [#3](https://github.com/liblaf/melon/pull/3)

## [0.1.0] - 2025-02-09

### ✨ Features

- **docs:** add mkdocs configuration and documentation setup - ([a42bdfa](https://github.com/liblaf/melon/commit/a42bdfadbfd0d190274a84a3bb15a29eaf74f043))
- **io:** refactor IO dispatchers and add reader/writer support - ([3984800](https://github.com/liblaf/melon/commit/398480041672f0f15ff2818b585291e05d0ba6b7))
- **melon:** add DICOM and PyVista IO modules with structured data handling - ([c8fba2d](https://github.com/liblaf/melon/commit/c8fba2dd60caf47e3e46c77d4f4efc590ff94202))
- add CT data collection and surface extraction scripts - ([cc98a1a](https://github.com/liblaf/melon/commit/cc98a1a89497a24bfdd7b53da3f7ad2485f7f730))
- enhance correspondence and transfer operations with new algorithms and transformations - ([ebc40ec](https://github.com/liblaf/melon/commit/ebc40eca918537d911c24f4693ef6473afb1560b))
- add point set and trimesh support with correspondence operations - ([6532af8](https://github.com/liblaf/melon/commit/6532af8fd488c44c4db17cfe59add111fb9d224d))
- add image data operations and enhance DICOM dataset handling - ([294f748](https://github.com/liblaf/melon/commit/294f748183b8b96bbeecbf56342ead028b48f67e))
- enhance DICOM dataset handling and add meta classes - ([a3b1377](https://github.com/liblaf/melon/commit/a3b13777daf1d83d57b55e73090a1f3bbadd2c14))

### ♻ Code Refactoring

- reorganize DICOM dataset and pyvista operations - ([329b2e5](https://github.com/liblaf/melon/commit/329b2e53d5ee3a0cf819ed0a1c4a01616b23b8e5))
- rename `Patient` to `Subject` and simplify DICOM dataset structure - ([e7174af](https://github.com/liblaf/melon/commit/e7174afe75e6ac2b4e693ba427285b7e44d942c0))
- rename project from pineapple to melon - ([ae040b0](https://github.com/liblaf/melon/commit/ae040b02ff7033d1390531e9c3ab363c0179d6c6))

### 👷 Build System

- initialize project structure and configuration - ([6f6b0ae](https://github.com/liblaf/melon/commit/6f6b0ae3d28d0cd172c9c1c6827aaa49c5c38b01))

### ❤️ New Contributors

- [@github-actions[bot]](https://github.com/apps/github-actions) made their first contribution in [#2](https://github.com/liblaf/melon/pull/2)
- [@liblaf](https://github.com/liblaf) made their first contribution
