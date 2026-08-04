---
pageClass: plain-doc
---

# Computer Graphics

Mastering computer graphics = writing every post below. The topic system mirrors the chapter structure of GAMES101 (Yan Lingqi) and the *Fundamentals of Computer Graphics* (the "Tiger Book"), covering everything from mathematical foundations, rasterization, geometry, and ray tracing to animation, simulation, and neural rendering.

## Topic Planning

<ProgressGrid cat="cs/computer-graphics" />


### Post 1 · Graphics Overview

- [ ] What is computer graphics: its research scope, and its relationship to computer vision / image processing
- [ ] The application landscape of graphics: games, film, visualization, CAD, and virtual reality
- [ ] Rendering pipeline overview: the overall flow from a 3D scene to a 2D image
- [ ] Color and image fundamentals: pixels, bitmaps, and a first look at color spaces

### Post 2 · Linear Algebra Review

- [ ] Vectors: the dot product and its use in lighting
- [ ] Vectors: the cross product and left/right-handed coordinate systems
- [ ] Vectors: normal vectors and orthonormal bases
- [ ] Matrices: matrix multiplication and the representation of linear transformations
- [ ] Matrices: inverse, transpose, and orthogonal matrices
- [ ] Determinants and determining the orientation of a transformation

### Post 3 · Transformations

- [ ] Basic 2D transformations: scaling, shearing, and rotation
- [ ] Homogeneous coordinates: expressing translation uniformly as matrices
- [ ] 2D composite transformations and transformation order
- [ ] 3D transformations: rotation about an arbitrary axis and Rodrigues' formula
- [ ] Model Transformation
- [ ] View Transformation: the camera frame
- [ ] Orthographic projection transformations
- [ ] Perspective projection transformations: squeezing the frustum into a canonical cube

### Post 4 · Rasterization

- [ ] Screen space and the viewport transformation
- [ ] Triangles: the basic primitive in graphics
- [ ] Sampling and aliasing: frequency-domain analysis
- [ ] Triangle rasterization: bounding boxes and the cross-product test
- [ ] Antialiasing: filter first, then sample
- [ ] MSAA and other practical antialiasing approaches
- [ ] Visibility and the depth buffer (Z-Buffer) algorithm

### Post 5 · Shading

- [ ] Shading and illumination: shading frequencies (Flat / Gouraud / Phong)
- [ ] The Blinn-Phong reflection model: the diffuse term
- [ ] The Blinn-Phong reflection model: the specular and ambient terms
- [ ] The Graphics Pipeline: from vertex shading to fragment shading
- [ ] Texture mapping: barycentric-coordinate interpolation
- [ ] Texture sampling issues: aliasing and mipmaps
- [ ] Anisotropic filtering and EWA filtering
- [ ] Applications of textures: environment maps and sphere/cube maps
- [ ] Bump Mapping
- [ ] Normal Mapping
- [ ] Displacement Mapping and 3D procedural textures

### Post 6 · Geometric Representation

- [ ] Overview of geometric representations: implicit and explicit
- [ ] Implicit representations: algebraic surfaces and CSG
- [ ] Implicit representations: distance functions and level sets
- [ ] Implicit representations: fractals and self-similarity
- [ ] Explicit representations: point clouds and polygon meshes
- [ ] Bézier curves: the de Casteljau algorithm
- [ ] Properties of Bézier curves and piecewise construction
- [ ] B-spline curves and NURBS
- [ ] Bézier surfaces
- [ ] Mesh processing: an overview of subdivision, simplification, and regularization
- [ ] Mesh subdivision: Loop subdivision
- [ ] Mesh subdivision: Catmull-Clark subdivision
- [ ] Mesh simplification: edge collapse and quadric error metrics (QEM)
- [ ] Mesh Regularization

### Post 7 · Ray Tracing

- [ ] Why ray tracing: the limitations of rasterization
- [ ] Whitted-style ray tracing: recursive ray intersection
- [ ] Ray generation: from pixels to the ray equation
- [ ] Ray–implicit-surface intersection
- [ ] Ray–triangle intersection: the Möller–Trumbore algorithm
- [ ] Acceleration structures: axis-aligned bounding boxes (AABBs) and uniform grids
- [ ] Spatial partitioning: KD-trees and octrees
- [ ] Object partitioning: bounding volume hierarchies (BVHs)
- [ ] BVH construction and traversal: SAH partitioning
- [ ] Radiometry (I): radiant flux, intensity, and irradiance
- [ ] Radiometry (II): radiance
- [ ] BRDFs and the reflection equation
- [ ] The rendering equation and its derivation
- [ ] Monte Carlo integration: estimating definite integrals
- [ ] Path tracing: from the rendering equation to sampling algorithms
- [ ] Terminating path-tracing recursion: Russian roulette
- [ ] Direct lighting sampling: importance sampling of light sources
- [ ] Sampling theory: importance sampling and multiple importance sampling

### Post 8 · Materials & Appearance

- [ ] Material as BRDF: diffuse, mirror, and refractive materials
- [ ] Perfect specular reflection and refraction: Snell's law and the Fresnel term
- [ ] Microfacet models: the Cook-Torrance BRDF
- [ ] The components of microfacet models: normal distribution, geometric shadowing, and Fresnel
- [ ] Anisotropic materials and fiber appearance

### Post 9 · Animation & Simulation

- [ ] Animation fundamentals: keyframes and interpolation
- [ ] Skeletal animation: forward kinematics and inverse kinematics
- [ ] Skinning and blend shapes
- [ ] Simulation basics: single particles and explicit Euler integration
- [ ] Stability of numerical integration: semi-implicit Euler, RK4, and implicit methods
- [ ] Mass-spring systems: cloth simulation
- [ ] Rigid-body simulation and collision detection
- [ ] Fluid simulation basics: grid-based (Eulerian) methods
- [ ] Fluid simulation basics: particle-based (Lagrangian) methods and SPH

### Post 10 · Real-Time Rendering

- [ ] Overview of the real-time rendering pipeline and GPU architecture
- [ ] Shadow Mapping: principles and aliasing
- [ ] Shadow mapping improvements: PCF, CSM, and VSM
- [ ] Ambient occlusion: SSAO and HBAO
- [ ] Comparing deferred rendering and forward rendering
- [ ] Tone mapping and the HDR pipeline
- [ ] Real-time global illumination: an overview of RSM, LPV, and VXGI
- [ ] Image-based lighting (IBL) and pre-filtered environment maps

### Post 11 · Modern Topics

- [ ] Deep dive into the GPU rendering pipeline: programmable shaders and GLSL/HLSL
- [ ] GPU parallel computing: CUDA / Compute Shaders and graphics acceleration
- [ ] Differentiable Rendering: principles and frameworks
- [ ] Inverse rendering: reconstructing geometry, materials, and lighting from images
- [ ] Neural radiance fields (NeRF): volume rendering and positional encoding
- [ ] Accelerating and improving NeRF: Instant-NGP, Mip-NeRF
- [ ] 3D Gaussian Splatting: real-time novel-view synthesis
- [ ] Merging neural rendering with the traditional pipeline: trends and outlook

> After finishing a post: create a new `xxx.md` in this directory, then change the corresponding entry above to `- [x] [标题](./xxx)`.
