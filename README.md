# Active Contour Re-Implemented (Model based segmentation - Snake)

Implement an active-contour segmentation approach on your own. Thereby, the fitness function
necessitates (a) the attraction field derived from input image. Furthermore, the intrinsic model
parameters should be (b) “min path length” and/or (c) maximized sphericity of the contour. For
optimization, own approaches to move one or several contour control points are requested. The
contour is constructed of control points and spline
interpolation for intermediate positions. The seed
position can be provided in a manual way.
Evaluate the influence of the seed positions on the
achievable quality of results. Test against circular
structures, e.g. coins. Intensively steer and evaluate the balance between extrinsic and intrinsic
fitness. Evaluate strategies to adapt the number of control points at shrinking or growing shape size. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them
