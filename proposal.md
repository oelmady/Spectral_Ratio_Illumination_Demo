Project Proposal: Spectral-Ratio Enhanced Retinex for Intrinsic Image Decomposition

Omar Elmady
Problem Statement
Standard Retinex algorithms for intrinsic image decomposition preserve original pixel chromaticity when brightening shadowed regions, failing to account for the physics of illumination. When direct illumination changes, reflected light color changes according to the scene's spectral ratio. Current methods produce color-inaccurate decompositions, particularly visible in shadow-to-light transitions where materials appear to shift hue unnaturally.
Contribution
I will extend the recursive Retinex algorithm (McCann-Sobel) with two physics-based enhancements:

1. Spectral-ratio constrained illumination estimation: During iterative illumination estimation, I will constrain intensity changes to align with the spectral ratio direction, ensuring reflectance changes don't contaminate illumination estimates.
2. Physics-based color correction: When brightening pixels, I will adjust colors using the spectral ratio to simulate actual illumination changes rather than preserving original chromaticity.

This bridges classical Retinex with modern deep learning while respecting physical light transport.
Data Requirements
- MIT Intrinsic Images dataset
- Real-world test images with strong shadows/highlights in addition to standard HDR images
Pre-trained network provided by professor

The compute requirements are expected to be light and straightforward. I will be using Python machine learning libraries. 
Expected Deliverables

- Working implementation of base recursive Retinex
- Integration with pretrained spectral ratio network
- Spectral-ratio constrained illumination estimation (Equations 2-6 from "Retinex Done Right")
- Color correction using spectral ratios (Equations 7-10)
- Quantitative evaluation on MIT dataset (compare against baseline Retinex)
- Qualitative results showing improved color accuracy in shadow regions
