# GPyTool
![Header](./Github_GPyTool_banner_v2.png)
GPyTool is the Python implementation of GP-FBM (Gaussian Process - Fractionnal Brownian Motion), a Bayesian framework developed to estimate the diffusive properties of a stochastic time trajectory.

## Description

By estimating the correlation between two trajectories, the GP-FBM framework can correct each individual trajectory for the substrate motion, inducing effective correlations between these two trajectories. GP-FBM was originally implemented through a C++ user-friendly interface, called GPTool, which also incorporates image processing tools (channels alignment, spot localization enhancement). This Python version, called GPyTool, incorporates the GP-FBM framework alone (no image processing tool); but 3D time trajectories (TXYZ) are now handled. GPyTool can be used to process single trajectories individually or with the substrate correction.

Original GP-FBM study: Oliveira, G.M., Oravecz, A., Kobi, D. et al. Precise measurements of chromatin diffusion dynamics by modeling using Gaussian processes. Nat Commun 12, 6184 (2021). https://doi.org/10.1038/s41467-021-26466-7

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
