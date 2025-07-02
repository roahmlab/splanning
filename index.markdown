---
# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "SPLANNING"
date:   2024-09-13 10:00:00 -0500
description: >- # Supports markdown
  Risk-Aware Trajectory Optimization in a Normalized Gaussian Splat
show-description: true

# Add page-specifi mathjax functionality. Manage global setting in _config.yml
mathjax: true
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: false

# Preview image for social media cards
image:
  path: /assets/splanning_compressed.jpg
  height: 600
  width: 800
  alt: SPLANNING Main Figure - Risk Aware Planning in a Real World Splat

# Only the first author is supported by twitter metadata
authors:
  - name: Jonathan Michaux
    email: jmichaux@umich.edu
    footnotes: "*"
  - name: Seth Isaacson
    email: sethgi@umich.edu
    footnotes: "*"
  - name: Challen Enninful Adu
    email: enninful@umich.edu
  - name: Adam Li
    email: adamli@umich.edu
  - name: Rahul Kashyap Swayampakula
    email: rahulswa@umich.edu
  - name: Parker Ewen
    email: pewen@umich.edu
  - name: Sean Rice
    email: seanrice@umich.edu
  - name: Katherine A. Skinner
    email: kskin@umich.edu
  - name: Ram Vasudevan
    email: ramv@umich.edu

# If you just want a general footnote, you can do that too.
# See the sel_map and armour-dev examples.
author-footnotes: |
  \* Denotes equal contribution.
  <br> All authors affiliated with the department of Department of Robotics of the University of Michigan, Ann Arbor.

links:
  - icon: arxiv
    icon-library: simpleicons
    text: ArXiv
    url: https://arxiv.org/abs/2409.16915
  - icon: bi-file-earmark-text-fill
    icon-library: bootstrap-icons
    text: Math Supplement
    url: https://drive.google.com/file/d/1OVRNBCspSVJt1fQErTBAhxzfQSrUavcK/view
  - icon: github
    icon-library: simpleicons
    text: Code (Coming Soon!)
    # url: https://github.com/roahmlab/splanning

# End Front Matter
---

{% include sections/authors %}
{% include sections/links %}

---

# [Overview Videos](#overview-videos)

<!-- BEGIN OVERVIEW VIDEOS -->
<div class="fullwidth">
  {% include util/video
    content="assets/mainvid_compressed.mp4"
    poster="assets/thumb/mainvid_compressed.jpg"
    autoplay-on-load=true
    preload="none"
    muted=true
    loop=true
    playsinline=true
    %}
  <p style="text-align:center; font-weight:bold;">SPLANNING side-by-side real world</p>
</div><!-- END OVERVIEW VIDEOS -->

<!-- BEGIN ABSTRACT -->
<div markdown="1" class="content-block justify grey">

# [Abstract](#abstract)

Neural Radiance Fields and Gaussian Splatting have recently transformed the field of computer vision.
By estimating a scene's radiance field, these techniques enable photo-realistic representation of complex scenes.
Despite this success, they have seen only limited use in real-world robotics tasks such as motion planning.
Two key factors have contributed to this limited success.
First, it is challenging to reason about collisions in radiance models.
Second, it is difficult to perform inference of radiance models fast enough for real-time motion planning.
This paper addresses these challenges by proposing SPLANNING, a risk-aware trajectory planner that operates in a Gaussian Splatting model. 
This paper first derives a method for rigorously upper-bounding the probability of collision between the robot and a radiance field.
Second, this paper introduces a normalized reformulation of Gaussian Splatting that allows for the efficient computation of the collision bound in a Gaussian Splat.
Third, a method is presented to optimize trajectories while avoiding collisions with a scene represented by a Gaussian Splat.
Experiments demonstrate that SPLANNING outperforms state-of-the-art methods in generating collision-free trajectories in highly cluttered environments.
The proposed system is also tested on a real-world robot manipulator.

</div> <!-- END ABSTRACT -->

<!-- BEGIN APPROACH -->
<div markdown="1" class="justify">

# [Approach](#approach)

![method_overview](./assets/fig2_compressed.webp)
{: class="fullwidth no-pre"}

<!-- # Contributions -->
This paper proposed SPLANNING, a real-time, receding-horizon motion planning algorithm.
SPLANNING combines a simple and tight sphere-based geometric representation of the reachability of a robot with radiance field representations of an environment scene to enable risk-aware planning through a rendering-based probability bound.
This differs from existing approaches such as CATNIPS and SplatNav in that planning can be performed with probabilistic bounds and directly in a scene with a normalized splatting representation without additional preprocessing passes, opening the door for real-time splatting and risk-aware planning.

![collision_comparison](./assets/collision_comparison.png)
{: class="fullwidth"}

Prior to planning, a normalized Gaussian Splatting representation of the scene is constructed.
At every planning iteration, the robot is given \\(t_{p} \leq 0.5\\) seconds to find a feasible trajectory by solving

$$
\newcommand{\qlim}{q_{j,\mathrm{lim}}}
\newcommand{\dqlim}{\dot{q}_{j,\mathrm{lim}}}
\newcommand{\bound}{\mathcal{H}}
  \begin{align}
\underset{k \in K}{\min}& \quad \mathrm{\texttt{cost}}(k) \\
% \text{s.t.}& \quad q_j(T_i; k) \subseteq [\qlim^-, \qlim^+] \quad\quad\quad \forall (i,j) \in N_t \times N_q \label{eq:pz_optpos} \\
% & \quad \dot{q}_j(T_i; k) \subseteq [\dqlim^-, \dqlim^+]  \quad\quad\quad \forall (i,j) \in N_t \times N_q \label{eq:pz_optvel}\\
\text{s.t.}&\quad \texttt{Prob}(\texttt{collision}) < \beta 
\end{align}
$$

The paper, coming soon, details the derivation of the bound on the probability of collision.

Our key contributions are:
1. A derivation of a bound on the probability that a 3D body collides with a scene represented by a radiance field, starting directly from the rendering equation;
2. A method to efficiently compute this collision bound in a Gaussian Splatting model;
3. A re-formulation of Gaussian Splatting that normalizes the 3D Gaussians to ensure the correctness of the collision probabilities;
4. A novel risk-aware trajectory planner for robot manipulators, with experiments that show the resulting risk-aware planner solves challenging tasks in simulation and on hardware in real-time.


</div><!-- END METHOD -->

<!-- START RESULTS -->
<div markdown="1" class="content-block grey justify">

# [Results](#results)

## [Simulation Results](#simulation-results)
The following videos demonstrate the performance of SPLANNING in simulated worlds with increasing amounts of clutter.
<!-- The following videos demonstrate the performance of SPLANNING to other methods in randomly generated scenarios. -->
<!-- In each of these, SPLANNING achieves the desired goal configuration in less steps, accounting for risk. -->
<!-- SPARROWS does acheive the goal or stop in a safe configuration, but it is more slower and conservative.
On the other hand, MPOT and TRAJOPT both stop due to colliding with the environment. -->

<!-- START SIMULATION VIDEOS -->
<div class="multicontent-container tighter">
  <div class="multicontent-item">
    {% include util/video
      content="assets/10obs.mp4"
      poster="assets/thumb/10obs.jpg"
      autoplay-in-frame=true
      autoplay-on-load=false
      preload="none"
      hide_controls=true
      picture_in_picture=false
      muted=true
      loop=true
      playsinline=true
      pause_on_click=true
      %}
    <p>10 obstacles</p>
  </div>
  <div class="multicontent-item">
    {% include util/video
      content="assets/20obs.mp4"
      poster="assets/thumb/20obs.jpg"
      autoplay-in-frame=true
      preload="none"
      hide_controls=true
      muted=true
      loop=true
      playsinline=true
      pause_on_click=true
      %}
    <p>20 obstacles</p>
  </div>
  <div class="multicontent-item">
    {% include util/video
      content="assets/40obs.mp4"
      poster="assets/thumb/40obs.jpg"
      autoplay-in-frame=true
      preload="none"
      hide_controls=true
      muted=true
      loop=true
      playsinline=true
      pause_on_click=true
      %}
    <p>40 obstacles</p>
  </div>
</div><!-- END SIMULATION VIDEOS -->

## [Hardware Results](#hardware-results)

We also demonstrate success with other hardware configurations.
<!-- START HARDWARE VIDEOS -->
<div class="multicontent-container">
  <div class="multicontent-item">
    {% include util/video
      content="assets/ikea.mp4"
      poster="assets/thumb/ikea.jpg"
      autoplay-in-frame=true
      preload="none"
      hide_controls=true
      muted=true
      loop=true
      playsinline=true
      pause_on_click=true
      %}
  </div>
  <div class="multicontent-item">
    {% include util/video
      content="assets/shelves.mp4"
      poster="assets/thumb/shelves.jpg"
      autoplay-in-frame=true
      preload="none"
      hide_controls=true
      muted=true
      loop=true
      playsinline=true
      pause_on_click=true
      %}
  </div>
</div><!-- END HARDWARE VIDEOS -->
</div><!-- END RESULTS -->

<!-- START RELATED PROJECTS -->
<!-- <div markdown="1" class="justify">

# [Related Projects](#related-projects)
  
* [Autonomous Robust Manipulation via Optimization with Uncertainty-aware Reachability](https://roahmlab.github.io/armour/)
* [Reachability-based Trajectory Design with Neural Implicit Safety Constraints](https://roahmlab.github.io/RDF/)
* [Reachability-based Trajectory Design via Exact Formulation of Implicit Neural Signed Distance Functions](https://roahmlab.github.io/redefined/)
* [Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres](https://roahmlab.github.io/sparrows/)

</div> -->
 <!-- END RELATED PROJECTS -->

<!-- START CITATION -->
<div markdown="1" class="content-block grey justify">
 
# [Citation](#citation)

This project was developed in [Robotics and Optimization for Analysis of Human Motion (ROAHM) Lab](http://www.roahmlab.com/) at the University of Michigan - Ann Arbor.

```bibtex
@article{michauxisaacson2024splanning,
  author={Michaux, Jonathan and Isaacson, Seth and Adu, Challen Enninful and Li, Adam and Swayampakula, Rahul Kashyap and Ewen, Parker and Rice, Sean and Skinner, Katherine A. and Vasudevan, Ram},
  journal={IEEE Transactions on Robotics}, 
  title={Let's Make a Splan: Risk-Aware Trajectory Optimization in a Normalized Gaussian Splat}, 
  year={2025},
  volume={},
  number={},
  pages={1-19},
  keywords={Robots;Collision avoidance;Three-dimensional displays;Planning;Neural radiance field;Trajectory optimization;Computational modeling;Geometry;Real-time systems;Point cloud compression;3D gaussian splatting;collision avoidance;motion and path planning},
  doi={10.1109/TRO.2025.3584559}}

```
</div>
<!-- END CITATION -->

<!-- below are some special scripts -->
<script>
window.addEventListener("load", function() {
  // Get all video elements and auto pause/play them depending on how in frame or not they are
  let videos = document.querySelectorAll('.autoplay-in-frame');

  // Create an IntersectionObserver instance for each video
  videos.forEach(video => {
    const observer = new IntersectionObserver(entries => {
      const isVisible = entries[0].isIntersecting;
      if (isVisible && video.paused) {
        video.play();
      } else if (!isVisible && !video.paused) {
        video.pause();
      }
    }, { threshold: 0.25 });

    observer.observe(video);
  });

  // document.addEventListener("DOMContentLoaded", function() {
  videos = document.querySelectorAll('.autoplay-on-load');

  videos.forEach(video => {
    video.play();
  });
});
</script>
