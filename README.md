# Detecting-Moving-Vehicles-in-Satellite-Videos-Using-Deep-Neural-Networks
This is the code for my soon-to-be published master thesis. 

**Code-cleanup and instructions are coming soon.

Aerial images captured by unmanned aerial vehicles and satellite images enable and
support tasks such as road network planning, parking condition evaluation, automatic
traffic monitoring as well as military reconnaissance. A limitation of still images is the
lack of temporal information, which is viable for tasks like estimation of average velocity,
dynamic velocity and traffic density. Satellite videos, on the other hand, are able to
capture dynamic behavior.
The detection of vehicles as small as ≈ 6.5 pixels in satellite videos is challenging due to
low ground sample distances of about 1.0 m, motion effects induced from the satellite
itself and noise. Applying out-of-the-box classifiers fail on satellite video by making
assumptions like rich texture and small to moderate ratios between image size and object
size.
Approaches utilizing the temporal consistency provided by satellite video use either frame
differencing, background subtraction or subspace methods showing moderate performance
(0.26 - 0.82 F 1 score).
In this thesis recent work on deep learning for wide-area motion imagery (FoveaNet) is
utilized and adapted for satellite video. Adaptions include modifications of the architecture
of FoveaNet as well as improved post-processing. The effects of improvements made is
demonstrated by six experiments. The resulting network is called FoveaNet4Sat. The
effect of transfer learning is also demonstrated by pre-learning on wide-area motion data
and fine-tuning on satellite video. FoveaNet4Sat consistently outperforms FoveaNet when
applied to satellite video, e.g. from 0.745 to 0.885 in F 1 score and also outperforms the
state-of-the-art on the SkySat-1 Las Vegas satellite video.

(R. Lalonde, D. Zhang, and M. Shah. ClusterNet: Detecting small objects in large
scenes by exploiting spatio-temporal information. In Proceedings of the IEEE
Computer Society Conference on Computer Vision and Pattern Recognition, pages
4003–4012, 2018.)

